use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use polars_core::frame::DataFrame;
use polars_error::{PolarsError, PolarsResult, polars_err};
use polars_plan::prelude::PlanCallback;
use tokio::sync::mpsc;

/// Shared stop/error state between the sink and the callback worker.
#[derive(Debug, Default)]
struct PrefetchState {
    stop: AtomicBool,
    error: Mutex<Option<PolarsError>>,
}

impl PrefetchState {
    fn is_stopped(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    fn take_error(&self) -> Option<PolarsError> {
        self.error.lock().unwrap().take()
    }

    fn set_error(&self, error: PolarsError) {
        let mut slot = self.error.lock().unwrap();
        if slot.is_none() {
            *slot = Some(error);
        }
        self.stop.store(true, Ordering::Release);
    }
}

/// Cloneable handle the sink uses to enqueue batches.
#[derive(Clone)]
pub(super) struct PrefetchSender {
    tx: mpsc::Sender<DataFrame>,
    state: Arc<PrefetchState>,
}

impl PrefetchSender {
    pub(super) async fn send(&self, df: DataFrame) -> PolarsResult<bool> {
        if let Some(error) = self.state.take_error() {
            return Err(error);
        }
        if self.state.is_stopped() {
            return Ok(true);
        }

        if self.tx.send(df).await.is_err() {
            return match self.state.take_error() {
                Some(error) => Err(error),
                None => Ok(true),
            };
        }

        match self.state.take_error() {
            Some(error) => Err(error),
            None => Ok(self.state.is_stopped()),
        }
    }
}

/// Runs the batch callback on a dedicated thread with a bounded queue.
///
/// The sink keeps producing while the callback processes the current batch;
/// peak memory grows by at most `capacity` batches. The worker keeps draining
/// after a stop or an error so that a sender blocked on a full queue is always
/// released.
pub(super) struct PrefetchWorker {
    tx: Option<mpsc::Sender<DataFrame>>,
    state: Arc<PrefetchState>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl PrefetchWorker {
    pub(super) fn new(
        capacity: NonZeroUsize,
        function: PlanCallback<DataFrame, bool>,
    ) -> PolarsResult<Self> {
        let (tx, mut rx) = mpsc::channel(capacity.get());
        let state = Arc::new(PrefetchState::default());
        let worker_state = state.clone();

        let handle = std::thread::Builder::new()
            .name("polars-batch-sink".into())
            .spawn(move || {
                while let Some(df) = rx.blocking_recv() {
                    if worker_state.is_stopped() {
                        drop(df);
                        continue;
                    }

                    match function.call(df) {
                        Ok(true) => worker_state.stop.store(true, Ordering::Release),
                        Ok(false) => {},
                        Err(error) => worker_state.set_error(error),
                    }
                }
            })
            .map_err(
                |error| polars_err!(ComputeError: "failed to spawn batch sink worker: {}", error),
            )?;

        Ok(Self {
            tx: Some(tx),
            state,
            handle: Some(handle),
        })
    }

    pub(super) fn sender(&self) -> PrefetchSender {
        PrefetchSender {
            tx: self.tx.as_ref().expect("batch sink queue is open").clone(),
            state: self.state.clone(),
        }
    }

    /// Close the queue and wait for the callback worker to finish.
    pub(super) fn finish(&mut self) -> PolarsResult<()> {
        self.tx.take();

        if let Some(handle) = self.handle.take() {
            handle.join().map_err(|payload| {
                let message = payload
                    .downcast_ref::<&str>()
                    .copied()
                    .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                    .unwrap_or("unknown panic");
                polars_err!(ComputeError: "batch sink worker panicked: {}", message)
            })?;
        }

        match self.state.take_error() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

impl Drop for PrefetchWorker {
    fn drop(&mut self) {
        // Closing the queue lets the worker exit once all senders are gone.
        // Do not join here: on an error path the callback may still be
        // running, and `finish` is where the sink waits for it.
        self.tx.take();
    }
}
