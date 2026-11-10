use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use polars_async::executor::{JoinHandle, TaskPriority, TaskScope};
use polars_core::frame::DataFrame;
use polars_core::runtime::ASYNC;
use polars_error::{PolarsError, PolarsResult, polars_err};
use polars_ops::frame::{first_run_len, split_by_key};
use polars_plan::prelude::PlanCallback;
use polars_utils::pl_str::PlSmallStr;
use tokio::sync::mpsc;

use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::nodes::ComputeNode;
use crate::pipe::{RecvPort, SendPort};

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

/// Delivers a batch to the direct callback or to the prefetch worker.
#[derive(Clone)]
enum SinkSender {
    Direct(PlanCallback<DataFrame, bool>),
    Prefetch(PrefetchSender),
}

impl SinkSender {
    async fn send(&self, df: DataFrame) -> PolarsResult<bool> {
        match self {
            Self::Direct(function) => {
                let function = function.clone();
                ASYNC
                    .spawn_blocking(move || function.call(df))
                    .await
                    .unwrap()
            },
            Self::Prefetch(sender) => sender.send(df).await,
        }
    }
}

/// Cloneable handle the sink uses to enqueue batches.
#[derive(Clone)]
struct PrefetchSender {
    tx: mpsc::Sender<DataFrame>,
    state: Arc<PrefetchState>,
}

impl PrefetchSender {
    async fn send(&self, df: DataFrame) -> PolarsResult<bool> {
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
struct PrefetchWorker {
    tx: Option<mpsc::Sender<DataFrame>>,
    state: Arc<PrefetchState>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl PrefetchWorker {
    fn new(capacity: NonZeroUsize, function: PlanCallback<DataFrame, bool>) -> PolarsResult<Self> {
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

    fn sender(&self) -> PrefetchSender {
        PrefetchSender {
            tx: self.tx.as_ref().expect("batch sink queue is open").clone(),
            state: self.state.clone(),
        }
    }

    /// Close the queue and wait for the callback worker to finish.
    fn finish(&mut self) -> PolarsResult<()> {
        self.tx.take();

        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| polars_err!(ComputeError: "batch sink worker panicked"))?;
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

pub struct CallbackSinkNode {
    sink: Option<SinkSender>,
    worker: Option<PrefetchWorker>,
    maintain_order: bool,
    split_by: Option<PlSmallStr>,

    buffer: DataFrame,
    chunk_size: Option<NonZeroUsize>,
    is_done: bool,
}

impl CallbackSinkNode {
    pub fn new(
        function: PlanCallback<DataFrame, bool>,
        maintain_order: bool,
        chunk_size: Option<NonZeroUsize>,
        split_by: Option<PlSmallStr>,
        prefetch: Option<NonZeroUsize>,
    ) -> PolarsResult<Self> {
        let (sink, worker) = match prefetch {
            Some(capacity) => {
                let worker = PrefetchWorker::new(capacity, function)?;
                (SinkSender::Prefetch(worker.sender()), Some(worker))
            },
            None => (SinkSender::Direct(function), None),
        };

        Ok(Self {
            sink: Some(sink),
            worker,
            maintain_order,
            split_by,

            buffer: DataFrame::empty(),
            chunk_size,
            is_done: false,
        })
    }

    fn sender(&self) -> SinkSender {
        self.sink
            .as_ref()
            .expect("batch sink is set until the node completes")
            .clone()
    }

    fn flush_and_finish(&mut self, state: &StreamingExecutionState) -> PolarsResult<()> {
        let flush = if self.buffer.height() > 0 && !self.is_done {
            let df = std::mem::take(&mut self.buffer);
            if let Some(chunk_size) = self.chunk_size {
                assert!(df.height() <= chunk_size.get());
            }
            Some(df)
        } else {
            None
        };

        if flush.is_none() && self.worker.is_none() {
            return Ok(());
        }

        let sink = self
            .sink
            .take()
            .expect("batch sink is set until the node completes");
        let worker = self.worker.take();

        state.spawn_subphase_task(async move {
            let send_result = match flush {
                Some(df) => sink.send(df).await.map(|_| ()),
                None => Ok(()),
            };
            // Release the last sender before closing the queue, or the worker
            // would never observe the channel close.
            drop(sink);

            let finish_result = match worker {
                Some(mut worker) => ASYNC.spawn_blocking(move || worker.finish()).await.unwrap(),
                None => Ok(()),
            };

            send_result?;
            finish_result
        });

        Ok(())
    }
}

impl Drop for CallbackSinkNode {
    fn drop(&mut self) {
        // Release the sender first so the queue closes; drop the worker
        // without joining because the callback may still be running.
        self.sink.take();
        self.worker.take();
    }
}

impl ComputeNode for CallbackSinkNode {
    fn name(&self) -> &str {
        "sink_batches"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.is_empty());

        if self.is_done || recv[0] == PortState::Done {
            recv[0] = PortState::Done;
            return self.flush_and_finish(state);
        }

        recv[0] = PortState::Ready;
        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        _state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 1 && send_ports.is_empty());
        let mut recv = recv_ports[0]
            .take()
            .unwrap()
            .serial_with_maintain_order(self.maintain_order);

        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            while !self.is_done
                && let Ok(m) = recv.recv().await
            {
                let (sf, _, _, consume_token) = m.into_inner();
                let morsel = sf.into_df().await;

                if let Some(split_by) = self.split_by.clone() {
                    if morsel.height() == 0 {
                        drop(consume_token);
                        continue;
                    }

                    let key = split_by.as_str();
                    let mut complete = Vec::new();

                    if self.buffer.height() == 0 {
                        let (mut groups, pending) = split_by_key(&morsel, key)?;
                        self.buffer = pending;
                        complete.append(&mut groups);
                    } else {
                        let continues = {
                            let last = self.buffer.column(key)?.get(self.buffer.height() - 1)?;
                            let first = morsel.column(key)?.get(0)?;
                            last == first
                        };

                        if !continues {
                            complete.push(std::mem::take(&mut self.buffer));
                            let (mut groups, pending) = split_by_key(&morsel, key)?;
                            self.buffer = pending;
                            complete.append(&mut groups);
                        } else {
                            let run = first_run_len(&morsel, key)?;
                            if run == morsel.height() {
                                self.buffer.vstack_mut_owned(morsel)?;
                            } else {
                                let head = morsel.slice(0, run);
                                let mut group = std::mem::take(&mut self.buffer);
                                group.vstack_mut_owned(head)?;
                                complete.push(group);

                                let rest = morsel.slice(run as i64, morsel.height() - run);
                                let (mut groups, pending) = split_by_key(&rest, key)?;
                                self.buffer = pending;
                                complete.append(&mut groups);
                            }
                        }
                    }

                    for df in complete {
                        let should_stop = self.sender().send(df).await?;

                        if should_stop {
                            self.is_done = true;
                            break;
                        }
                    }
                } else {
                    // @NOTE: This also performs schema validation.
                    self.buffer.vstack_mut_owned(morsel)?;

                    while self.buffer.height() > 0
                        && self
                            .chunk_size
                            .is_none_or(|chunk_size| self.buffer.height() >= chunk_size.get())
                    {
                        let chunk_size = self.chunk_size.map_or(usize::MAX, Into::into);

                        let df;
                        (df, self.buffer) = self
                            .buffer
                            .split_at(self.buffer.height().min(chunk_size) as i64);

                        let should_stop = self.sender().send(df).await?;

                        if should_stop {
                            self.is_done = true;
                            break;
                        }
                    }
                }
                drop(consume_token);
                // Increase the backpressure. Only free up a pipeline when the morsel has been
                // processed in its entirety.
            }

            Ok(())
        }));
    }
}
