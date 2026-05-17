use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Stream {
    inner: Arc<StreamInner>,
}

#[derive(Debug)]
struct StreamInner {
    id: u64,
    device_ordinal: usize,
}

impl Stream {
    pub(crate) fn new(device_ordinal: usize) -> Self {
        static NEXT_STREAM_ID: AtomicU64 = AtomicU64::new(1);
        Self {
            inner: Arc::new(StreamInner {
                id: NEXT_STREAM_ID.fetch_add(1, Ordering::Relaxed),
                device_ordinal,
            }),
        }
    }

    pub fn id(&self) -> u64 {
        self.inner.id
    }

    pub fn device_ordinal(&self) -> usize {
        self.inner.device_ordinal
    }
}
