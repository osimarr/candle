use crate::{Buffer, Result};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Allocator {
    inner: Arc<AllocatorInner>,
}

#[derive(Debug)]
struct AllocatorInner {
    device_ordinal: usize,
    next_buffer_id: AtomicU64,
}

impl Allocator {
    pub(crate) fn new(device_ordinal: usize) -> Self {
        Self {
            inner: Arc::new(AllocatorInner {
                device_ordinal,
                next_buffer_id: AtomicU64::new(1),
            }),
        }
    }

    pub fn device_ordinal(&self) -> usize {
        self.inner.device_ordinal
    }

    pub fn allocate(&self, bytes: usize) -> Result<Buffer> {
        self.allocate_impl(bytes, false)
    }

    pub fn allocate_zeroed(&self, bytes: usize) -> Result<Buffer> {
        self.allocate_impl(bytes, true)
    }

    fn allocate_impl(&self, bytes: usize, zeroed: bool) -> Result<Buffer> {
        let id = self.inner.next_buffer_id.fetch_add(1, Ordering::Relaxed);
        Buffer::new(id, self.device_ordinal(), bytes, zeroed)
    }
}
