use crate::{Buffer, Result, RocmError};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

#[derive(Clone, Debug)]
pub struct Allocator {
    inner: Arc<AllocatorInner>,
}

#[derive(Debug)]
pub(crate) struct AllocatorInner {
    device_ordinal: usize,
    next_buffer_id: AtomicU64,
    lifecycle: Mutex<LifecycleState>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FreePolicy {
    #[default]
    SynchronizeOnDrop,
    DeferUntilSynchronize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AllocationStats {
    pub active_allocations: usize,
    pub active_bytes: usize,
    pub pending_frees: usize,
    pub pending_free_bytes: usize,
    pub total_allocations: u64,
    pub total_allocated_bytes: usize,
    pub total_released_allocations: u64,
    pub total_released_bytes: usize,
    pub synchronize_count: u64,
    pub synchronized_frees: u64,
}

#[derive(Debug, Clone, Copy)]
struct AllocationMeta {
    bytes: usize,
}

#[derive(Debug, Clone, Copy)]
struct PendingFree {
    bytes: usize,
}

#[derive(Debug, Default)]
struct LifecycleState {
    active: HashMap<u64, AllocationMeta>,
    pending_frees: Vec<PendingFree>,
    #[cfg(hip_runtime)]
    free_blocks: HashMap<usize, Vec<crate::hip::DeviceMemory>>,
    free_policy: FreePolicy,
    total_allocations: u64,
    total_allocated_bytes: usize,
    total_released_allocations: u64,
    total_released_bytes: usize,
    synchronize_count: u64,
    synchronized_frees: u64,
}

#[derive(Debug)]
pub(crate) struct AllocationHandle {
    id: u64,
    device_ordinal: usize,
    bytes: usize,
    allocator: Weak<AllocatorInner>,
    released: bool,
    #[cfg(hip_runtime)]
    reused_memory: Option<crate::hip::DeviceMemory>,
}

impl AllocationHandle {
    fn new(
        id: u64,
        device_ordinal: usize,
        bytes: usize,
        allocator: Weak<AllocatorInner>,
        #[cfg(hip_runtime)] reused_memory: Option<crate::hip::DeviceMemory>,
    ) -> Self {
        Self {
            id,
            device_ordinal,
            bytes,
            allocator,
            released: false,
            #[cfg(hip_runtime)]
            reused_memory,
        }
    }

    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    pub(crate) fn size_in_bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn allocate_on_same_allocator(&self, bytes: usize, zeroed: bool) -> Result<Buffer> {
        let allocator = self
            .allocator
            .upgrade()
            .ok_or_else(|| RocmError::Runtime("ROCm allocator was dropped".to_string()))?;
        let id = allocator.next_buffer_id.fetch_add(1, Ordering::Relaxed);
        let allocation = allocator.register_allocation(id, bytes)?;
        Buffer::new(allocation, zeroed)
    }

    pub(crate) fn release(&mut self) {
        if self.released {
            return;
        }
        self.released = true;
        if let Some(allocator) = self.allocator.upgrade() {
            #[cfg(hip_runtime)]
            allocator.release(self.id, self.bytes, None);
            #[cfg(not(hip_runtime))]
            allocator.release(self.id, self.bytes);
        }
    }

    #[cfg(hip_runtime)]
    pub(crate) fn take_reused_memory(&mut self) -> Option<crate::hip::DeviceMemory> {
        self.reused_memory.take()
    }

    #[cfg(hip_runtime)]
    pub(crate) fn release_memory(&mut self, memory: crate::hip::DeviceMemory) {
        if self.released {
            return;
        }
        self.released = true;
        if let Some(allocator) = self.allocator.upgrade() {
            allocator.release(self.id, self.bytes, Some(memory));
        }
    }
}

impl Drop for AllocationHandle {
    fn drop(&mut self) {
        self.release();
    }
}

impl Allocator {
    pub(crate) fn new(device_ordinal: usize) -> Self {
        Self {
            inner: Arc::new(AllocatorInner {
                device_ordinal,
                next_buffer_id: AtomicU64::new(1),
                lifecycle: Mutex::new(LifecycleState {
                    free_policy: FreePolicy::default(),
                    ..LifecycleState::default()
                }),
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

    pub fn free_policy(&self) -> Result<FreePolicy> {
        let lifecycle = self
            .inner
            .lifecycle
            .lock()
            .map_err(|_| RocmError::MutexPoisoned("rocm allocator"))?;
        Ok(lifecycle.free_policy)
    }

    pub fn set_free_policy(&self, policy: FreePolicy) -> Result<()> {
        let mut lifecycle = self
            .inner
            .lifecycle
            .lock()
            .map_err(|_| RocmError::MutexPoisoned("rocm allocator"))?;
        lifecycle.free_policy = policy;
        Ok(())
    }

    pub fn stats(&self) -> Result<AllocationStats> {
        self.inner.stats()
    }

    pub fn synchronize(&self) -> Result<()> {
        self.inner.synchronize()
    }

    fn allocate_impl(&self, bytes: usize, zeroed: bool) -> Result<Buffer> {
        let id = self.inner.next_buffer_id.fetch_add(1, Ordering::Relaxed);
        let allocation = self.inner.register_allocation(id, bytes)?;
        Buffer::new(allocation, zeroed)
    }
}

impl AllocatorInner {
    fn register_allocation(self: &Arc<Self>, id: u64, bytes: usize) -> Result<AllocationHandle> {
        let mut lifecycle = self
            .lifecycle
            .lock()
            .map_err(|_| RocmError::MutexPoisoned("rocm allocator"))?;
        lifecycle.active.insert(id, AllocationMeta { bytes });
        lifecycle.total_allocations += 1;
        lifecycle.total_allocated_bytes += bytes;
        Ok(AllocationHandle::new(
            id,
            self.device_ordinal,
            bytes,
            Arc::downgrade(self),
            #[cfg(hip_runtime)]
            Self::take_reusable_memory(&mut lifecycle, bytes),
        ))
    }

    #[cfg(hip_runtime)]
    fn take_reusable_memory(
        lifecycle: &mut LifecycleState,
        bytes: usize,
    ) -> Option<crate::hip::DeviceMemory> {
        let block = lifecycle
            .free_blocks
            .get_mut(&bytes)
            .and_then(|blocks| blocks.pop())?;
        if lifecycle
            .free_blocks
            .get(&bytes)
            .is_some_and(|blocks| blocks.is_empty())
        {
            lifecycle.free_blocks.remove(&bytes);
        }
        if let Some(index) = lifecycle
            .pending_frees
            .iter()
            .position(|pending| pending.bytes == bytes)
        {
            lifecycle.pending_frees.swap_remove(index);
        }
        Some(block)
    }

    #[cfg(hip_runtime)]
    fn release(&self, id: u64, bytes: usize, memory: Option<crate::hip::DeviceMemory>) {
        let mut drop_now = None;
        let Ok(mut lifecycle) = self.lifecycle.lock() else {
            return;
        };
        let bytes = lifecycle
            .active
            .remove(&id)
            .map(|meta| meta.bytes)
            .unwrap_or(bytes);
        lifecycle.total_released_allocations += 1;
        lifecycle.total_released_bytes += bytes;
        match lifecycle.free_policy {
            FreePolicy::SynchronizeOnDrop => {
                if memory.is_some() {
                    lifecycle.synchronized_frees += 1;
                }
                drop_now = memory;
            }
            FreePolicy::DeferUntilSynchronize => {
                if let Some(memory) = memory {
                    lifecycle.pending_frees.push(PendingFree { bytes });
                    lifecycle
                        .free_blocks
                        .entry(memory.bytes())
                        .or_default()
                        .push(memory);
                }
            }
        }
        drop(lifecycle);
        drop(drop_now);
    }

    #[cfg(not(hip_runtime))]
    fn release(&self, id: u64, bytes: usize) {
        let Ok(mut lifecycle) = self.lifecycle.lock() else {
            return;
        };
        let bytes = lifecycle
            .active
            .remove(&id)
            .map(|meta| meta.bytes)
            .unwrap_or(bytes);
        lifecycle.total_released_allocations += 1;
        lifecycle.total_released_bytes += bytes;
        match lifecycle.free_policy {
            FreePolicy::SynchronizeOnDrop => {
                lifecycle.synchronized_frees += 1;
            }
            FreePolicy::DeferUntilSynchronize => {
                lifecycle.pending_frees.push(PendingFree { bytes });
            }
        }
    }

    fn synchronize(&self) -> Result<()> {
        let mut lifecycle = self
            .lifecycle
            .lock()
            .map_err(|_| RocmError::MutexPoisoned("rocm allocator"))?;
        lifecycle.synchronize_count += 1;
        lifecycle.synchronized_frees += lifecycle.pending_frees.len() as u64;
        lifecycle.pending_frees.clear();
        Ok(())
    }

    fn stats(&self) -> Result<AllocationStats> {
        let lifecycle = self
            .lifecycle
            .lock()
            .map_err(|_| RocmError::MutexPoisoned("rocm allocator"))?;
        Ok(AllocationStats {
            active_allocations: lifecycle.active.len(),
            active_bytes: lifecycle.active.values().map(|meta| meta.bytes).sum(),
            pending_frees: lifecycle.pending_frees.len(),
            pending_free_bytes: lifecycle
                .pending_frees
                .iter()
                .map(|pending| pending.bytes)
                .sum(),
            total_allocations: lifecycle.total_allocations,
            total_allocated_bytes: lifecycle.total_allocated_bytes,
            total_released_allocations: lifecycle.total_released_allocations,
            total_released_bytes: lifecycle.total_released_bytes,
            synchronize_count: lifecycle.synchronize_count,
            synchronized_frees: lifecycle.synchronized_frees,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Allocator, FreePolicy};

    #[test]
    fn allocator_tracks_active_and_released_buffers() {
        let allocator = Allocator::new(0);
        let buffer = allocator.allocate(16).unwrap();
        let stats = allocator.stats().unwrap();
        assert_eq!(stats.active_allocations, 1);
        assert_eq!(stats.active_bytes, 16);
        assert_eq!(stats.total_allocations, 1);

        drop(buffer);
        let stats = allocator.stats().unwrap();
        assert_eq!(stats.active_allocations, 0);
        assert_eq!(stats.pending_frees, 0);
        assert_eq!(stats.total_released_allocations, 1);
        assert_eq!(stats.total_released_bytes, 16);
        assert_eq!(stats.synchronized_frees, 1);
    }

    #[test]
    fn deferred_free_policy_reclaims_on_synchronize() {
        let allocator = Allocator::new(0);
        allocator
            .set_free_policy(FreePolicy::DeferUntilSynchronize)
            .unwrap();
        let buffer = allocator.allocate(24).unwrap();
        drop(buffer);

        let stats = allocator.stats().unwrap();
        assert_eq!(stats.active_allocations, 0);
        assert_eq!(stats.pending_frees, 1);
        assert_eq!(stats.pending_free_bytes, 24);
        assert_eq!(stats.synchronized_frees, 0);

        allocator.synchronize().unwrap();
        let stats = allocator.stats().unwrap();
        assert_eq!(stats.pending_frees, 0);
        assert_eq!(stats.pending_free_bytes, 0);
        assert_eq!(stats.synchronize_count, 1);
        assert_eq!(stats.synchronized_frees, 1);
    }
}
