use crate::{Allocator, Buffer, Result, Stream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Device {
    inner: Arc<DeviceInner>,
}

#[derive(Debug)]
struct DeviceInner {
    ordinal: usize,
    default_stream: Stream,
    allocator: Allocator,
    seed: AtomicU64,
}

impl Device {
    pub fn new(ordinal: usize) -> Result<Self> {
        #[cfg(hip_runtime)]
        {
            let count = crate::hip::device_count()?;
            if ordinal >= count {
                return Err(crate::RocmError::Runtime(format!(
                    "requested ROCm device ordinal {ordinal}, but only {count} HIP device(s) are visible"
                )));
            }
            crate::hip::set_device(ordinal)?;
        }
        Ok(Self {
            inner: Arc::new(DeviceInner {
                ordinal,
                default_stream: Stream::new(ordinal),
                allocator: Allocator::new(ordinal),
                seed: AtomicU64::new(0),
            }),
        })
    }

    pub fn ordinal(&self) -> usize {
        self.inner.ordinal
    }

    pub fn default_stream(&self) -> &Stream {
        &self.inner.default_stream
    }

    pub fn allocator(&self) -> &Allocator {
        &self.inner.allocator
    }

    pub fn same_device(&self, rhs: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &rhs.inner)
    }

    pub fn same_ordinal(&self, rhs: &Self) -> bool {
        self.ordinal() == rhs.ordinal()
    }

    pub fn allocate(&self, bytes: usize) -> Result<Buffer> {
        self.allocator().allocate(bytes)
    }

    pub fn allocate_zeroed(&self, bytes: usize) -> Result<Buffer> {
        self.allocator().allocate_zeroed(bytes)
    }

    pub fn copy_from_host(&self, bytes: &[u8]) -> Result<Buffer> {
        let buffer = self.allocate(bytes.len())?;
        buffer.write_all(bytes)?;
        Ok(buffer)
    }

    pub fn copy_to_host(&self, buffer: &Buffer, dst: &mut [u8]) -> Result<()> {
        buffer.check_device(self.ordinal(), "copy_to_host")?;
        buffer.read_all(dst)
    }

    pub fn set_seed(&self, seed: u64) {
        self.inner.seed.store(seed, Ordering::Relaxed);
    }

    pub fn get_current_seed(&self) -> u64 {
        self.inner.seed.load(Ordering::Relaxed)
    }

    pub fn next_seed(&self) -> u64 {
        self.inner.seed.fetch_add(1, Ordering::Relaxed)
    }

    pub fn synchronize(&self) -> Result<()> {
        #[cfg(hip_runtime)]
        crate::hip::synchronize(self.ordinal())?;
        self.allocator().synchronize()
    }
}

#[cfg(test)]
mod tests {
    use super::Device;
    use crate::FreePolicy;

    #[test]
    fn host_copy_round_trip() {
        let device = Device::new(0).unwrap();
        let buffer = device.copy_from_host(&[1, 2, 3, 4]).unwrap();
        let mut dst = [0; 4];
        device.copy_to_host(&buffer, &mut dst).unwrap();
        assert_eq!(dst, [1, 2, 3, 4]);
    }

    #[test]
    fn host_copy_empty_buffer() {
        let device = Device::new(0).unwrap();
        let buffer = device.copy_from_host(&[]).unwrap();
        let mut dst = [];
        device.copy_to_host(&buffer, &mut dst).unwrap();
        assert_eq!(buffer.size_in_bytes(), 0);
    }

    #[test]
    fn device_identity_is_exact() {
        let d1 = Device::new(0).unwrap();
        let d2 = d1.clone();
        let d3 = Device::new(0).unwrap();
        assert!(d1.same_device(&d2));
        assert!(!d1.same_device(&d3));
        assert!(d1.same_ordinal(&d3));
    }

    #[test]
    fn synchronize_reclaims_deferred_frees() {
        let device = Device::new(0).unwrap();
        device
            .allocator()
            .set_free_policy(FreePolicy::DeferUntilSynchronize)
            .unwrap();
        let buffer = device.allocate(8).unwrap();
        drop(buffer);
        assert_eq!(device.allocator().stats().unwrap().pending_frees, 1);

        device.synchronize().unwrap();
        let stats = device.allocator().stats().unwrap();
        assert_eq!(stats.pending_frees, 0);
        assert_eq!(stats.synchronize_count, 1);
    }
}
