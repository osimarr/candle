use crate::{allocator::AllocationHandle, KernelDType, Result, RocmError};
use std::sync::Arc;
#[cfg(not(hip_runtime))]
use std::sync::Mutex;

#[derive(Clone)]
pub struct Buffer {
    inner: Arc<BufferInner>,
}

struct BufferInner {
    storage: BufferStorage,
    allocation: AllocationHandle,
}

enum BufferStorage {
    #[cfg(hip_runtime)]
    Hip(crate::hip::DeviceMemory),
    #[cfg(not(hip_runtime))]
    Host(Mutex<Vec<u8>>),
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("id", &self.id())
            .field("device_ordinal", &self.device_ordinal())
            .field("bytes", &self.size_in_bytes())
            .field("strong_count", &self.strong_count())
            .finish()
    }
}

impl Buffer {
    pub(crate) fn new(allocation: AllocationHandle, zeroed: bool) -> Result<Self> {
        let bytes = allocation.size_in_bytes();
        #[cfg(hip_runtime)]
        let storage = {
            let memory = crate::hip::DeviceMemory::allocate(allocation.device_ordinal(), bytes)?;
            if zeroed {
                crate::hip::memset(&memory, 0)?;
            }
            BufferStorage::Hip(memory)
        };
        #[cfg(not(hip_runtime))]
        let storage = {
            let data = if zeroed {
                vec![0; bytes]
            } else {
                // Keep the host fallback initialized so the shim remains safe.
                vec![0; bytes]
            };
            BufferStorage::Host(Mutex::new(data))
        };
        Ok(Self {
            inner: Arc::new(BufferInner {
                storage,
                allocation,
            }),
        })
    }

    pub fn id(&self) -> u64 {
        self.inner.allocation.id()
    }

    pub fn device_ordinal(&self) -> usize {
        self.inner.allocation.device_ordinal()
    }

    pub fn size_in_bytes(&self) -> usize {
        self.inner.allocation.size_in_bytes()
    }

    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    pub fn is_unique(&self) -> bool {
        self.strong_count() == 1
    }

    pub(crate) fn allocate_on_same_allocator(&self, bytes: usize, zeroed: bool) -> Result<Self> {
        self.inner
            .allocation
            .allocate_on_same_allocator(bytes, zeroed)
    }

    pub fn view(
        &self,
        byte_offset: usize,
        bytes: usize,
        dtype: KernelDType,
    ) -> Result<BufferView<'_>> {
        BufferView::new(self, byte_offset, bytes, dtype)
    }

    pub fn view_for_elems(
        &self,
        byte_offset: usize,
        elem_count: usize,
        dtype: KernelDType,
    ) -> Result<BufferView<'_>> {
        BufferView::for_elems(self, byte_offset, elem_count, dtype)
    }

    pub(crate) fn read_all(&self, dst: &mut [u8]) -> Result<()> {
        self.check_bounds(0, dst.len())?;
        match &self.inner.storage {
            #[cfg(hip_runtime)]
            BufferStorage::Hip(memory) => crate::hip::copy_d2h(memory, dst),
            #[cfg(not(hip_runtime))]
            BufferStorage::Host(data) => {
                let data = data
                    .lock()
                    .map_err(|_| RocmError::MutexPoisoned("rocm buffer"))?;
                dst.copy_from_slice(&data[..dst.len()]);
                Ok(())
            }
        }
    }

    pub(crate) fn write_all(&self, src: &[u8]) -> Result<()> {
        self.check_bounds(0, src.len())?;
        match &self.inner.storage {
            #[cfg(hip_runtime)]
            BufferStorage::Hip(memory) => crate::hip::copy_h2d(memory, src),
            #[cfg(not(hip_runtime))]
            BufferStorage::Host(data) => {
                let mut data = data
                    .lock()
                    .map_err(|_| RocmError::MutexPoisoned("rocm buffer"))?;
                data[..src.len()].copy_from_slice(src);
                Ok(())
            }
        }
    }

    #[cfg(hip_runtime)]
    pub(crate) fn device_ptr(&self) -> *mut std::os::raw::c_void {
        match &self.inner.storage {
            BufferStorage::Hip(memory) => memory.ptr(),
        }
    }

    pub(crate) fn check_device(&self, device_ordinal: usize, op: &'static str) -> Result<()> {
        if self.device_ordinal() == device_ordinal {
            Ok(())
        } else {
            Err(RocmError::DeviceMismatch {
                expected: device_ordinal,
                got: self.device_ordinal(),
                op,
            })
        }
    }

    fn check_bounds(&self, offset: usize, bytes: usize) -> Result<()> {
        let end = offset
            .checked_add(bytes)
            .ok_or(RocmError::BufferOutOfBounds {
                buffer_bytes: self.size_in_bytes(),
                offset,
                requested: bytes,
            })?;
        if end <= self.size_in_bytes() {
            Ok(())
        } else {
            Err(RocmError::BufferOutOfBounds {
                buffer_bytes: self.size_in_bytes(),
                offset,
                requested: bytes,
            })
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BufferView<'a> {
    buffer: &'a Buffer,
    byte_offset: usize,
    bytes: usize,
    dtype: KernelDType,
}

impl<'a> BufferView<'a> {
    pub fn new(
        buffer: &'a Buffer,
        byte_offset: usize,
        bytes: usize,
        dtype: KernelDType,
    ) -> Result<Self> {
        buffer.check_bounds(byte_offset, bytes)?;
        Ok(Self {
            buffer,
            byte_offset,
            bytes,
            dtype,
        })
    }

    pub fn for_elems(
        buffer: &'a Buffer,
        byte_offset: usize,
        elem_count: usize,
        dtype: KernelDType,
    ) -> Result<Self> {
        Self::new(
            buffer,
            byte_offset,
            dtype.storage_size_in_bytes(elem_count),
            dtype,
        )
    }

    pub fn buffer(&self) -> &'a Buffer {
        self.buffer
    }

    pub fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    pub fn size_in_bytes(&self) -> usize {
        self.bytes
    }

    pub fn dtype(&self) -> KernelDType {
        self.dtype
    }
}

#[cfg(test)]
mod tests {
    use crate::{Allocator, KernelDType, RocmError};

    #[test]
    fn view_checks_bounds() {
        let allocator = Allocator::new(0);
        let buffer = allocator.allocate_zeroed(8).unwrap();
        assert!(buffer.view(4, 4, KernelDType::F32).is_ok());
        assert!(matches!(
            buffer.view(5, 4, KernelDType::F32),
            Err(RocmError::BufferOutOfBounds { .. })
        ));
    }

    #[test]
    fn clone_tracks_shared_ownership() {
        let allocator = Allocator::new(0);
        let buffer = allocator.allocate_zeroed(8).unwrap();
        assert!(buffer.is_unique());
        let _clone = buffer.clone();
        assert!(!buffer.is_unique());
    }
}
