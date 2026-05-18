//! ROCm kernel dispatch surface for Candle.
//!
//! `candle-core` calls into this crate with typed ROCm operation descriptors
//! first. When compiled with HIP support, supported f32, f16, bf16, and fp8
//! operation slices launch HIP kernels behind these entry points; unsupported
//! operations still use fallback closures while coverage is expanded.

mod allocator;
mod buffer;
mod dtype;
mod error;
#[cfg(hip_runtime)]
mod hip;
#[path = "device.rs"]
mod runtime_device;
mod stream;

pub use allocator::{AllocationStats, Allocator, FreePolicy};
pub use buffer::{Buffer, BufferView};
pub use dtype::KernelDType;
pub use error::{Result, RocmError};
pub use runtime_device::Device;
pub use stream::Stream;

fn cpu_fallback<T, E, F>(_op: &'static str, fallback: F) -> std::result::Result<T, E>
where
    F: FnOnce() -> std::result::Result<T, E>,
{
    fallback()
}

#[derive(Clone, Debug)]
pub struct TensorArg {
    buffer: Buffer,
    dtype: KernelDType,
    elem_count: usize,
}

impl TensorArg {
    pub fn new(buffer: &Buffer, dtype: KernelDType, elem_count: usize) -> Result<Self> {
        buffer.view_for_elems(0, elem_count, dtype)?;
        Ok(Self {
            buffer: buffer.clone(),
            dtype,
            elem_count,
        })
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn view(&self) -> Result<BufferView<'_>> {
        self.buffer.view_for_elems(0, self.elem_count, self.dtype)
    }

    pub fn dtype(&self) -> KernelDType {
        self.dtype
    }

    pub fn elem_count(&self) -> usize {
        self.elem_count
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TensorOutput {
    dtype: KernelDType,
    elem_count: usize,
}

impl TensorOutput {
    pub fn new(dtype: KernelDType, elem_count: usize) -> Self {
        Self { dtype, elem_count }
    }

    pub fn dtype(&self) -> KernelDType {
        self.dtype
    }

    pub fn elem_count(&self) -> usize {
        self.elem_count
    }
}

#[derive(Clone, Copy, Debug)]
pub enum KernelScalar {
    F32(f32),
    U8(u8),
    U32(u32),
    I16(i16),
    I32(i32),
    I64(i64),
    BF16(u16),
    F16(u16),
    F64(f64),
    F8E4M3(u8),
}

impl KernelScalar {
    pub fn dtype(self) -> KernelDType {
        match self {
            Self::F32(_) => KernelDType::F32,
            Self::U8(_) => KernelDType::U8,
            Self::U32(_) => KernelDType::U32,
            Self::I16(_) => KernelDType::I16,
            Self::I32(_) => KernelDType::I32,
            Self::I64(_) => KernelDType::I64,
            Self::BF16(_) => KernelDType::BF16,
            Self::F16(_) => KernelDType::F16,
            Self::F64(_) => KernelDType::F64,
            Self::F8E4M3(_) => KernelDType::F8E4M3,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LayoutArg {
    dims: Vec<usize>,
    stride: Vec<usize>,
    start_offset: usize,
}

impl LayoutArg {
    pub fn new(dims: Vec<usize>, stride: Vec<usize>, start_offset: usize) -> Result<Self> {
        if dims.len() != stride.len() {
            return Err(RocmError::Runtime(format!(
                "layout rank mismatch: dims rank {}, stride rank {}",
                dims.len(),
                stride.len()
            )));
        }
        Ok(Self {
            dims,
            stride,
            start_offset,
        })
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    pub fn start_offset(&self) -> usize {
        self.start_offset
    }

    pub fn elem_count(&self) -> usize {
        self.dims.iter().product()
    }

    fn storage_index(&self, logical_index: usize) -> usize {
        let mut logical_index = logical_index;
        let mut storage_index = self.start_offset;
        for (&dim, &stride) in self.dims.iter().zip(self.stride.iter()).rev() {
            if dim == 0 {
                return self.start_offset;
            }
            let index = logical_index % dim;
            logical_index /= dim;
            storage_index += index * stride;
        }
        storage_index
    }

    fn max_storage_index(&self) -> Option<usize> {
        if self.dims.contains(&0) {
            return None;
        }
        let mut max_index = self.start_offset;
        for (&dim, &stride) in self.dims.iter().zip(self.stride.iter()) {
            max_index = max_index.checked_add((dim - 1).checked_mul(stride)?)?;
        }
        Some(max_index)
    }
}

pub mod custom {
    use crate::{cpu_fallback, TensorArg, TensorOutput};

    #[derive(Clone, Debug)]
    pub struct Op1 {
        pub name: &'static str,
        pub input: TensorArg,
        pub output: Option<TensorOutput>,
    }

    #[derive(Clone, Debug)]
    pub struct Op2 {
        pub name: &'static str,
        pub lhs: TensorArg,
        pub rhs: TensorArg,
        pub output: Option<TensorOutput>,
    }

    #[derive(Clone, Debug)]
    pub struct Op3 {
        pub name: &'static str,
        pub lhs: TensorArg,
        pub rhs2: TensorArg,
        pub rhs3: TensorArg,
        pub output: Option<TensorOutput>,
    }

    #[derive(Clone, Debug)]
    pub struct InplaceOp1 {
        pub name: &'static str,
        pub dst: TensorArg,
    }

    #[derive(Clone, Debug)]
    pub struct InplaceOp2 {
        pub name: &'static str,
        pub dst: TensorArg,
        pub rhs: TensorArg,
    }

    #[derive(Clone, Debug)]
    pub struct InplaceOp3 {
        pub name: &'static str,
        pub dst: TensorArg,
        pub rhs2: TensorArg,
        pub rhs3: TensorArg,
    }

    pub fn call_apply_op1<T, E, F>(op: Op1, fallback: F) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        cpu_fallback(op.name, fallback)
    }

    pub fn call_arg_sort<T, E, F>(op: Op1, fallback: F) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        cpu_fallback(op.name, fallback)
    }

    pub fn call_apply_op2<T, E, F>(op: Op2, fallback: F) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        cpu_fallback(op.name, fallback)
    }

    pub fn call_apply_op3<T, E, F>(op: Op3, fallback: F) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        cpu_fallback(op.name, fallback)
    }

    pub fn call_inplace_op1<T, E, F>(op: InplaceOp1, fallback: F) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        cpu_fallback(op.name, fallback)
    }

    pub fn call_inplace_op2<T, E, F>(op: InplaceOp2, fallback: F) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        cpu_fallback(op.name, fallback)
    }

    pub fn call_inplace_op3<T, E, F>(op: InplaceOp3, fallback: F) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        cpu_fallback(op.name, fallback)
    }
}

pub mod device {
    use crate::{cpu_fallback, Buffer, Device, KernelDType, RocmError, TensorOutput};

    #[derive(Clone, Copy, Debug)]
    pub struct DeviceOp<'a> {
        pub name: &'static str,
        pub device: &'a Device,
    }

    #[derive(Clone, Copy, Debug)]
    pub struct AllocOp<'a> {
        pub name: &'static str,
        pub device: &'a Device,
        pub output: TensorOutput,
    }

    #[derive(Clone, Copy, Debug)]
    pub struct SliceLoadOp<'a> {
        pub name: &'static str,
        pub device: &'a Device,
        pub dtype: KernelDType,
        pub elem_count: usize,
    }

    pub fn call_zeros(op: AllocOp<'_>) -> crate::Result<Buffer> {
        op.device.allocate_zeroed(
            op.output
                .dtype()
                .storage_size_in_bytes(op.output.elem_count()),
        )
    }

    pub fn call_alloc_uninit(op: AllocOp<'_>) -> crate::Result<Buffer> {
        op.device.allocate(
            op.output
                .dtype()
                .storage_size_in_bytes(op.output.elem_count()),
        )
    }

    pub fn call_storage_from_host_bytes(
        op: SliceLoadOp<'_>,
        bytes: &[u8],
    ) -> crate::Result<Buffer> {
        let expected = op.dtype.storage_size_in_bytes(op.elem_count);
        if bytes.len() != expected {
            return Err(RocmError::BufferOutOfBounds {
                buffer_bytes: bytes.len(),
                offset: 0,
                requested: expected,
            });
        }
        op.device.copy_from_host(bytes)
    }

    pub fn call_storage_from_slice<T, E, F>(
        op: SliceLoadOp<'_>,
        fallback: F,
    ) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        cpu_fallback(op.name, fallback)
    }

    pub fn call_storage_from_cpu_storage<T, E, F>(
        op: SliceLoadOp<'_>,
        fallback: F,
    ) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        cpu_fallback(op.name, fallback)
    }

    pub fn call_storage_from_cpu_storage_owned<T, E, F>(
        op: SliceLoadOp<'_>,
        fallback: F,
    ) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        cpu_fallback(op.name, fallback)
    }

    pub fn call_rand_uniform<T, E, F>(op: AllocOp<'_>, fallback: F) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        cpu_fallback(op.name, fallback)
    }

    pub fn call_rand_normal<T, E, F>(op: AllocOp<'_>, fallback: F) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        cpu_fallback(op.name, fallback)
    }

    pub fn try_rand_uniform(
        op: AllocOp<'_>,
        seed: u64,
        lo: f32,
        up: f32,
    ) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (op, seed, lo, up);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        {
            if !matches!(
                op.output.dtype(),
                KernelDType::F32 | KernelDType::BF16 | KernelDType::F8E4M3
            ) {
                return Ok(None);
            }
            let dst = op.device.allocate(
                op.output
                    .dtype()
                    .storage_size_in_bytes(op.output.elem_count()),
            )?;
            match op.output.dtype() {
                KernelDType::F32 => {
                    crate::hip::random_uniform_f32(&dst, op.output.elem_count(), seed, lo, up)?
                }
                KernelDType::BF16 => {
                    crate::hip::random_uniform_bf16(&dst, op.output.elem_count(), seed, lo, up)?
                }
                KernelDType::F8E4M3 => {
                    crate::hip::random_uniform_f8e4m3(&dst, op.output.elem_count(), seed, lo, up)?
                }
                _ => unreachable!(),
            }
            Ok(Some(dst))
        }
    }

    pub fn try_rand_normal(
        op: AllocOp<'_>,
        seed: u64,
        mean: f32,
        std: f32,
    ) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (op, seed, mean, std);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        {
            if !matches!(
                op.output.dtype(),
                KernelDType::F32 | KernelDType::BF16 | KernelDType::F8E4M3
            ) {
                return Ok(None);
            }
            let dst = op.device.allocate(
                op.output
                    .dtype()
                    .storage_size_in_bytes(op.output.elem_count()),
            )?;
            match op.output.dtype() {
                KernelDType::F32 => {
                    crate::hip::random_normal_f32(&dst, op.output.elem_count(), seed, mean, std)?
                }
                KernelDType::BF16 => {
                    crate::hip::random_normal_bf16(&dst, op.output.elem_count(), seed, mean, std)?
                }
                KernelDType::F8E4M3 => {
                    crate::hip::random_normal_f8e4m3(&dst, op.output.elem_count(), seed, mean, std)?
                }
                _ => unreachable!(),
            }
            Ok(Some(dst))
        }
    }

    pub fn call_set_seed<T, E, F>(op: DeviceOp<'_>, fallback: F) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        cpu_fallback(op.name, fallback)
    }

    pub fn call_get_current_seed<T, E, F>(
        op: DeviceOp<'_>,
        fallback: F,
    ) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        cpu_fallback(op.name, fallback)
    }
}

pub mod quantized {
    #[cfg(not(hip_runtime))]
    use crate::LayoutArg;
    use crate::{cpu_fallback, Buffer, Device, TensorArg, TensorOutput};
    #[cfg(hip_runtime)]
    use crate::{KernelDType, LayoutArg, RocmError};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum QuantizedDType {
        Q5_0,
        Q8_0,
        Q4K,
        Q6K,
    }

    #[derive(Clone, Debug)]
    pub struct QuantizedOp<'a> {
        pub name: &'static str,
        pub device: Option<&'a Device>,
        pub input: Option<TensorArg>,
    }

    #[derive(Clone, Debug)]
    pub struct MatMulOp<'a> {
        pub name: &'static str,
        pub device: &'a Device,
        pub weights: &'a Buffer,
        pub rhs: TensorArg,
        pub rhs_layout: LayoutArg,
        pub output: TensorOutput,
        pub batch_size: usize,
        pub nrows: usize,
        pub ncols: usize,
    }

    pub fn call_zeros<T, E, F>(op: QuantizedOp<'_>, fallback: F) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        cpu_fallback(op.name, fallback)
    }

    pub fn call_load_quantized<T, E, F>(
        op: QuantizedOp<'_>,
        fallback: F,
    ) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        cpu_fallback(op.name, fallback)
    }

    pub fn call_dequantize<T, E, F>(op: QuantizedOp<'_>, fallback: F) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        cpu_fallback(op.name, fallback)
    }

    pub fn call_quantize<T, E, F>(op: QuantizedOp<'_>, fallback: F) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        cpu_fallback(op.name, fallback)
    }

    pub fn call_quantize_imatrix<T, E, F>(
        op: QuantizedOp<'_>,
        fallback: F,
    ) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        cpu_fallback(op.name, fallback)
    }

    pub fn call_quantize_onto<T, E, F>(
        op: QuantizedOp<'_>,
        fallback: F,
    ) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        cpu_fallback(op.name, fallback)
    }

    pub fn call_quantize_imatrix_onto<T, E, F>(
        op: QuantizedOp<'_>,
        fallback: F,
    ) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        cpu_fallback(op.name, fallback)
    }

    pub fn call_matmul_t<T, E, F>(op: QuantizedOp<'_>, fallback: F) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        cpu_fallback(op.name, fallback)
    }

    pub fn call_data<T, E, F>(op: QuantizedOp<'_>, fallback: F) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        cpu_fallback(op.name, fallback)
    }

    pub fn zeros(device: &Device, size_in_bytes: usize) -> crate::Result<Buffer> {
        device.allocate_zeroed(size_in_bytes)
    }

    pub fn load_quantized(device: &Device, data: &[u8]) -> crate::Result<Buffer> {
        device.copy_from_host(data)
    }

    pub fn data(device: &Device, buffer: &Buffer) -> crate::Result<Vec<u8>> {
        let mut data = vec![0; buffer.size_in_bytes()];
        device.copy_to_host(buffer, &mut data)?;
        Ok(data)
    }

    pub fn try_matmul_t(dtype: QuantizedDType, op: &MatMulOp<'_>) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (dtype, op);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        {
            if op.rhs.dtype() != KernelDType::F32 || op.output.dtype() != KernelDType::F32 {
                return Ok(None);
            }
            if op.weights.device_ordinal() != op.device.ordinal()
                || op.rhs.buffer().device_ordinal() != op.device.ordinal()
            {
                let got = if op.weights.device_ordinal() != op.device.ordinal() {
                    op.weights.device_ordinal()
                } else {
                    op.rhs.buffer().device_ordinal()
                };
                return Err(RocmError::DeviceMismatch {
                    expected: op.device.ordinal(),
                    got,
                    op: op.name,
                });
            }
            if op.ncols == 0
                || op.nrows == 0
                || op.batch_size == 0
                || op.output.elem_count() != op.batch_size * op.nrows
                || op.rhs_layout.elem_count() != op.batch_size * op.ncols
            {
                return Err(RocmError::Runtime(format!(
                    "invalid quantized matmul shape for {}: batch {}, nrows {}, ncols {}, rhs elems {}, output elems {}",
                    op.name,
                    op.batch_size,
                    op.nrows,
                    op.ncols,
                    op.rhs_layout.elem_count(),
                    op.output.elem_count()
                )));
            }

            let (block_size, type_size) = match dtype {
                QuantizedDType::Q5_0 => (32, 22),
                QuantizedDType::Q8_0 => (32, 34),
                QuantizedDType::Q4K => (256, 144),
                QuantizedDType::Q6K => (256, 210),
            };
            if !op.ncols.is_multiple_of(block_size) {
                return Err(RocmError::Runtime(format!(
                    "invalid quantized matmul shape for {}: ncols {} is not divisible by block size {}",
                    op.name, op.ncols, block_size
                )));
            }
            let expected_weights = op.nrows * (op.ncols / block_size) * type_size;
            if op.weights.size_in_bytes() != expected_weights {
                return Err(RocmError::BufferOutOfBounds {
                    buffer_bytes: op.weights.size_in_bytes(),
                    offset: 0,
                    requested: expected_weights,
                });
            }

            let dst = op.device.allocate(
                op.output
                    .dtype()
                    .storage_size_in_bytes(op.output.elem_count()),
            )?;
            match dtype {
                QuantizedDType::Q5_0 => crate::hip::qmatmul_t_q5_0_f32(
                    op.weights,
                    op.rhs.buffer(),
                    &op.rhs_layout,
                    &dst,
                    op.batch_size,
                    op.nrows,
                    op.ncols,
                )?,
                QuantizedDType::Q8_0 => crate::hip::qmatmul_t_q8_0_f32(
                    op.weights,
                    op.rhs.buffer(),
                    &op.rhs_layout,
                    &dst,
                    op.batch_size,
                    op.nrows,
                    op.ncols,
                )?,
                QuantizedDType::Q4K => crate::hip::qmatmul_t_q4k_f32(
                    op.weights,
                    op.rhs.buffer(),
                    &op.rhs_layout,
                    &dst,
                    op.batch_size,
                    op.nrows,
                    op.ncols,
                )?,
                QuantizedDType::Q6K => crate::hip::qmatmul_t_q6k_f32(
                    op.weights,
                    op.rhs.buffer(),
                    &op.rhs_layout,
                    &dst,
                    op.batch_size,
                    op.nrows,
                    op.ncols,
                )?,
            }
            Ok(Some(dst))
        }
    }

    pub fn supports_native_bf16() -> bool {
        cfg!(hip_runtime)
    }

    pub fn supports_native_f16() -> bool {
        cfg!(hip_runtime)
    }

    pub fn try_quantize_bf16(src: &TensorArg, dst: &Buffer) -> crate::Result<bool> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (src, dst);
            Ok(false)
        }
        #[cfg(hip_runtime)]
        {
            if dst.size_in_bytes() != KernelDType::BF16.storage_size_in_bytes(src.elem_count()) {
                return Err(RocmError::BufferOutOfBounds {
                    buffer_bytes: dst.size_in_bytes(),
                    offset: 0,
                    requested: KernelDType::BF16.storage_size_in_bytes(src.elem_count()),
                });
            }
            match src.dtype() {
                KernelDType::F32 => {
                    let layout = LayoutArg::new(vec![src.elem_count()], vec![1], 0)?;
                    crate::hip::cast_f32_to_bf16(src.buffer(), &layout, dst)?;
                    Ok(true)
                }
                KernelDType::BF16 => {
                    crate::hip::copy_d2d(src.buffer(), dst, dst.size_in_bytes())?;
                    Ok(true)
                }
                _ => Ok(false),
            }
        }
    }

    pub fn try_quantize_f16(src: &TensorArg, dst: &Buffer) -> crate::Result<bool> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (src, dst);
            Ok(false)
        }
        #[cfg(hip_runtime)]
        {
            if dst.size_in_bytes() != KernelDType::F16.storage_size_in_bytes(src.elem_count()) {
                return Err(RocmError::BufferOutOfBounds {
                    buffer_bytes: dst.size_in_bytes(),
                    offset: 0,
                    requested: KernelDType::F16.storage_size_in_bytes(src.elem_count()),
                });
            }
            match src.dtype() {
                KernelDType::F32 => {
                    let layout = LayoutArg::new(vec![src.elem_count()], vec![1], 0)?;
                    crate::hip::cast_f32_to_f16(src.buffer(), &layout, dst)?;
                    Ok(true)
                }
                KernelDType::F16 => {
                    crate::hip::copy_d2d(src.buffer(), dst, dst.size_in_bytes())?;
                    Ok(true)
                }
                _ => Ok(false),
            }
        }
    }

    pub fn try_dequantize_bf16_to_f32(
        device: &Device,
        src: &Buffer,
        elem_count: usize,
    ) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (device, src, elem_count);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        {
            if src.size_in_bytes() != KernelDType::BF16.storage_size_in_bytes(elem_count) {
                return Err(RocmError::BufferOutOfBounds {
                    buffer_bytes: src.size_in_bytes(),
                    offset: 0,
                    requested: KernelDType::BF16.storage_size_in_bytes(elem_count),
                });
            }
            let dst = device.allocate(KernelDType::F32.storage_size_in_bytes(elem_count))?;
            let layout = LayoutArg::new(vec![elem_count], vec![1], 0)?;
            crate::hip::cast_bf16_to_f32(src, &layout, &dst)?;
            Ok(Some(dst))
        }
    }

    pub fn try_dequantize_f16_to_f32(
        device: &Device,
        src: &Buffer,
        elem_count: usize,
    ) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (device, src, elem_count);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        {
            if src.size_in_bytes() != KernelDType::F16.storage_size_in_bytes(elem_count) {
                return Err(RocmError::BufferOutOfBounds {
                    buffer_bytes: src.size_in_bytes(),
                    offset: 0,
                    requested: KernelDType::F16.storage_size_in_bytes(elem_count),
                });
            }
            let dst = device.allocate(KernelDType::F32.storage_size_in_bytes(elem_count))?;
            let layout = LayoutArg::new(vec![elem_count], vec![1], 0)?;
            crate::hip::cast_f16_to_f32(src, &layout, &dst)?;
            Ok(Some(dst))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn quantized_load_round_trip() {
            let device = Device::new(0).unwrap();
            let raw = [1u8, 2, 3, 5, 8, 13];
            let buffer = load_quantized(&device, &raw).unwrap();
            assert_eq!(buffer.size_in_bytes(), raw.len());
            assert_eq!(data(&device, &buffer).unwrap(), raw);
        }

        #[test]
        fn quantized_zeros_are_zeroed() {
            let device = Device::new(0).unwrap();
            let buffer = zeros(&device, 17).unwrap();
            assert_eq!(data(&device, &buffer).unwrap(), vec![0; 17]);
        }
    }
}

pub mod tensor {
    use crate::{
        cpu_fallback, Buffer, Device, KernelDType, KernelScalar, LayoutArg, RocmError, TensorArg,
        TensorOutput,
    };

    #[derive(Clone, Debug)]
    pub struct Op1 {
        pub name: &'static str,
        pub input: TensorArg,
        pub input_layout: Option<LayoutArg>,
        pub output: Option<TensorOutput>,
    }

    #[derive(Clone, Debug)]
    pub struct Op2 {
        pub name: &'static str,
        pub lhs: TensorArg,
        pub rhs: TensorArg,
        pub lhs_layout: Option<LayoutArg>,
        pub rhs_layout: Option<LayoutArg>,
        pub output: Option<TensorOutput>,
    }

    #[derive(Clone, Debug)]
    pub struct Op3 {
        pub name: &'static str,
        pub first: TensorArg,
        pub second: TensorArg,
        pub third: TensorArg,
        pub output: Option<TensorOutput>,
    }

    #[derive(Clone, Debug)]
    pub struct InplaceOp1 {
        pub name: &'static str,
        pub dst: TensorArg,
        pub dst_layout: Option<LayoutArg>,
        pub scalar: Option<KernelScalar>,
    }

    #[derive(Clone, Debug)]
    pub struct InplaceOp2 {
        pub name: &'static str,
        pub dst: TensorArg,
        pub src: TensorArg,
        pub copy: Option<CopySpec>,
    }

    #[derive(Clone, Debug)]
    pub struct InplaceOp3 {
        pub name: &'static str,
        pub dst: TensorArg,
        pub second: TensorArg,
        pub third: TensorArg,
    }

    #[derive(Clone, Debug)]
    pub struct TransferOp<'a> {
        pub name: &'static str,
        pub src: TensorArg,
        pub dst_device: &'a Device,
        pub output: TensorOutput,
    }

    #[derive(Clone, Debug)]
    pub enum CopySpec {
        StridedSrc {
            dst_offset: usize,
            src_layout: LayoutArg,
        },
        Copy2d {
            d1: usize,
            d2: usize,
            src_stride1: usize,
            dst_stride1: usize,
            src_offset: usize,
            dst_offset: usize,
        },
    }

    #[derive(Clone, Copy, Debug)]
    pub struct Conv1dParams {
        pub padding: usize,
        pub stride: usize,
        pub dilation: usize,
        pub l_out: usize,
        pub elem_count: usize,
    }

    #[derive(Clone, Copy, Debug)]
    pub struct Conv2dParams {
        pub padding: usize,
        pub stride: usize,
        pub dilation: usize,
        pub out_h: usize,
        pub out_w: usize,
        pub elem_count: usize,
    }

    macro_rules! call_op {
        ($fn_name:ident, $op_ty:ty) => {
            pub fn $fn_name<T, E, F>(op: $op_ty, fallback: F) -> std::result::Result<T, E>
            where
                F: FnOnce() -> std::result::Result<T, E>,
            {
                cpu_fallback(op.name, fallback)
            }
        };
    }

    call_op!(call_try_clone, Op1);
    call_op!(call_transfer_to_cpu, Op1);
    call_op!(call_affine, Op1);
    call_op!(call_powf, Op1);
    call_op!(call_elu, Op1);
    call_op!(call_reduce, Op1);
    call_op!(call_to_dtype, Op1);
    call_op!(call_unary, Op1);
    call_op!(call_avg_pool2d, Op1);
    call_op!(call_max_pool2d, Op1);
    call_op!(call_upsample_nearest1d, Op1);
    call_op!(call_upsample_nearest2d, Op1);
    call_op!(call_upsample_bilinear2d, Op1);

    call_op!(call_cmp, Op2);
    call_op!(call_binary, Op2);
    call_op!(call_conv1d, Op2);
    call_op!(call_conv2d, Op2);
    call_op!(call_conv_transpose1d, Op2);
    call_op!(call_conv_transpose2d, Op2);
    call_op!(call_gather, Op2);
    call_op!(call_index_select, Op2);
    call_op!(call_matmul, Op2);

    call_op!(call_where_cond, Op3);
    call_op!(call_index_add, Op3);

    call_op!(call_scatter_set, InplaceOp3);
    call_op!(call_scatter_add_set, InplaceOp3);

    pub fn call_transfer<E, F>(op: TransferOp<'_>, fallback: F) -> std::result::Result<Buffer, E>
    where
        E: From<RocmError>,
        F: FnOnce() -> std::result::Result<Buffer, E>,
    {
        match transfer(&op) {
            Ok(buffer) => Ok(buffer),
            Err(RocmError::UnsupportedDType { .. } | RocmError::NotImplemented(_)) => fallback(),
            Err(err) => Err(E::from(err)),
        }
    }

    pub fn call_copy_strided_src<E, F>(op: InplaceOp2, fallback: F) -> std::result::Result<(), E>
    where
        E: From<RocmError>,
        F: FnOnce() -> std::result::Result<(), E>,
    {
        match copy_strided_src(&op) {
            Ok(()) => Ok(()),
            Err(RocmError::UnsupportedDType { .. } | RocmError::NotImplemented(_)) => fallback(),
            Err(err) => Err(E::from(err)),
        }
    }

    pub fn call_copy2d<E, F>(op: InplaceOp2, fallback: F) -> std::result::Result<(), E>
    where
        E: From<RocmError>,
        F: FnOnce() -> std::result::Result<(), E>,
    {
        match copy2d(&op) {
            Ok(()) => Ok(()),
            Err(RocmError::UnsupportedDType { .. } | RocmError::NotImplemented(_)) => fallback(),
            Err(err) => Err(E::from(err)),
        }
    }

    pub fn call_const_set<E, F>(op: InplaceOp1, fallback: F) -> std::result::Result<(), E>
    where
        E: From<RocmError>,
        F: FnOnce() -> std::result::Result<(), E>,
    {
        match const_set(&op) {
            Ok(()) => Ok(()),
            Err(RocmError::UnsupportedDType { .. } | RocmError::NotImplemented(_)) => fallback(),
            Err(err) => Err(E::from(err)),
        }
    }

    pub fn try_clone(op: &Op1) -> crate::Result<Option<Buffer>> {
        let Some(layout) = op.input_layout.as_ref() else {
            return Ok(None);
        };
        let Some(output) = op.output else {
            return Ok(None);
        };
        if op.input.dtype() != output.dtype() || op.input.elem_count() < layout.elem_count() {
            return Ok(None);
        }
        check_layout(&op.input, layout)?;
        let dst = op.input.buffer().allocate_on_same_allocator(
            output.dtype().storage_size_in_bytes(output.elem_count()),
            false,
        )?;
        let copy = InplaceOp2 {
            name: "try_clone",
            dst: TensorArg::new(&dst, output.dtype(), output.elem_count())?,
            src: op.input.clone(),
            copy: Some(CopySpec::StridedSrc {
                dst_offset: 0,
                src_layout: layout.clone(),
            }),
        };
        copy_strided_src(&copy)?;
        Ok(Some(dst))
    }

    pub fn try_affine(op: &Op1, mul: f32, add: f32) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (op, mul, add);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        try_scalar(op, |dtype, src, layout, dst| {
            match dtype {
                KernelDType::F32 => crate::hip::affine_f32(src, layout, dst, mul, add)?,
                KernelDType::BF16 => crate::hip::affine_bf16(src, layout, dst, mul, add)?,
                KernelDType::F8E4M3 => crate::hip::affine_f8e4m3(src, layout, dst, mul, add)?,
                _ => unreachable!(),
            }
            Ok(true)
        })
    }

    pub fn try_powf(op: &Op1, value: f32) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (op, value);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        try_scalar(op, |dtype, src, layout, dst| {
            match dtype {
                KernelDType::F32 => crate::hip::powf_f32(src, layout, dst, value)?,
                KernelDType::BF16 => crate::hip::powf_bf16(src, layout, dst, value)?,
                KernelDType::F8E4M3 => crate::hip::powf_f8e4m3(src, layout, dst, value)?,
                _ => unreachable!(),
            }
            Ok(true)
        })
    }

    pub fn try_elu(op: &Op1, alpha: f32) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (op, alpha);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        try_scalar(op, |dtype, src, layout, dst| {
            match dtype {
                KernelDType::F32 => crate::hip::elu_f32(src, layout, dst, alpha)?,
                KernelDType::BF16 => crate::hip::elu_bf16(src, layout, dst, alpha)?,
                KernelDType::F8E4M3 => crate::hip::elu_f8e4m3(src, layout, dst, alpha)?,
                _ => unreachable!(),
            }
            Ok(true)
        })
    }

    pub fn try_reduce(op: &Op1, reduce_dims: &[usize]) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (op, reduce_dims);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        {
            let Some(layout) = op.input_layout.as_ref() else {
                return Ok(None);
            };
            let Some(output) = op.output else {
                return Ok(None);
            };
            let Some(code) = reduce_opcode(op.name) else {
                return Ok(None);
            };
            if !matches!(
                op.input.dtype(),
                KernelDType::F32 | KernelDType::BF16 | KernelDType::F8E4M3
            ) {
                return Ok(None);
            }
            let output_dtype = if matches!(code, 4 | 5) {
                KernelDType::U32
            } else {
                op.input.dtype()
            };
            if output.dtype() != output_dtype {
                return Ok(None);
            }
            check_layout(&op.input, layout)?;
            let mut reduce_mask = 0u64;
            let mut reduce_count = 1usize;
            for &dim in reduce_dims {
                if dim >= layout.dims().len() || dim >= u64::BITS as usize {
                    return Ok(None);
                }
                reduce_mask |= 1u64 << dim;
                reduce_count = reduce_count.saturating_mul(layout.dims()[dim]);
            }
            let dst = op.input.buffer().allocate_on_same_allocator(
                output.dtype().storage_size_in_bytes(output.elem_count()),
                false,
            )?;
            match op.input.dtype() {
                KernelDType::F32 => crate::hip::reduce_f32(
                    code,
                    op.input.buffer(),
                    layout,
                    reduce_mask,
                    reduce_count,
                    &dst,
                    output.elem_count(),
                )?,
                KernelDType::BF16 => crate::hip::reduce_bf16(
                    code,
                    op.input.buffer(),
                    layout,
                    reduce_mask,
                    reduce_count,
                    &dst,
                    output.elem_count(),
                )?,
                KernelDType::F8E4M3 => crate::hip::reduce_f8e4m3(
                    code,
                    op.input.buffer(),
                    layout,
                    reduce_mask,
                    reduce_count,
                    &dst,
                    output.elem_count(),
                )?,
                _ => unreachable!(),
            }
            Ok(Some(dst))
        }
    }

    pub fn try_to_dtype(op: &Op1) -> crate::Result<Option<Buffer>> {
        let Some(layout) = op.input_layout.as_ref() else {
            return Ok(None);
        };
        let Some(output) = op.output else {
            return Ok(None);
        };
        if op.input.dtype() == output.dtype() {
            return try_clone(op);
        }
        #[cfg(not(hip_runtime))]
        {
            let _ = (layout, output);
        }
        #[cfg(hip_runtime)]
        {
            let supported_cast = matches!(
                (op.input.dtype(), output.dtype()),
                (KernelDType::F32, KernelDType::BF16)
                    | (KernelDType::BF16, KernelDType::F32)
                    | (KernelDType::F32, KernelDType::F16)
                    | (KernelDType::F16, KernelDType::F32)
                    | (KernelDType::F32, KernelDType::F8E4M3)
                    | (KernelDType::F8E4M3, KernelDType::F32)
                    | (KernelDType::BF16, KernelDType::F8E4M3)
                    | (KernelDType::F8E4M3, KernelDType::BF16)
            );
            if !supported_cast || output.elem_count() != layout.elem_count() {
                return Ok(None);
            }
            check_layout(&op.input, layout)?;
            let dst = op.input.buffer().allocate_on_same_allocator(
                output.dtype().storage_size_in_bytes(output.elem_count()),
                false,
            )?;
            match (op.input.dtype(), output.dtype()) {
                (KernelDType::F32, KernelDType::BF16) => {
                    crate::hip::cast_f32_to_bf16(op.input.buffer(), layout, &dst)?
                }
                (KernelDType::BF16, KernelDType::F32) => {
                    crate::hip::cast_bf16_to_f32(op.input.buffer(), layout, &dst)?
                }
                (KernelDType::F32, KernelDType::F16) => {
                    crate::hip::cast_f32_to_f16(op.input.buffer(), layout, &dst)?
                }
                (KernelDType::F16, KernelDType::F32) => {
                    crate::hip::cast_f16_to_f32(op.input.buffer(), layout, &dst)?
                }
                (KernelDType::F32, KernelDType::F8E4M3) => {
                    crate::hip::cast_f32_to_f8e4m3(op.input.buffer(), layout, &dst)?
                }
                (KernelDType::F8E4M3, KernelDType::F32) => {
                    crate::hip::cast_f8e4m3_to_f32(op.input.buffer(), layout, &dst)?
                }
                (KernelDType::BF16, KernelDType::F8E4M3) => {
                    let tmp = op.input.buffer().allocate_on_same_allocator(
                        KernelDType::F32.storage_size_in_bytes(output.elem_count()),
                        false,
                    )?;
                    crate::hip::cast_bf16_to_f32(op.input.buffer(), layout, &tmp)?;
                    let contiguous = LayoutArg::new(vec![output.elem_count()], vec![1], 0)?;
                    crate::hip::cast_f32_to_f8e4m3(&tmp, &contiguous, &dst)?;
                }
                (KernelDType::F8E4M3, KernelDType::BF16) => {
                    let tmp = op.input.buffer().allocate_on_same_allocator(
                        KernelDType::F32.storage_size_in_bytes(output.elem_count()),
                        false,
                    )?;
                    crate::hip::cast_f8e4m3_to_f32(op.input.buffer(), layout, &tmp)?;
                    let contiguous = LayoutArg::new(vec![output.elem_count()], vec![1], 0)?;
                    crate::hip::cast_f32_to_bf16(&tmp, &contiguous, &dst)?;
                }
                _ => unreachable!(),
            }
            return Ok(Some(dst));
        }
        #[allow(unreachable_code)]
        Ok(None)
    }

    pub fn try_unary(op: &Op1) -> crate::Result<Option<Buffer>> {
        #[cfg(hip_runtime)]
        if let Some(buffer) = try_unary_hip(op)? {
            return Ok(Some(buffer));
        }
        try_unary_host(op)
    }

    fn try_unary_host(op: &Op1) -> crate::Result<Option<Buffer>> {
        let Some(layout) = op.input_layout.as_ref() else {
            return Ok(None);
        };
        let Some(output) = op.output else {
            return Ok(None);
        };
        if op.input.dtype() != KernelDType::F32 || output.dtype() != KernelDType::F32 {
            return Ok(None);
        }
        let input = read_f32(&op.input)?;
        check_layout(&op.input, layout)?;
        let mut output_values = Vec::with_capacity(output.elem_count());
        for storage_index in layout.storage_indices() {
            let value = input[storage_index];
            let value = match op.name {
                "abs" => value.abs(),
                "ceil" => value.ceil(),
                "cos" => value.cos(),
                "exp" => value.exp(),
                "floor" => value.floor(),
                "log" => value.ln(),
                "neg" => -value,
                "recip" => value.recip(),
                "relu" => value.max(0.),
                "round" => value.round(),
                "sin" => value.sin(),
                "sqr" => value * value,
                "sqrt" => value.sqrt(),
                "tanh" => value.tanh(),
                _ => return Ok(None),
            };
            output_values.push(value);
        }
        Ok(Some(write_f32_output(
            op.input.buffer(),
            output,
            &output_values,
        )?))
    }

    pub fn try_binary(op: &Op2) -> crate::Result<Option<Buffer>> {
        #[cfg(hip_runtime)]
        if let Some(buffer) = try_binary_hip(op)? {
            return Ok(Some(buffer));
        }
        try_binary_host(op)
    }

    fn try_binary_host(op: &Op2) -> crate::Result<Option<Buffer>> {
        let Some(lhs_layout) = op.lhs_layout.as_ref() else {
            return Ok(None);
        };
        let Some(rhs_layout) = op.rhs_layout.as_ref() else {
            return Ok(None);
        };
        let Some(output) = op.output else {
            return Ok(None);
        };
        if op.lhs.dtype() != KernelDType::F32
            || op.rhs.dtype() != KernelDType::F32
            || output.dtype() != KernelDType::F32
        {
            return Ok(None);
        }
        check_layout(&op.lhs, lhs_layout)?;
        check_layout(&op.rhs, rhs_layout)?;
        if lhs_layout.elem_count() != rhs_layout.elem_count()
            || lhs_layout.elem_count() != output.elem_count()
        {
            return Ok(None);
        }
        let lhs = read_f32(&op.lhs)?;
        let rhs = read_f32(&op.rhs)?;
        let mut output_values = Vec::with_capacity(output.elem_count());
        for (lhs_index, rhs_index) in lhs_layout
            .storage_indices()
            .zip(rhs_layout.storage_indices())
        {
            let lhs = lhs[lhs_index];
            let rhs = rhs[rhs_index];
            let value = match op.name {
                "add" => lhs + rhs,
                "div" => lhs / rhs,
                "maximum" => lhs.max(rhs),
                "minimum" => lhs.min(rhs),
                "mul" => lhs * rhs,
                "sub" => lhs - rhs,
                _ => return Ok(None),
            };
            output_values.push(value);
        }
        Ok(Some(write_f32_output(
            op.lhs.buffer(),
            output,
            &output_values,
        )?))
    }

    pub fn try_cmp(op: &Op2) -> crate::Result<Option<Buffer>> {
        #[cfg(hip_runtime)]
        if let Some(buffer) = try_cmp_hip(op)? {
            return Ok(Some(buffer));
        }
        try_cmp_host(op)
    }

    pub fn try_where_cond(
        op: &Op3,
        cond_layout: &LayoutArg,
        true_layout: &LayoutArg,
        false_layout: &LayoutArg,
    ) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (op, cond_layout, true_layout, false_layout);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        {
            let Some(output) = op.output else {
                return Ok(None);
            };
            if op.first.dtype() != KernelDType::U8
                || op.second.dtype() != op.third.dtype()
                || op.second.dtype() != output.dtype()
                || !matches!(
                    op.second.dtype(),
                    KernelDType::F32 | KernelDType::BF16 | KernelDType::F8E4M3
                )
            {
                return Ok(None);
            }
            if cond_layout.elem_count() != output.elem_count()
                || true_layout.elem_count() != output.elem_count()
                || false_layout.elem_count() != output.elem_count()
            {
                return Ok(None);
            }
            check_layout(&op.first, cond_layout)?;
            check_layout(&op.second, true_layout)?;
            check_layout(&op.third, false_layout)?;
            let dst = op.first.buffer().allocate_on_same_allocator(
                output.dtype().storage_size_in_bytes(output.elem_count()),
                false,
            )?;
            match op.second.dtype() {
                KernelDType::F32 => crate::hip::where_u8_f32(
                    op.first.buffer(),
                    cond_layout,
                    op.second.buffer(),
                    true_layout,
                    op.third.buffer(),
                    false_layout,
                    &dst,
                )?,
                KernelDType::BF16 => crate::hip::where_u8_bf16(
                    op.first.buffer(),
                    cond_layout,
                    op.second.buffer(),
                    true_layout,
                    op.third.buffer(),
                    false_layout,
                    &dst,
                )?,
                KernelDType::F8E4M3 => crate::hip::where_u8_f8e4m3(
                    op.first.buffer(),
                    cond_layout,
                    op.second.buffer(),
                    true_layout,
                    op.third.buffer(),
                    false_layout,
                    &dst,
                )?,
                _ => unreachable!(),
            }
            Ok(Some(dst))
        }
    }

    pub fn try_arg_sort(op: &Op1, asc: bool, last_dim: usize) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (op, asc, last_dim);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        {
            let Some(layout) = op.input_layout.as_ref() else {
                return Ok(None);
            };
            let Some(output) = op.output else {
                return Ok(None);
            };
            if !matches!(
                op.input.dtype(),
                KernelDType::F32 | KernelDType::BF16 | KernelDType::F8E4M3
            ) || output.dtype() != KernelDType::U32
                || output.elem_count() != layout.elem_count()
                || layout.dims().last().copied() != Some(last_dim)
                || last_dim == 0
                || !is_contiguous(layout)
            {
                return Ok(None);
            }
            check_layout(&op.input, layout)?;
            let dst = op.input.buffer().allocate_on_same_allocator(
                output.dtype().storage_size_in_bytes(output.elem_count()),
                false,
            )?;
            match op.input.dtype() {
                KernelDType::F32 => {
                    crate::hip::arg_sort_f32(op.input.buffer(), layout, &dst, asc, last_dim)?
                }
                KernelDType::BF16 => {
                    crate::hip::arg_sort_bf16(op.input.buffer(), layout, &dst, asc, last_dim)?
                }
                KernelDType::F8E4M3 => {
                    crate::hip::arg_sort_f8e4m3(op.input.buffer(), layout, &dst, asc, last_dim)?
                }
                _ => unreachable!(),
            }
            Ok(Some(dst))
        }
    }

    pub fn try_conv1d(op: &Op2, params: Conv1dParams) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (op, params);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        {
            let Some(src_layout) = op.lhs_layout.as_ref() else {
                return Ok(None);
            };
            let Some(kernel_layout) = op.rhs_layout.as_ref() else {
                return Ok(None);
            };
            let Some(output) = op.output else {
                return Ok(None);
            };
            if op.lhs.dtype() != op.rhs.dtype()
                || op.lhs.dtype() != output.dtype()
                || !matches!(
                    op.lhs.dtype(),
                    KernelDType::F32 | KernelDType::BF16 | KernelDType::F8E4M3
                )
                || src_layout.dims().len() != 3
                || kernel_layout.dims().len() != 3
                || src_layout.dims()[1] != kernel_layout.dims()[1]
                || output.elem_count() != params.elem_count
                || params.elem_count
                    != src_layout.dims()[0] * kernel_layout.dims()[0] * params.l_out
                || params.stride == 0
                || params.dilation == 0
            {
                return Ok(None);
            }
            check_layout(&op.lhs, src_layout)?;
            check_layout(&op.rhs, kernel_layout)?;
            let dst = op.lhs.buffer().allocate_on_same_allocator(
                output.dtype().storage_size_in_bytes(output.elem_count()),
                false,
            )?;
            match op.lhs.dtype() {
                KernelDType::F32 => crate::hip::conv1d_f32(
                    op.lhs.buffer(),
                    src_layout,
                    op.rhs.buffer(),
                    kernel_layout,
                    &dst,
                    params.padding,
                    params.stride,
                    params.dilation,
                    params.l_out,
                    params.elem_count,
                )?,
                KernelDType::BF16 => crate::hip::conv1d_bf16(
                    op.lhs.buffer(),
                    src_layout,
                    op.rhs.buffer(),
                    kernel_layout,
                    &dst,
                    params.padding,
                    params.stride,
                    params.dilation,
                    params.l_out,
                    params.elem_count,
                )?,
                KernelDType::F8E4M3 => crate::hip::conv1d_f8e4m3(
                    op.lhs.buffer(),
                    src_layout,
                    op.rhs.buffer(),
                    kernel_layout,
                    &dst,
                    params.padding,
                    params.stride,
                    params.dilation,
                    params.l_out,
                    params.elem_count,
                )?,
                _ => unreachable!(),
            }
            Ok(Some(dst))
        }
    }

    pub fn try_conv_transpose1d(op: &Op2, params: Conv1dParams) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (op, params);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        {
            let Some(src_layout) = op.lhs_layout.as_ref() else {
                return Ok(None);
            };
            let Some(kernel_layout) = op.rhs_layout.as_ref() else {
                return Ok(None);
            };
            let Some(output) = op.output else {
                return Ok(None);
            };
            if op.lhs.dtype() != op.rhs.dtype()
                || op.lhs.dtype() != output.dtype()
                || !matches!(
                    op.lhs.dtype(),
                    KernelDType::F32 | KernelDType::BF16 | KernelDType::F8E4M3
                )
                || src_layout.dims().len() != 3
                || kernel_layout.dims().len() != 3
                || src_layout.dims()[1] != kernel_layout.dims()[0]
                || output.elem_count() != params.elem_count
                || params.elem_count
                    != src_layout.dims()[0] * kernel_layout.dims()[1] * params.l_out
                || params.stride == 0
                || params.dilation == 0
            {
                return Ok(None);
            }
            check_layout(&op.lhs, src_layout)?;
            check_layout(&op.rhs, kernel_layout)?;
            let dst = op.lhs.buffer().allocate_on_same_allocator(
                output.dtype().storage_size_in_bytes(output.elem_count()),
                false,
            )?;
            match op.lhs.dtype() {
                KernelDType::F32 => crate::hip::conv_transpose1d_f32(
                    op.lhs.buffer(),
                    src_layout,
                    op.rhs.buffer(),
                    kernel_layout,
                    &dst,
                    params.padding,
                    params.stride,
                    params.dilation,
                    params.l_out,
                    params.elem_count,
                )?,
                KernelDType::BF16 => crate::hip::conv_transpose1d_bf16(
                    op.lhs.buffer(),
                    src_layout,
                    op.rhs.buffer(),
                    kernel_layout,
                    &dst,
                    params.padding,
                    params.stride,
                    params.dilation,
                    params.l_out,
                    params.elem_count,
                )?,
                KernelDType::F8E4M3 => crate::hip::conv_transpose1d_f8e4m3(
                    op.lhs.buffer(),
                    src_layout,
                    op.rhs.buffer(),
                    kernel_layout,
                    &dst,
                    params.padding,
                    params.stride,
                    params.dilation,
                    params.l_out,
                    params.elem_count,
                )?,
                _ => unreachable!(),
            }
            Ok(Some(dst))
        }
    }

    pub fn try_conv2d(op: &Op2, params: Conv2dParams) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (op, params);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        {
            let Some(src_layout) = op.lhs_layout.as_ref() else {
                return Ok(None);
            };
            let Some(kernel_layout) = op.rhs_layout.as_ref() else {
                return Ok(None);
            };
            let Some(output) = op.output else {
                return Ok(None);
            };
            if op.lhs.dtype() != op.rhs.dtype()
                || op.lhs.dtype() != output.dtype()
                || !matches!(
                    op.lhs.dtype(),
                    KernelDType::F32 | KernelDType::BF16 | KernelDType::F8E4M3
                )
                || src_layout.dims().len() != 4
                || kernel_layout.dims().len() != 4
                || src_layout.dims()[1] != kernel_layout.dims()[1]
                || output.elem_count() != params.elem_count
                || params.elem_count
                    != src_layout.dims()[0] * kernel_layout.dims()[0] * params.out_h * params.out_w
                || params.stride == 0
                || params.dilation == 0
            {
                return Ok(None);
            }
            check_layout(&op.lhs, src_layout)?;
            check_layout(&op.rhs, kernel_layout)?;
            let dst = op.lhs.buffer().allocate_on_same_allocator(
                output.dtype().storage_size_in_bytes(output.elem_count()),
                false,
            )?;
            match op.lhs.dtype() {
                KernelDType::F32 => crate::hip::conv2d_f32(
                    op.lhs.buffer(),
                    src_layout,
                    op.rhs.buffer(),
                    kernel_layout,
                    &dst,
                    params.padding,
                    params.stride,
                    params.dilation,
                    params.out_h,
                    params.out_w,
                    params.elem_count,
                )?,
                KernelDType::BF16 => crate::hip::conv2d_bf16(
                    op.lhs.buffer(),
                    src_layout,
                    op.rhs.buffer(),
                    kernel_layout,
                    &dst,
                    params.padding,
                    params.stride,
                    params.dilation,
                    params.out_h,
                    params.out_w,
                    params.elem_count,
                )?,
                KernelDType::F8E4M3 => crate::hip::conv2d_f8e4m3(
                    op.lhs.buffer(),
                    src_layout,
                    op.rhs.buffer(),
                    kernel_layout,
                    &dst,
                    params.padding,
                    params.stride,
                    params.dilation,
                    params.out_h,
                    params.out_w,
                    params.elem_count,
                )?,
                _ => unreachable!(),
            }
            Ok(Some(dst))
        }
    }

    pub fn try_conv_transpose2d(op: &Op2, params: Conv2dParams) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (op, params);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        {
            let Some(src_layout) = op.lhs_layout.as_ref() else {
                return Ok(None);
            };
            let Some(kernel_layout) = op.rhs_layout.as_ref() else {
                return Ok(None);
            };
            let Some(output) = op.output else {
                return Ok(None);
            };
            if op.lhs.dtype() != op.rhs.dtype()
                || op.lhs.dtype() != output.dtype()
                || !matches!(
                    op.lhs.dtype(),
                    KernelDType::F32 | KernelDType::BF16 | KernelDType::F8E4M3
                )
                || src_layout.dims().len() != 4
                || kernel_layout.dims().len() != 4
                || src_layout.dims()[1] != kernel_layout.dims()[0]
                || output.elem_count() != params.elem_count
                || params.elem_count
                    != src_layout.dims()[0] * kernel_layout.dims()[1] * params.out_h * params.out_w
                || params.stride == 0
                || params.dilation == 0
            {
                return Ok(None);
            }
            check_layout(&op.lhs, src_layout)?;
            check_layout(&op.rhs, kernel_layout)?;
            let dst = op.lhs.buffer().allocate_on_same_allocator(
                output.dtype().storage_size_in_bytes(output.elem_count()),
                false,
            )?;
            match op.lhs.dtype() {
                KernelDType::F32 => crate::hip::conv_transpose2d_f32(
                    op.lhs.buffer(),
                    src_layout,
                    op.rhs.buffer(),
                    kernel_layout,
                    &dst,
                    params.padding,
                    params.stride,
                    params.dilation,
                    params.out_h,
                    params.out_w,
                    params.elem_count,
                )?,
                KernelDType::BF16 => crate::hip::conv_transpose2d_bf16(
                    op.lhs.buffer(),
                    src_layout,
                    op.rhs.buffer(),
                    kernel_layout,
                    &dst,
                    params.padding,
                    params.stride,
                    params.dilation,
                    params.out_h,
                    params.out_w,
                    params.elem_count,
                )?,
                KernelDType::F8E4M3 => crate::hip::conv_transpose2d_f8e4m3(
                    op.lhs.buffer(),
                    src_layout,
                    op.rhs.buffer(),
                    kernel_layout,
                    &dst,
                    params.padding,
                    params.stride,
                    params.dilation,
                    params.out_h,
                    params.out_w,
                    params.elem_count,
                )?,
                _ => unreachable!(),
            }
            Ok(Some(dst))
        }
    }

    pub fn try_pool2d(
        op: &Op1,
        kernel: (usize, usize),
        stride: (usize, usize),
        max_pool: bool,
    ) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (op, kernel, stride, max_pool);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        {
            let Some(layout) = op.input_layout.as_ref() else {
                return Ok(None);
            };
            let Some(output) = op.output else {
                return Ok(None);
            };
            if op.input.dtype() != output.dtype()
                || !matches!(
                    op.input.dtype(),
                    KernelDType::F32 | KernelDType::BF16 | KernelDType::F8E4M3
                )
                || layout.dims().len() != 4
                || kernel.0 == 0
                || kernel.1 == 0
                || stride.0 == 0
                || stride.1 == 0
                || layout.dims()[2] < kernel.0
                || layout.dims()[3] < kernel.1
            {
                return Ok(None);
            }
            let out_h = (layout.dims()[2] - kernel.0) / stride.0 + 1;
            let out_w = (layout.dims()[3] - kernel.1) / stride.1 + 1;
            let elem_count = layout.dims()[0] * layout.dims()[1] * out_h * out_w;
            if output.elem_count() != elem_count {
                return Ok(None);
            }
            check_layout(&op.input, layout)?;
            let dst = op.input.buffer().allocate_on_same_allocator(
                output.dtype().storage_size_in_bytes(output.elem_count()),
                false,
            )?;
            match op.input.dtype() {
                KernelDType::F32 => crate::hip::pool2d_f32(
                    if max_pool { 2 } else { 1 },
                    op.input.buffer(),
                    layout,
                    &dst,
                    kernel,
                    stride,
                    out_h,
                    out_w,
                )?,
                KernelDType::BF16 => crate::hip::pool2d_bf16(
                    if max_pool { 2 } else { 1 },
                    op.input.buffer(),
                    layout,
                    &dst,
                    kernel,
                    stride,
                    out_h,
                    out_w,
                )?,
                KernelDType::F8E4M3 => crate::hip::pool2d_f8e4m3(
                    if max_pool { 2 } else { 1 },
                    op.input.buffer(),
                    layout,
                    &dst,
                    kernel,
                    stride,
                    out_h,
                    out_w,
                )?,
                _ => unreachable!(),
            }
            Ok(Some(dst))
        }
    }

    pub fn try_upsample_nearest1d(op: &Op1, out_size: usize) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (op, out_size);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        {
            let Some(layout) = op.input_layout.as_ref() else {
                return Ok(None);
            };
            let Some(output) = op.output else {
                return Ok(None);
            };
            if op.input.dtype() != output.dtype()
                || !matches!(
                    op.input.dtype(),
                    KernelDType::F32 | KernelDType::BF16 | KernelDType::F8E4M3
                )
                || layout.dims().len() != 3
                || out_size == 0
            {
                return Ok(None);
            }
            let elem_count = layout.dims()[0] * layout.dims()[1] * out_size;
            if output.elem_count() != elem_count {
                return Ok(None);
            }
            check_layout(&op.input, layout)?;
            let dst = op.input.buffer().allocate_on_same_allocator(
                output.dtype().storage_size_in_bytes(output.elem_count()),
                false,
            )?;
            match op.input.dtype() {
                KernelDType::F32 => {
                    crate::hip::upsample_nearest1d_f32(op.input.buffer(), layout, &dst, out_size)?
                }
                KernelDType::BF16 => {
                    crate::hip::upsample_nearest1d_bf16(op.input.buffer(), layout, &dst, out_size)?
                }
                KernelDType::F8E4M3 => crate::hip::upsample_nearest1d_f8e4m3(
                    op.input.buffer(),
                    layout,
                    &dst,
                    out_size,
                )?,
                _ => unreachable!(),
            }
            Ok(Some(dst))
        }
    }

    pub fn try_upsample_nearest2d(
        op: &Op1,
        out_h: usize,
        out_w: usize,
    ) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (op, out_h, out_w);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        {
            let Some(layout) = op.input_layout.as_ref() else {
                return Ok(None);
            };
            let Some(output) = op.output else {
                return Ok(None);
            };
            if op.input.dtype() != output.dtype()
                || !matches!(
                    op.input.dtype(),
                    KernelDType::F32 | KernelDType::BF16 | KernelDType::F8E4M3
                )
                || layout.dims().len() != 4
                || out_h == 0
                || out_w == 0
            {
                return Ok(None);
            }
            let elem_count = layout.dims()[0] * layout.dims()[1] * out_h * out_w;
            if output.elem_count() != elem_count {
                return Ok(None);
            }
            check_layout(&op.input, layout)?;
            let dst = op.input.buffer().allocate_on_same_allocator(
                output.dtype().storage_size_in_bytes(output.elem_count()),
                false,
            )?;
            match op.input.dtype() {
                KernelDType::F32 => crate::hip::upsample_nearest2d_f32(
                    op.input.buffer(),
                    layout,
                    &dst,
                    out_h,
                    out_w,
                )?,
                KernelDType::BF16 => crate::hip::upsample_nearest2d_bf16(
                    op.input.buffer(),
                    layout,
                    &dst,
                    out_h,
                    out_w,
                )?,
                KernelDType::F8E4M3 => crate::hip::upsample_nearest2d_f8e4m3(
                    op.input.buffer(),
                    layout,
                    &dst,
                    out_h,
                    out_w,
                )?,
                _ => unreachable!(),
            }
            Ok(Some(dst))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_upsample_bilinear2d(
        op: &Op1,
        out_h: usize,
        out_w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (op, out_h, out_w, align_corners, scale_h, scale_w);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        {
            let Some(layout) = op.input_layout.as_ref() else {
                return Ok(None);
            };
            let Some(output) = op.output else {
                return Ok(None);
            };
            if op.input.dtype() != output.dtype()
                || !matches!(
                    op.input.dtype(),
                    KernelDType::F32 | KernelDType::BF16 | KernelDType::F8E4M3
                )
                || layout.dims().len() != 4
                || out_h == 0
                || out_w == 0
            {
                return Ok(None);
            }
            let elem_count = layout.dims()[0] * layout.dims()[1] * out_h * out_w;
            if output.elem_count() != elem_count {
                return Ok(None);
            }
            let in_h = layout.dims()[2];
            let in_w = layout.dims()[3];
            let scale_h = if align_corners {
                if out_h > 1 {
                    (in_h - 1) as f64 / (out_h - 1) as f64
                } else {
                    0.0
                }
            } else {
                scale_h
                    .map(|scale| 1.0 / scale)
                    .unwrap_or(in_h as f64 / out_h as f64)
            };
            let scale_w = if align_corners {
                if out_w > 1 {
                    (in_w - 1) as f64 / (out_w - 1) as f64
                } else {
                    0.0
                }
            } else {
                scale_w
                    .map(|scale| 1.0 / scale)
                    .unwrap_or(in_w as f64 / out_w as f64)
            };
            check_layout(&op.input, layout)?;
            let dst = op.input.buffer().allocate_on_same_allocator(
                output.dtype().storage_size_in_bytes(output.elem_count()),
                false,
            )?;
            match op.input.dtype() {
                KernelDType::F32 => crate::hip::upsample_bilinear2d_f32(
                    op.input.buffer(),
                    layout,
                    &dst,
                    out_h,
                    out_w,
                    scale_h,
                    scale_w,
                    align_corners,
                )?,
                KernelDType::BF16 => crate::hip::upsample_bilinear2d_bf16(
                    op.input.buffer(),
                    layout,
                    &dst,
                    out_h,
                    out_w,
                    scale_h,
                    scale_w,
                    align_corners,
                )?,
                KernelDType::F8E4M3 => crate::hip::upsample_bilinear2d_f8e4m3(
                    op.input.buffer(),
                    layout,
                    &dst,
                    out_h,
                    out_w,
                    scale_h,
                    scale_w,
                    align_corners,
                )?,
                _ => unreachable!(),
            }
            Ok(Some(dst))
        }
    }

    pub fn try_index_select(op: &Op2, dim: usize) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (op, dim);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        {
            let Some(src_layout) = op.lhs_layout.as_ref() else {
                return Ok(None);
            };
            let Some(ids_layout) = op.rhs_layout.as_ref() else {
                return Ok(None);
            };
            let Some(output) = op.output else {
                return Ok(None);
            };
            if op.lhs.dtype() != output.dtype()
                || !matches!(
                    op.lhs.dtype(),
                    KernelDType::F32 | KernelDType::BF16 | KernelDType::F8E4M3
                )
            {
                return Ok(None);
            }
            if ids_layout.dims().len() != 1 || dim >= src_layout.dims().len() {
                return Ok(None);
            }
            let n_ids = ids_layout.dims()[0];
            if ids_layout.start_offset() + ids_layout.stride()[0].saturating_mul(n_ids)
                > op.rhs.elem_count()
            {
                return Ok(None);
            }
            check_layout(&op.lhs, src_layout)?;
            let dst = op.lhs.buffer().allocate_on_same_allocator(
                output.dtype().storage_size_in_bytes(output.elem_count()),
                false,
            )?;
            match (op.lhs.dtype(), op.rhs.dtype()) {
                (KernelDType::F32, KernelDType::U32) => crate::hip::index_select_u32_f32(
                    op.lhs.buffer(),
                    src_layout,
                    op.rhs.buffer(),
                    ids_layout.start_offset(),
                    ids_layout.stride()[0],
                    dim,
                    n_ids,
                    &dst,
                    output.elem_count(),
                )?,
                (KernelDType::F32, KernelDType::I64) => crate::hip::index_select_i64_f32(
                    op.lhs.buffer(),
                    src_layout,
                    op.rhs.buffer(),
                    ids_layout.start_offset(),
                    ids_layout.stride()[0],
                    dim,
                    n_ids,
                    &dst,
                    output.elem_count(),
                )?,
                (KernelDType::BF16, KernelDType::U32) => crate::hip::index_select_u32_bf16(
                    op.lhs.buffer(),
                    src_layout,
                    op.rhs.buffer(),
                    ids_layout.start_offset(),
                    ids_layout.stride()[0],
                    dim,
                    n_ids,
                    &dst,
                    output.elem_count(),
                )?,
                (KernelDType::BF16, KernelDType::I64) => crate::hip::index_select_i64_bf16(
                    op.lhs.buffer(),
                    src_layout,
                    op.rhs.buffer(),
                    ids_layout.start_offset(),
                    ids_layout.stride()[0],
                    dim,
                    n_ids,
                    &dst,
                    output.elem_count(),
                )?,
                (KernelDType::F8E4M3, KernelDType::U32) => crate::hip::index_select_u32_f8e4m3(
                    op.lhs.buffer(),
                    src_layout,
                    op.rhs.buffer(),
                    ids_layout.start_offset(),
                    ids_layout.stride()[0],
                    dim,
                    n_ids,
                    &dst,
                    output.elem_count(),
                )?,
                (KernelDType::F8E4M3, KernelDType::I64) => crate::hip::index_select_i64_f8e4m3(
                    op.lhs.buffer(),
                    src_layout,
                    op.rhs.buffer(),
                    ids_layout.start_offset(),
                    ids_layout.stride()[0],
                    dim,
                    n_ids,
                    &dst,
                    output.elem_count(),
                )?,
                _ => return Ok(None),
            }
            Ok(Some(dst))
        }
    }

    pub fn try_gather(op: &Op2, dim: usize) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (op, dim);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        {
            let Some(src_layout) = op.lhs_layout.as_ref() else {
                return Ok(None);
            };
            let Some(ids_layout) = op.rhs_layout.as_ref() else {
                return Ok(None);
            };
            let Some(output) = op.output else {
                return Ok(None);
            };
            if !matches!(
                op.lhs.dtype(),
                KernelDType::F32 | KernelDType::BF16 | KernelDType::F8E4M3
            ) || op.rhs.dtype() != KernelDType::U32
                || output.dtype() != op.lhs.dtype()
                || src_layout.dims().len() != ids_layout.dims().len()
                || dim >= src_layout.dims().len()
                || ids_layout.elem_count() != output.elem_count()
            {
                return Ok(None);
            }
            check_layout(&op.lhs, src_layout)?;
            check_layout(&op.rhs, ids_layout)?;
            let dst = op.lhs.buffer().allocate_on_same_allocator(
                output.dtype().storage_size_in_bytes(output.elem_count()),
                false,
            )?;
            match op.lhs.dtype() {
                KernelDType::F32 => crate::hip::gather_u32_f32(
                    op.lhs.buffer(),
                    src_layout,
                    op.rhs.buffer(),
                    ids_layout,
                    dim,
                    &dst,
                )?,
                KernelDType::BF16 => crate::hip::gather_u32_bf16(
                    op.lhs.buffer(),
                    src_layout,
                    op.rhs.buffer(),
                    ids_layout,
                    dim,
                    &dst,
                )?,
                KernelDType::F8E4M3 => crate::hip::gather_u32_f8e4m3(
                    op.lhs.buffer(),
                    src_layout,
                    op.rhs.buffer(),
                    ids_layout,
                    dim,
                    &dst,
                )?,
                _ => unreachable!(),
            }
            Ok(Some(dst))
        }
    }

    pub fn try_scatter(
        op: &InplaceOp3,
        dst_layout: &LayoutArg,
        ids_layout: &LayoutArg,
        src_layout: &LayoutArg,
        dim: usize,
        add: bool,
    ) -> crate::Result<bool> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (op, dst_layout, ids_layout, src_layout, dim, add);
            Ok(false)
        }
        #[cfg(hip_runtime)]
        {
            if !matches!(
                op.dst.dtype(),
                KernelDType::F32 | KernelDType::BF16 | KernelDType::F8E4M3
            ) || op.second.dtype() != KernelDType::U32
                || op.third.dtype() != op.dst.dtype()
                || dst_layout.dims().len() != ids_layout.dims().len()
                || ids_layout.dims().len() != src_layout.dims().len()
                || dim >= dst_layout.dims().len()
                || ids_layout.elem_count() != src_layout.elem_count()
            {
                return Ok(false);
            }
            for (axis, (&ids_dim, &src_dim)) in
                ids_layout.dims().iter().zip(src_layout.dims()).enumerate()
            {
                if ids_dim != src_dim || (axis != dim && ids_dim != dst_layout.dims()[axis]) {
                    return Ok(false);
                }
            }
            check_layout(&op.dst, dst_layout)?;
            check_layout(&op.second, ids_layout)?;
            check_layout(&op.third, src_layout)?;
            match op.dst.dtype() {
                KernelDType::F32 => crate::hip::scatter_u32_f32(
                    add,
                    op.dst.buffer(),
                    dst_layout,
                    op.second.buffer(),
                    ids_layout,
                    op.third.buffer(),
                    src_layout,
                    dim,
                )?,
                KernelDType::BF16 => crate::hip::scatter_u32_bf16(
                    add,
                    op.dst.buffer(),
                    dst_layout,
                    op.second.buffer(),
                    ids_layout,
                    op.third.buffer(),
                    src_layout,
                    dim,
                )?,
                KernelDType::F8E4M3 => crate::hip::scatter_u32_f8e4m3(
                    add,
                    op.dst.buffer(),
                    dst_layout,
                    op.second.buffer(),
                    ids_layout,
                    op.third.buffer(),
                    src_layout,
                    dim,
                )?,
                _ => unreachable!(),
            }
            Ok(true)
        }
    }

    pub fn try_index_add(
        op: &Op3,
        input_layout: &LayoutArg,
        ids_layout: &LayoutArg,
        src_layout: &LayoutArg,
        dim: usize,
    ) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (op, input_layout, ids_layout, src_layout, dim);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        {
            let Some(output) = op.output else {
                return Ok(None);
            };
            if !matches!(
                op.first.dtype(),
                KernelDType::F32 | KernelDType::BF16 | KernelDType::F8E4M3
            ) || op.second.dtype() != KernelDType::U32
                || op.third.dtype() != op.first.dtype()
                || output.dtype() != op.first.dtype()
                || ids_layout.dims().len() != 1
                || dim >= input_layout.dims().len()
                || input_layout.dims().len() != src_layout.dims().len()
                || src_layout.dims()[dim] != ids_layout.dims()[0]
                || input_layout.elem_count() != output.elem_count()
            {
                return Ok(None);
            }
            for (axis, (&input_dim, &src_dim)) in input_layout
                .dims()
                .iter()
                .zip(src_layout.dims())
                .enumerate()
            {
                if axis != dim && input_dim != src_dim {
                    return Ok(None);
                }
            }
            check_layout(&op.first, input_layout)?;
            check_layout(&op.second, ids_layout)?;
            check_layout(&op.third, src_layout)?;
            let dst = op.first.buffer().allocate_on_same_allocator(
                output.dtype().storage_size_in_bytes(output.elem_count()),
                false,
            )?;
            match op.first.dtype() {
                KernelDType::F32 => crate::hip::index_add_u32_f32(
                    op.first.buffer(),
                    input_layout,
                    op.second.buffer(),
                    ids_layout.start_offset(),
                    ids_layout.stride()[0],
                    ids_layout.dims()[0],
                    op.third.buffer(),
                    src_layout,
                    dim,
                    &dst,
                )?,
                KernelDType::BF16 => crate::hip::index_add_u32_bf16(
                    op.first.buffer(),
                    input_layout,
                    op.second.buffer(),
                    ids_layout.start_offset(),
                    ids_layout.stride()[0],
                    ids_layout.dims()[0],
                    op.third.buffer(),
                    src_layout,
                    dim,
                    &dst,
                )?,
                KernelDType::F8E4M3 => crate::hip::index_add_u32_f8e4m3(
                    op.first.buffer(),
                    input_layout,
                    op.second.buffer(),
                    ids_layout.start_offset(),
                    ids_layout.stride()[0],
                    ids_layout.dims()[0],
                    op.third.buffer(),
                    src_layout,
                    dim,
                    &dst,
                )?,
                _ => unreachable!(),
            }
            Ok(Some(dst))
        }
    }

    pub fn try_matmul(
        op: &Op2,
        bmnk: (usize, usize, usize, usize),
    ) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (op, bmnk);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        {
            let Some(lhs_layout) = op.lhs_layout.as_ref() else {
                return Ok(None);
            };
            let Some(rhs_layout) = op.rhs_layout.as_ref() else {
                return Ok(None);
            };
            let Some(output) = op.output else {
                return Ok(None);
            };
            if op.lhs.dtype() != op.rhs.dtype()
                || op.lhs.dtype() != output.dtype()
                || !matches!(
                    op.lhs.dtype(),
                    KernelDType::F32 | KernelDType::BF16 | KernelDType::F8E4M3
                )
            {
                return Ok(None);
            }
            let (b, m, n, k) = bmnk;
            if output.elem_count() != b * m * n
                || lhs_layout.dims().len() < 2
                || rhs_layout.dims().len() < 2
            {
                return Ok(None);
            }
            check_layout(&op.lhs, lhs_layout)?;
            check_layout(&op.rhs, rhs_layout)?;
            let Some((lhs_batch_stride, rhs_batch_stride)) =
                batch_strides(lhs_layout, rhs_layout, m, n, k)
            else {
                return Ok(None);
            };
            let lhs_rank = lhs_layout.dims().len();
            let rhs_rank = rhs_layout.dims().len();
            let lhs_row_stride = lhs_layout.stride()[lhs_rank - 2];
            let lhs_col_stride = lhs_layout.stride()[lhs_rank - 1];
            let rhs_row_stride = rhs_layout.stride()[rhs_rank - 2];
            let rhs_col_stride = rhs_layout.stride()[rhs_rank - 1];
            let dst = op.lhs.buffer().allocate_on_same_allocator(
                output.dtype().storage_size_in_bytes(output.elem_count()),
                false,
            )?;
            match op.lhs.dtype() {
                KernelDType::F32 => crate::hip::matmul_f32(
                    op.lhs.buffer(),
                    op.rhs.buffer(),
                    &dst,
                    bmnk,
                    lhs_layout.start_offset(),
                    rhs_layout.start_offset(),
                    lhs_batch_stride,
                    rhs_batch_stride,
                    lhs_row_stride,
                    lhs_col_stride,
                    rhs_row_stride,
                    rhs_col_stride,
                )?,
                KernelDType::BF16 => crate::hip::matmul_bf16(
                    op.lhs.buffer(),
                    op.rhs.buffer(),
                    &dst,
                    bmnk,
                    lhs_layout.start_offset(),
                    rhs_layout.start_offset(),
                    lhs_batch_stride,
                    rhs_batch_stride,
                    lhs_row_stride,
                    lhs_col_stride,
                    rhs_row_stride,
                    rhs_col_stride,
                )?,
                KernelDType::F8E4M3 => crate::hip::matmul_f8e4m3(
                    op.lhs.buffer(),
                    op.rhs.buffer(),
                    &dst,
                    bmnk,
                    lhs_layout.start_offset(),
                    rhs_layout.start_offset(),
                    lhs_batch_stride,
                    rhs_batch_stride,
                    lhs_row_stride,
                    lhs_col_stride,
                    rhs_row_stride,
                    rhs_col_stride,
                )?,
                _ => unreachable!(),
            }
            Ok(Some(dst))
        }
    }

    pub fn try_softmax_last_dim(op: &Op1) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = op;
            Ok(None)
        }
        #[cfg(hip_runtime)]
        {
            let Some(layout) = op.input_layout.as_ref() else {
                return Ok(None);
            };
            let Some(output) = op.output else {
                return Ok(None);
            };
            if op.input.dtype() != output.dtype()
                || !matches!(
                    op.input.dtype(),
                    KernelDType::F32 | KernelDType::BF16 | KernelDType::F8E4M3
                )
                || !is_contiguous(layout)
                || layout.dims().is_empty()
            {
                return Ok(None);
            }
            let cols = *layout.dims().last().unwrap();
            let rows = layout.elem_count() / cols;
            let dst = op.input.buffer().allocate_on_same_allocator(
                output.dtype().storage_size_in_bytes(output.elem_count()),
                false,
            )?;
            match op.input.dtype() {
                KernelDType::F32 => crate::hip::softmax_last_dim_f32(
                    op.input.buffer(),
                    layout.start_offset(),
                    &dst,
                    rows,
                    cols,
                )?,
                KernelDType::BF16 => crate::hip::softmax_last_dim_bf16(
                    op.input.buffer(),
                    layout.start_offset(),
                    &dst,
                    rows,
                    cols,
                )?,
                KernelDType::F8E4M3 => crate::hip::softmax_last_dim_f8e4m3(
                    op.input.buffer(),
                    layout.start_offset(),
                    &dst,
                    rows,
                    cols,
                )?,
                _ => unreachable!(),
            }
            Ok(Some(dst))
        }
    }

    pub fn try_repeat_penalty(op: &Op2, penalty: f32) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (op, penalty);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        {
            let Some(logits_layout) = op.lhs_layout.as_ref() else {
                return Ok(None);
            };
            let Some(token_ids_layout) = op.rhs_layout.as_ref() else {
                return Ok(None);
            };
            let Some(output) = op.output else {
                return Ok(None);
            };
            if op.lhs.dtype() != KernelDType::F32
                || op.rhs.dtype() != KernelDType::U32
                || output.dtype() != KernelDType::F32
                || logits_layout.dims().len() != 1
                || token_ids_layout.dims().len() != 1
                || output.elem_count() != logits_layout.elem_count()
            {
                return Ok(None);
            }
            check_layout(&op.lhs, logits_layout)?;
            check_layout(&op.rhs, token_ids_layout)?;
            let dst = op.lhs.buffer().allocate_on_same_allocator(
                output.dtype().storage_size_in_bytes(output.elem_count()),
                false,
            )?;
            crate::hip::repeat_penalty_f32(
                op.lhs.buffer(),
                logits_layout.start_offset(),
                logits_layout.stride()[0],
                op.rhs.buffer(),
                token_ids_layout.start_offset(),
                token_ids_layout.stride()[0],
                &dst,
                output.elem_count(),
                token_ids_layout.elem_count(),
                penalty,
            )?;
            Ok(Some(dst))
        }
    }

    pub fn try_rms_norm(op: &Op2, eps: f32) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (op, eps);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        {
            let Some(src_layout) = op.lhs_layout.as_ref() else {
                return Ok(None);
            };
            let Some(alpha_layout) = op.rhs_layout.as_ref() else {
                return Ok(None);
            };
            let Some(output) = op.output else {
                return Ok(None);
            };
            if op.lhs.dtype() != op.rhs.dtype()
                || op.lhs.dtype() != output.dtype()
                || !matches!(
                    op.lhs.dtype(),
                    KernelDType::F32 | KernelDType::BF16 | KernelDType::F8E4M3
                )
                || !is_contiguous(src_layout)
                || !is_contiguous(alpha_layout)
                || src_layout.dims().is_empty()
            {
                return Ok(None);
            }
            let cols = *src_layout.dims().last().unwrap();
            if alpha_layout.elem_count() != cols {
                return Ok(None);
            }
            let rows = src_layout.elem_count() / cols;
            let dst = op.lhs.buffer().allocate_on_same_allocator(
                output.dtype().storage_size_in_bytes(output.elem_count()),
                false,
            )?;
            match op.lhs.dtype() {
                KernelDType::F32 => crate::hip::rms_norm_f32(
                    op.lhs.buffer(),
                    src_layout.start_offset(),
                    op.rhs.buffer(),
                    alpha_layout.start_offset(),
                    &dst,
                    rows,
                    cols,
                    eps,
                )?,
                KernelDType::BF16 => crate::hip::rms_norm_bf16(
                    op.lhs.buffer(),
                    src_layout.start_offset(),
                    op.rhs.buffer(),
                    alpha_layout.start_offset(),
                    &dst,
                    rows,
                    cols,
                    eps,
                )?,
                KernelDType::F8E4M3 => crate::hip::rms_norm_f8e4m3(
                    op.lhs.buffer(),
                    src_layout.start_offset(),
                    op.rhs.buffer(),
                    alpha_layout.start_offset(),
                    &dst,
                    rows,
                    cols,
                    eps,
                )?,
                _ => unreachable!(),
            }
            Ok(Some(dst))
        }
    }

    pub fn try_layer_norm(
        op: &Op3,
        src_layout: &LayoutArg,
        alpha_layout: &LayoutArg,
        beta_layout: &LayoutArg,
        eps: f32,
    ) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (op, src_layout, alpha_layout, beta_layout, eps);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        {
            let Some(output) = op.output else {
                return Ok(None);
            };
            if op.first.dtype() != op.second.dtype()
                || op.first.dtype() != op.third.dtype()
                || op.first.dtype() != output.dtype()
                || !matches!(
                    op.first.dtype(),
                    KernelDType::F32 | KernelDType::BF16 | KernelDType::F8E4M3
                )
                || !is_contiguous(src_layout)
                || !is_contiguous(alpha_layout)
                || !is_contiguous(beta_layout)
                || src_layout.dims().is_empty()
            {
                return Ok(None);
            }
            let cols = *src_layout.dims().last().unwrap();
            if alpha_layout.elem_count() != cols || beta_layout.elem_count() != cols {
                return Ok(None);
            }
            let rows = src_layout.elem_count() / cols;
            let dst = op.first.buffer().allocate_on_same_allocator(
                output.dtype().storage_size_in_bytes(output.elem_count()),
                false,
            )?;
            match op.first.dtype() {
                KernelDType::F32 => crate::hip::layer_norm_f32(
                    op.first.buffer(),
                    src_layout.start_offset(),
                    op.second.buffer(),
                    alpha_layout.start_offset(),
                    op.third.buffer(),
                    beta_layout.start_offset(),
                    &dst,
                    rows,
                    cols,
                    eps,
                )?,
                KernelDType::BF16 => crate::hip::layer_norm_bf16(
                    op.first.buffer(),
                    src_layout.start_offset(),
                    op.second.buffer(),
                    alpha_layout.start_offset(),
                    op.third.buffer(),
                    beta_layout.start_offset(),
                    &dst,
                    rows,
                    cols,
                    eps,
                )?,
                KernelDType::F8E4M3 => crate::hip::layer_norm_f8e4m3(
                    op.first.buffer(),
                    src_layout.start_offset(),
                    op.second.buffer(),
                    alpha_layout.start_offset(),
                    op.third.buffer(),
                    beta_layout.start_offset(),
                    &dst,
                    rows,
                    cols,
                    eps,
                )?,
                _ => unreachable!(),
            }
            Ok(Some(dst))
        }
    }

    pub fn try_rope(
        op: &Op3,
        src_layout: &LayoutArg,
        cos_layout: &LayoutArg,
        sin_layout: &LayoutArg,
        interleaved: bool,
        thd: bool,
    ) -> crate::Result<Option<Buffer>> {
        #[cfg(not(hip_runtime))]
        {
            let _ = (op, src_layout, cos_layout, sin_layout, interleaved, thd);
            Ok(None)
        }
        #[cfg(hip_runtime)]
        {
            let Some(output) = op.output else {
                return Ok(None);
            };
            if op.first.dtype() != op.second.dtype()
                || op.first.dtype() != op.third.dtype()
                || op.first.dtype() != output.dtype()
                || !matches!(
                    op.first.dtype(),
                    KernelDType::F32 | KernelDType::BF16 | KernelDType::F8E4M3
                )
                || src_layout.dims().len() != 4
                || !is_contiguous(src_layout)
                || !is_contiguous(cos_layout)
                || !is_contiguous(sin_layout)
            {
                return Ok(None);
            }
            let dims = src_layout.dims();
            let (b, h, t, d) = if thd {
                (dims[0], dims[2], dims[1], dims[3])
            } else {
                (dims[0], dims[1], dims[2], dims[3])
            };
            let unbatched_rope = cos_layout.dims().len() == 3 && sin_layout.dims().len() == 3;
            let dst = op.first.buffer().allocate_on_same_allocator(
                output.dtype().storage_size_in_bytes(output.elem_count()),
                false,
            )?;
            match op.first.dtype() {
                KernelDType::F32 => crate::hip::rope_f32(
                    op.first.buffer(),
                    src_layout.start_offset(),
                    op.second.buffer(),
                    cos_layout.start_offset(),
                    op.third.buffer(),
                    sin_layout.start_offset(),
                    &dst,
                    b,
                    h,
                    t,
                    d,
                    interleaved,
                    unbatched_rope,
                    thd,
                )?,
                KernelDType::BF16 => crate::hip::rope_bf16(
                    op.first.buffer(),
                    src_layout.start_offset(),
                    op.second.buffer(),
                    cos_layout.start_offset(),
                    op.third.buffer(),
                    sin_layout.start_offset(),
                    &dst,
                    b,
                    h,
                    t,
                    d,
                    interleaved,
                    unbatched_rope,
                    thd,
                )?,
                KernelDType::F8E4M3 => crate::hip::rope_f8e4m3(
                    op.first.buffer(),
                    src_layout.start_offset(),
                    op.second.buffer(),
                    cos_layout.start_offset(),
                    op.third.buffer(),
                    sin_layout.start_offset(),
                    &dst,
                    b,
                    h,
                    t,
                    d,
                    interleaved,
                    unbatched_rope,
                    thd,
                )?,
                _ => unreachable!(),
            }
            Ok(Some(dst))
        }
    }

    fn try_cmp_host(op: &Op2) -> crate::Result<Option<Buffer>> {
        let Some(lhs_layout) = op.lhs_layout.as_ref() else {
            return Ok(None);
        };
        let Some(rhs_layout) = op.rhs_layout.as_ref() else {
            return Ok(None);
        };
        let Some(output) = op.output else {
            return Ok(None);
        };
        if op.lhs.dtype() != KernelDType::F32
            || op.rhs.dtype() != KernelDType::F32
            || output.dtype() != KernelDType::U8
        {
            return Ok(None);
        }
        check_layout(&op.lhs, lhs_layout)?;
        check_layout(&op.rhs, rhs_layout)?;
        if lhs_layout.elem_count() != rhs_layout.elem_count()
            || lhs_layout.elem_count() != output.elem_count()
        {
            return Ok(None);
        }
        let lhs = read_f32(&op.lhs)?;
        let rhs = read_f32(&op.rhs)?;
        let mut output_values = Vec::with_capacity(output.elem_count());
        for (lhs_index, rhs_index) in lhs_layout
            .storage_indices()
            .zip(rhs_layout.storage_indices())
        {
            let lhs = lhs[lhs_index];
            let rhs = rhs[rhs_index];
            let value = match op.name {
                "eq" => lhs == rhs,
                "ge" => lhs >= rhs,
                "gt" => lhs > rhs,
                "le" => lhs <= rhs,
                "lt" => lhs < rhs,
                "ne" => lhs != rhs,
                _ => return Ok(None),
            };
            output_values.push(u8::from(value));
        }
        let dst = op
            .lhs
            .buffer()
            .allocate_on_same_allocator(output.elem_count(), false)?;
        dst.write_all(&output_values)?;
        Ok(Some(dst))
    }

    #[cfg(hip_runtime)]
    fn try_unary_hip(op: &Op1) -> crate::Result<Option<Buffer>> {
        let Some(layout) = op.input_layout.as_ref() else {
            return Ok(None);
        };
        let Some(output) = op.output else {
            return Ok(None);
        };
        let Some(code) = unary_opcode(op.name) else {
            return Ok(None);
        };
        if op.input.dtype() != output.dtype()
            || !matches!(
                op.input.dtype(),
                KernelDType::F32 | KernelDType::BF16 | KernelDType::F8E4M3
            )
        {
            return Ok(None);
        }
        if layout.elem_count() != output.elem_count() {
            return Ok(None);
        }
        check_layout(&op.input, layout)?;
        let dst = op.input.buffer().allocate_on_same_allocator(
            output.dtype().storage_size_in_bytes(output.elem_count()),
            false,
        )?;
        match op.input.dtype() {
            KernelDType::F32 => crate::hip::unary_f32(code, op.input.buffer(), layout, &dst)?,
            KernelDType::BF16 => crate::hip::unary_bf16(code, op.input.buffer(), layout, &dst)?,
            KernelDType::F8E4M3 => crate::hip::unary_f8e4m3(code, op.input.buffer(), layout, &dst)?,
            _ => unreachable!(),
        }
        Ok(Some(dst))
    }

    #[cfg(hip_runtime)]
    fn try_binary_hip(op: &Op2) -> crate::Result<Option<Buffer>> {
        let Some(lhs_layout) = op.lhs_layout.as_ref() else {
            return Ok(None);
        };
        let Some(rhs_layout) = op.rhs_layout.as_ref() else {
            return Ok(None);
        };
        let Some(output) = op.output else {
            return Ok(None);
        };
        let Some(code) = binary_opcode(op.name) else {
            return Ok(None);
        };
        if op.lhs.dtype() != op.rhs.dtype()
            || op.lhs.dtype() != output.dtype()
            || !matches!(
                op.lhs.dtype(),
                KernelDType::F32 | KernelDType::BF16 | KernelDType::F8E4M3
            )
        {
            return Ok(None);
        }
        if lhs_layout.elem_count() != rhs_layout.elem_count()
            || lhs_layout.elem_count() != output.elem_count()
        {
            return Ok(None);
        }
        check_layout(&op.lhs, lhs_layout)?;
        check_layout(&op.rhs, rhs_layout)?;
        let dst = op.lhs.buffer().allocate_on_same_allocator(
            output.dtype().storage_size_in_bytes(output.elem_count()),
            false,
        )?;
        match op.lhs.dtype() {
            KernelDType::F32 => crate::hip::binary_f32(
                code,
                op.lhs.buffer(),
                lhs_layout,
                op.rhs.buffer(),
                rhs_layout,
                &dst,
            )?,
            KernelDType::BF16 => crate::hip::binary_bf16(
                code,
                op.lhs.buffer(),
                lhs_layout,
                op.rhs.buffer(),
                rhs_layout,
                &dst,
            )?,
            KernelDType::F8E4M3 => crate::hip::binary_f8e4m3(
                code,
                op.lhs.buffer(),
                lhs_layout,
                op.rhs.buffer(),
                rhs_layout,
                &dst,
            )?,
            _ => unreachable!(),
        }
        Ok(Some(dst))
    }

    #[cfg(hip_runtime)]
    fn try_cmp_hip(op: &Op2) -> crate::Result<Option<Buffer>> {
        let Some(lhs_layout) = op.lhs_layout.as_ref() else {
            return Ok(None);
        };
        let Some(rhs_layout) = op.rhs_layout.as_ref() else {
            return Ok(None);
        };
        let Some(output) = op.output else {
            return Ok(None);
        };
        let Some(code) = cmp_opcode(op.name) else {
            return Ok(None);
        };
        if op.lhs.dtype() != op.rhs.dtype()
            || !matches!(
                op.lhs.dtype(),
                KernelDType::F32 | KernelDType::BF16 | KernelDType::F8E4M3
            )
            || output.dtype() != KernelDType::U8
        {
            return Ok(None);
        }
        if lhs_layout.elem_count() != rhs_layout.elem_count()
            || lhs_layout.elem_count() != output.elem_count()
        {
            return Ok(None);
        }
        check_layout(&op.lhs, lhs_layout)?;
        check_layout(&op.rhs, rhs_layout)?;
        let dst = op
            .lhs
            .buffer()
            .allocate_on_same_allocator(output.elem_count(), false)?;
        match op.lhs.dtype() {
            KernelDType::F32 => crate::hip::cmp_f32(
                code,
                op.lhs.buffer(),
                lhs_layout,
                op.rhs.buffer(),
                rhs_layout,
                &dst,
            )?,
            KernelDType::BF16 => crate::hip::cmp_bf16(
                code,
                op.lhs.buffer(),
                lhs_layout,
                op.rhs.buffer(),
                rhs_layout,
                &dst,
            )?,
            KernelDType::F8E4M3 => crate::hip::cmp_f8e4m3(
                code,
                op.lhs.buffer(),
                lhs_layout,
                op.rhs.buffer(),
                rhs_layout,
                &dst,
            )?,
            _ => unreachable!(),
        }
        Ok(Some(dst))
    }

    trait LayoutIndices {
        fn storage_indices(&self) -> LayoutIndexIter<'_>;
    }

    impl LayoutIndices for LayoutArg {
        fn storage_indices(&self) -> LayoutIndexIter<'_> {
            LayoutIndexIter::new(self)
        }
    }

    struct LayoutIndexIter<'a> {
        layout: &'a LayoutArg,
        next_logical_index: usize,
        elem_count: usize,
    }

    impl<'a> LayoutIndexIter<'a> {
        fn new(layout: &'a LayoutArg) -> Self {
            Self {
                layout,
                next_logical_index: 0,
                elem_count: layout.elem_count(),
            }
        }
    }

    impl Iterator for LayoutIndexIter<'_> {
        type Item = usize;

        fn next(&mut self) -> Option<Self::Item> {
            if self.next_logical_index >= self.elem_count {
                return None;
            }
            let storage_index = self.layout.storage_index(self.next_logical_index);
            self.next_logical_index += 1;
            Some(storage_index)
        }
    }

    fn transfer(op: &TransferOp<'_>) -> crate::Result<Buffer> {
        if op.src.dtype() != op.output.dtype() || op.src.elem_count() != op.output.elem_count() {
            return Err(RocmError::UnsupportedDType {
                dtype: op.src.dtype(),
                op: "transfer",
            });
        }
        #[cfg(hip_runtime)]
        {
            let dst = op.dst_device.allocate(
                op.output
                    .dtype()
                    .storage_size_in_bytes(op.output.elem_count()),
            )?;
            crate::hip::copy_d2d(op.src.buffer(), &dst, op.src.buffer().size_in_bytes())?;
            Ok(dst)
        }
        #[cfg(not(hip_runtime))]
        {
            let mut bytes = vec![0; op.src.buffer().size_in_bytes()];
            op.src.buffer().read_all(&mut bytes)?;
            op.dst_device.copy_from_host(&bytes)
        }
    }

    fn const_set(op: &InplaceOp1) -> crate::Result<()> {
        let Some(layout) = op.dst_layout.as_ref() else {
            return Err(RocmError::NotImplemented("const_set without layout"));
        };
        let Some(scalar) = op.scalar else {
            return Err(RocmError::NotImplemented("const_set without scalar"));
        };
        if op.dst.dtype() != scalar.dtype() {
            return Err(RocmError::UnsupportedDType {
                dtype: op.dst.dtype(),
                op: "const_set",
            });
        }
        check_layout(&op.dst, layout)?;
        #[cfg(hip_runtime)]
        {
            match scalar {
                KernelScalar::F32(value) => {
                    crate::hip::const_set_f32(op.dst.buffer(), layout, value)?;
                }
                KernelScalar::U8(value) => {
                    crate::hip::const_set_u8(op.dst.buffer(), layout, value)?;
                }
                KernelScalar::U32(value) => {
                    crate::hip::const_set_u32(op.dst.buffer(), layout, value)?;
                }
                KernelScalar::I16(value) => {
                    crate::hip::const_set_i16(op.dst.buffer(), layout, value)?;
                }
                KernelScalar::I32(value) => {
                    crate::hip::const_set_i32(op.dst.buffer(), layout, value)?;
                }
                KernelScalar::I64(value) => {
                    crate::hip::const_set_i64(op.dst.buffer(), layout, value)?;
                }
                KernelScalar::BF16(value) => {
                    crate::hip::const_set_bf16(op.dst.buffer(), layout, value)?;
                }
                KernelScalar::F16(value) => {
                    crate::hip::const_set_f16(op.dst.buffer(), layout, value)?;
                }
                KernelScalar::F64(value) => {
                    crate::hip::const_set_f64(op.dst.buffer(), layout, value)?;
                }
                KernelScalar::F8E4M3(value) => {
                    crate::hip::const_set_f8e4m3(op.dst.buffer(), layout, value)?;
                }
            }
            Ok(())
        }
        #[cfg(not(hip_runtime))]
        {
            const_set_host(&op.dst, layout, scalar)
        }
    }

    #[cfg(not(hip_runtime))]
    fn const_set_host(
        dst: &TensorArg,
        layout: &LayoutArg,
        scalar: KernelScalar,
    ) -> crate::Result<()> {
        let elem_size = scalar
            .dtype()
            .size_in_bytes()
            .ok_or(RocmError::UnsupportedDType {
                dtype: scalar.dtype(),
                op: "const_set",
            })?;
        let value = scalar_to_ne_bytes(scalar);
        if value.len() != elem_size {
            return Err(RocmError::BufferOutOfBounds {
                buffer_bytes: value.len(),
                offset: 0,
                requested: elem_size,
            });
        }
        let mut bytes = vec![0; dst.buffer().size_in_bytes()];
        dst.buffer().read_all(&mut bytes)?;
        for storage_index in layout.storage_indices() {
            let offset = storage_index * elem_size;
            let end = offset + elem_size;
            if end > bytes.len() {
                return Err(RocmError::BufferOutOfBounds {
                    buffer_bytes: bytes.len(),
                    offset,
                    requested: elem_size,
                });
            }
            bytes[offset..end].copy_from_slice(&value);
        }
        dst.buffer().write_all(&bytes)
    }

    #[cfg(not(hip_runtime))]
    fn scalar_to_ne_bytes(scalar: KernelScalar) -> Vec<u8> {
        match scalar {
            KernelScalar::F32(value) => value.to_ne_bytes().to_vec(),
            KernelScalar::U8(value) => vec![value],
            KernelScalar::U32(value) => value.to_ne_bytes().to_vec(),
            KernelScalar::I16(value) => value.to_ne_bytes().to_vec(),
            KernelScalar::I32(value) => value.to_ne_bytes().to_vec(),
            KernelScalar::I64(value) => value.to_ne_bytes().to_vec(),
            KernelScalar::BF16(value) => value.to_ne_bytes().to_vec(),
            KernelScalar::F16(value) => value.to_ne_bytes().to_vec(),
            KernelScalar::F64(value) => value.to_ne_bytes().to_vec(),
            KernelScalar::F8E4M3(value) => vec![value],
        }
    }

    fn copy_strided_src(op: &InplaceOp2) -> crate::Result<()> {
        let Some(CopySpec::StridedSrc {
            dst_offset,
            src_layout,
        }) = op.copy.as_ref()
        else {
            return Err(RocmError::NotImplemented(
                "copy_strided_src without copy spec",
            ));
        };
        if op.dst.dtype() != op.src.dtype() {
            return Err(RocmError::UnsupportedDType {
                dtype: op.src.dtype(),
                op: "copy_strided_src",
            });
        }
        check_layout(&op.src, src_layout)?;
        #[cfg(hip_runtime)]
        {
            let elem_size = op
                .src
                .dtype()
                .size_in_bytes()
                .ok_or(RocmError::UnsupportedDType {
                    dtype: op.src.dtype(),
                    op: "copy_strided_src",
                })?;
            if *dst_offset + src_layout.elem_count() > op.dst.elem_count() {
                return Err(RocmError::BufferOutOfBounds {
                    buffer_bytes: op.dst.buffer().size_in_bytes(),
                    offset: (*dst_offset + src_layout.elem_count()) * elem_size,
                    requested: elem_size,
                });
            }
            crate::hip::copy_strided_src(
                op.src.buffer(),
                src_layout,
                op.dst.buffer(),
                *dst_offset,
                elem_size,
            )?;
            Ok(())
        }
        #[cfg(not(hip_runtime))]
        {
            copy_elements(
                &op.src,
                src_layout.storage_indices(),
                &op.dst,
                *dst_offset..(*dst_offset + src_layout.elem_count()),
                "copy_strided_src",
            )
        }
    }

    fn copy2d(op: &InplaceOp2) -> crate::Result<()> {
        let Some(CopySpec::Copy2d {
            d1,
            d2,
            src_stride1,
            dst_stride1,
            src_offset,
            dst_offset,
        }) = op.copy.as_ref()
        else {
            return Err(RocmError::NotImplemented("copy2d without copy spec"));
        };
        if op.dst.dtype() != op.src.dtype() {
            return Err(RocmError::UnsupportedDType {
                dtype: op.src.dtype(),
                op: "copy2d",
            });
        }
        #[cfg(hip_runtime)]
        {
            let elem_size = op
                .src
                .dtype()
                .size_in_bytes()
                .ok_or(RocmError::UnsupportedDType {
                    dtype: op.src.dtype(),
                    op: "copy2d",
                })?;
            if *d1 > 0 && *d2 > 0 {
                let src_max = *src_offset + (*d1 - 1) * *src_stride1 + (*d2 - 1);
                let dst_max = *dst_offset + (*d1 - 1) * *dst_stride1 + (*d2 - 1);
                if src_max >= op.src.elem_count() {
                    return Err(RocmError::BufferOutOfBounds {
                        buffer_bytes: op.src.buffer().size_in_bytes(),
                        offset: src_max * elem_size,
                        requested: elem_size,
                    });
                }
                if dst_max >= op.dst.elem_count() {
                    return Err(RocmError::BufferOutOfBounds {
                        buffer_bytes: op.dst.buffer().size_in_bytes(),
                        offset: dst_max * elem_size,
                        requested: elem_size,
                    });
                }
            }
            crate::hip::copy2d(
                op.src.buffer(),
                op.dst.buffer(),
                *d1,
                *d2,
                *src_stride1,
                *dst_stride1,
                *src_offset,
                *dst_offset,
                elem_size,
            )?;
            Ok(())
        }
        #[cfg(not(hip_runtime))]
        {
            let src_indices =
                (0..*d1).flat_map(|i| (0..*d2).map(move |j| *src_offset + i * *src_stride1 + j));
            let dst_indices =
                (0..*d1).flat_map(|i| (0..*d2).map(move |j| *dst_offset + i * *dst_stride1 + j));
            copy_elements(&op.src, src_indices, &op.dst, dst_indices, "copy2d")
        }
    }

    #[cfg(not(hip_runtime))]
    fn copy_elements<I, J>(
        src: &TensorArg,
        src_indices: I,
        dst: &TensorArg,
        dst_indices: J,
        op: &'static str,
    ) -> crate::Result<()>
    where
        I: IntoIterator<Item = usize>,
        J: IntoIterator<Item = usize>,
    {
        let elem_size = src
            .dtype()
            .size_in_bytes()
            .ok_or(RocmError::UnsupportedDType {
                dtype: src.dtype(),
                op,
            })?;
        let mut src_bytes = vec![0; src.buffer().size_in_bytes()];
        let mut dst_bytes = vec![0; dst.buffer().size_in_bytes()];
        src.buffer().read_all(&mut src_bytes)?;
        dst.buffer().read_all(&mut dst_bytes)?;
        for (src_index, dst_index) in src_indices.into_iter().zip(dst_indices) {
            let src_start = src_index * elem_size;
            let dst_start = dst_index * elem_size;
            let src_end = src_start + elem_size;
            let dst_end = dst_start + elem_size;
            if src_index >= src.elem_count() || dst_index >= dst.elem_count() {
                return Err(RocmError::BufferOutOfBounds {
                    buffer_bytes: dst.buffer().size_in_bytes(),
                    offset: dst_start,
                    requested: elem_size,
                });
            }
            dst_bytes[dst_start..dst_end].copy_from_slice(&src_bytes[src_start..src_end]);
        }
        dst.buffer().write_all(&dst_bytes)
    }

    fn check_layout(arg: &TensorArg, layout: &LayoutArg) -> crate::Result<()> {
        if let Some(max_index) = layout.max_storage_index() {
            if max_index >= arg.elem_count() {
                return Err(RocmError::BufferOutOfBounds {
                    buffer_bytes: arg.buffer().size_in_bytes(),
                    offset: max_index.saturating_mul(arg.dtype().storage_size_in_bytes(1)),
                    requested: arg.dtype().storage_size_in_bytes(1),
                });
            }
        }
        Ok(())
    }

    #[cfg(hip_runtime)]
    fn try_scalar<F>(op: &Op1, launch: F) -> crate::Result<Option<Buffer>>
    where
        F: FnOnce(KernelDType, &Buffer, &LayoutArg, &Buffer) -> crate::Result<bool>,
    {
        let Some(layout) = op.input_layout.as_ref() else {
            return Ok(None);
        };
        let Some(output) = op.output else {
            return Ok(None);
        };
        if op.input.dtype() != output.dtype()
            || !matches!(
                op.input.dtype(),
                KernelDType::F32 | KernelDType::BF16 | KernelDType::F8E4M3
            )
            || output.elem_count() != layout.elem_count()
        {
            return Ok(None);
        }
        check_layout(&op.input, layout)?;
        let dst = op.input.buffer().allocate_on_same_allocator(
            output.dtype().storage_size_in_bytes(output.elem_count()),
            false,
        )?;
        if launch(op.input.dtype(), op.input.buffer(), layout, &dst)? {
            Ok(Some(dst))
        } else {
            Ok(None)
        }
    }

    #[cfg(hip_runtime)]
    fn is_contiguous(layout: &LayoutArg) -> bool {
        let mut expected = 1usize;
        for (&dim, &stride) in layout.dims().iter().zip(layout.stride()).rev() {
            if stride != expected {
                return false;
            }
            expected = expected.saturating_mul(dim);
        }
        true
    }

    #[cfg(hip_runtime)]
    fn batch_strides(
        lhs_layout: &LayoutArg,
        rhs_layout: &LayoutArg,
        m: usize,
        n: usize,
        k: usize,
    ) -> Option<(usize, usize)> {
        let lhs_stride = lhs_layout.stride();
        let rhs_stride = rhs_layout.stride();
        let rank = lhs_stride.len();
        if rhs_stride.len() != rank || rank < 2 {
            return None;
        }
        let lhs_batch_stride = match &lhs_stride[..rank - 2] {
            [s1, stride] if *s1 == *stride * lhs_layout.dims()[1] => *stride,
            [_, stride] if lhs_layout.dims()[0] == 1 => *stride,
            [stride, _] if lhs_layout.dims()[1] == 1 => *stride,
            [stride] => *stride,
            [] => m * k,
            _ => return None,
        };
        let rhs_batch_stride = match &rhs_stride[..rank - 2] {
            [s1, stride] if *s1 == *stride * rhs_layout.dims()[1] => *stride,
            [_, stride] if rhs_layout.dims()[0] == 1 => *stride,
            [stride, _] if rhs_layout.dims()[1] == 1 => *stride,
            [stride] => *stride,
            [] => n * k,
            _ => return None,
        };
        Some((lhs_batch_stride, rhs_batch_stride))
    }

    #[cfg(hip_runtime)]
    fn unary_opcode(name: &str) -> Option<i32> {
        match name {
            "abs" => Some(1),
            "ceil" => Some(2),
            "cos" => Some(3),
            "exp" => Some(4),
            "floor" => Some(5),
            "log" => Some(6),
            "neg" => Some(7),
            "recip" => Some(8),
            "relu" => Some(9),
            "round" => Some(10),
            "sin" => Some(11),
            "sqr" => Some(12),
            "sqrt" => Some(13),
            "tanh" => Some(14),
            "silu" => Some(15),
            "gelu" => Some(16),
            "erf" => Some(17),
            "gelu_erf" => Some(18),
            "sign" => Some(19),
            "sigmoid" => Some(20),
            _ => None,
        }
    }

    #[cfg(hip_runtime)]
    fn reduce_opcode(name: &str) -> Option<i32> {
        match name {
            "sum" => Some(1),
            "min" => Some(2),
            "max" => Some(3),
            "argmin" => Some(4),
            "argmax" => Some(5),
            _ => None,
        }
    }

    #[cfg(hip_runtime)]
    fn binary_opcode(name: &str) -> Option<i32> {
        match name {
            "add" => Some(1),
            "div" => Some(2),
            "maximum" => Some(3),
            "minimum" => Some(4),
            "mul" => Some(5),
            "sub" => Some(6),
            _ => None,
        }
    }

    #[cfg(hip_runtime)]
    fn cmp_opcode(name: &str) -> Option<i32> {
        match name {
            "eq" => Some(1),
            "ge" => Some(2),
            "gt" => Some(3),
            "le" => Some(4),
            "lt" => Some(5),
            "ne" => Some(6),
            _ => None,
        }
    }

    fn read_f32(arg: &TensorArg) -> crate::Result<Vec<f32>> {
        if arg.dtype() != KernelDType::F32 {
            return Err(RocmError::UnsupportedDType {
                dtype: arg.dtype(),
                op: "read_f32",
            });
        }
        let mut bytes = vec![0; arg.buffer().size_in_bytes()];
        arg.buffer().read_all(&mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    fn write_f32_output(
        src: &Buffer,
        output: TensorOutput,
        values: &[f32],
    ) -> crate::Result<Buffer> {
        if values.len() != output.elem_count() {
            return Err(RocmError::BufferOutOfBounds {
                buffer_bytes: values.len() * 4,
                offset: 0,
                requested: output.elem_count() * 4,
            });
        }
        let dst = src.allocate_on_same_allocator(
            output.dtype().storage_size_in_bytes(values.len()),
            false,
        )?;
        dst.write_all(&f32s_to_bytes(values))?;
        Ok(dst)
    }

    fn f32s_to_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::Device;

        fn f32_arg(device: &Device, values: &[f32]) -> (Buffer, TensorArg) {
            let buffer = device.copy_from_host(&f32s_to_bytes(values)).unwrap();
            let arg = TensorArg::new(&buffer, KernelDType::F32, values.len()).unwrap();
            (buffer, arg)
        }

        fn buffer_to_f32(buffer: &Buffer) -> Vec<f32> {
            let arg = TensorArg::new(buffer, KernelDType::F32, buffer.size_in_bytes() / 4).unwrap();
            read_f32(&arg).unwrap()
        }

        fn i64_arg(device: &Device, values: &[i64]) -> (Buffer, TensorArg) {
            let bytes = values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<_>>();
            let buffer = device.copy_from_host(&bytes).unwrap();
            let arg = TensorArg::new(&buffer, KernelDType::I64, values.len()).unwrap();
            (buffer, arg)
        }

        fn buffer_to_i64(buffer: &Buffer) -> Vec<i64> {
            let mut bytes = vec![0; buffer.size_in_bytes()];
            buffer.read_all(&mut bytes).unwrap();
            bytes
                .chunks_exact(8)
                .map(|chunk| i64::from_ne_bytes(chunk.try_into().unwrap()))
                .collect()
        }

        #[test]
        fn unary_uses_layout_without_cpu_storage() {
            let device = Device::new(0).unwrap();
            let (_buffer, input) = f32_arg(&device, &[1., 2., 3., 4.]);
            let op = Op1 {
                name: "sqr",
                input,
                input_layout: Some(LayoutArg::new(vec![2, 2], vec![1, 2], 0).unwrap()),
                output: Some(TensorOutput::new(KernelDType::F32, 4)),
            };

            let output = try_unary(&op).unwrap().unwrap();
            assert_eq!(buffer_to_f32(&output), vec![1., 9., 4., 16.]);
        }

        #[test]
        fn binary_supports_stride_zero_broadcast_layouts() {
            let device = Device::new(0).unwrap();
            let (_lhs_buffer, lhs) = f32_arg(&device, &[1., 2., 3., 4.]);
            let (_rhs_buffer, rhs) = f32_arg(&device, &[10., 20.]);
            let op = Op2 {
                name: "add",
                lhs,
                rhs,
                lhs_layout: Some(LayoutArg::new(vec![2, 2], vec![2, 1], 0).unwrap()),
                rhs_layout: Some(LayoutArg::new(vec![2, 2], vec![0, 1], 0).unwrap()),
                output: Some(TensorOutput::new(KernelDType::F32, 4)),
            };

            let output = try_binary(&op).unwrap().unwrap();
            assert_eq!(buffer_to_f32(&output), vec![11., 22., 13., 24.]);
        }

        #[test]
        fn const_set_updates_strided_destination() {
            let device = Device::new(0).unwrap();
            let (buffer, dst) = f32_arg(&device, &[0., 0., 0., 0.]);
            let op = InplaceOp1 {
                name: "const_set",
                dst,
                dst_layout: Some(LayoutArg::new(vec![2], vec![2], 1).unwrap()),
                scalar: Some(KernelScalar::F32(5.)),
            };

            call_const_set::<RocmError, _>(op, || unreachable!()).unwrap();
            assert_eq!(buffer_to_f32(&buffer), vec![0., 5., 0., 5.]);
        }

        #[test]
        fn const_set_updates_i64_destination() {
            let device = Device::new(0).unwrap();
            let (buffer, dst) = i64_arg(&device, &[0, 0, 0, 0]);
            let op = InplaceOp1 {
                name: "const_set",
                dst,
                dst_layout: Some(LayoutArg::new(vec![2], vec![2], 1).unwrap()),
                scalar: Some(KernelScalar::I64(-7)),
            };

            call_const_set::<RocmError, _>(op, || unreachable!()).unwrap();
            assert_eq!(buffer_to_i64(&buffer), vec![0, -7, 0, -7]);
        }
    }
}
