//! ROCm kernel dispatch surface for Candle.
//!
//! `candle-core` calls into this crate with typed ROCm operation descriptors
//! first. When compiled with HIP support, the supported f32 operation slice
//! launches HIP kernels behind these entry points; unsupported operations still
//! use fallback closures while coverage is expanded.

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
}

impl KernelScalar {
    pub fn dtype(self) -> KernelDType {
        match self {
            Self::F32(_) => KernelDType::F32,
            Self::U8(_) => KernelDType::U8,
            Self::U32(_) => KernelDType::U32,
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
        let elem_count = self.elem_count();
        (elem_count > 0).then(|| {
            (0..elem_count)
                .map(|index| self.storage_index(index))
                .max()
                .unwrap_or(self.start_offset)
        })
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
    use crate::{cpu_fallback, Device, TensorArg};

    #[derive(Clone, Debug)]
    pub struct QuantizedOp<'a> {
        pub name: &'static str,
        pub device: Option<&'a Device>,
        pub input: Option<TensorArg>,
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
        if op.input.dtype() != KernelDType::F32 || output.dtype() != KernelDType::F32 {
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
        crate::hip::unary_f32(code, op.input.buffer(), layout, &dst)?;
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
        if op.lhs.dtype() != KernelDType::F32
            || op.rhs.dtype() != KernelDType::F32
            || output.dtype() != KernelDType::F32
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
        crate::hip::binary_f32(
            code,
            op.lhs.buffer(),
            lhs_layout,
            op.rhs.buffer(),
            rhs_layout,
            &dst,
        )?;
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
        if op.lhs.dtype() != KernelDType::F32
            || op.rhs.dtype() != KernelDType::F32
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
        crate::hip::cmp_f32(
            code,
            op.lhs.buffer(),
            lhs_layout,
            op.rhs.buffer(),
            rhs_layout,
            &dst,
        )?;
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
            }
            Ok(())
        }
        #[cfg(not(hip_runtime))]
        {
            match scalar {
                KernelScalar::F32(value) => {
                    let mut values = read_f32(&op.dst)?;
                    for storage_index in layout.storage_indices() {
                        values[storage_index] = value;
                    }
                    op.dst.buffer().write_all(&f32s_to_bytes(&values))
                }
                KernelScalar::U8(value) => {
                    let mut values = vec![0; op.dst.buffer().size_in_bytes()];
                    op.dst.buffer().read_all(&mut values)?;
                    for storage_index in layout.storage_indices() {
                        values[storage_index] = value;
                    }
                    op.dst.buffer().write_all(&values)
                }
                KernelScalar::U32(value) => {
                    let mut values = read_u32(&op.dst)?;
                    for storage_index in layout.storage_indices() {
                        values[storage_index] = value;
                    }
                    op.dst.buffer().write_all(&u32s_to_bytes(&values))
                }
            }
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
                    offset: max_index * arg.dtype().storage_size_in_bytes(1),
                    requested: arg.dtype().storage_size_in_bytes(1),
                });
            }
        }
        Ok(())
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

    #[cfg(not(hip_runtime))]
    fn read_u32(arg: &TensorArg) -> crate::Result<Vec<u32>> {
        if arg.dtype() != KernelDType::U32 {
            return Err(RocmError::UnsupportedDType {
                dtype: arg.dtype(),
                op: "read_u32",
            });
        }
        let mut bytes = vec![0; arg.buffer().size_in_bytes()];
        arg.buffer().read_all(&mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_ne_bytes(chunk.try_into().unwrap()))
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

    #[cfg(not(hip_runtime))]
    fn u32s_to_bytes(values: &[u32]) -> Vec<u8> {
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
    }
}
