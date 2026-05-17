//! ROCm kernel dispatch surface for Candle.
//!
//! The current implementation is a CPU fallback shim. `candle-core` calls into
//! this crate with typed ROCm operation descriptors first, and these wrappers
//! execute the provided fallback closure until real ROCm kernels are
//! implemented behind the same entry points.

mod allocator;
mod buffer;
mod dtype;
mod error;
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
    use crate::{cpu_fallback, Device, KernelDType, TensorOutput};

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

    pub fn call_zeros<T, E, F>(op: AllocOp<'_>, fallback: F) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        cpu_fallback(op.name, fallback)
    }

    pub fn call_alloc_uninit<T, E, F>(op: AllocOp<'_>, fallback: F) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        cpu_fallback(op.name, fallback)
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
    use crate::{cpu_fallback, Device, TensorArg, TensorOutput};

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
        pub first: TensorArg,
        pub second: TensorArg,
        pub third: TensorArg,
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
        pub src: TensorArg,
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

    call_op!(call_copy_strided_src, InplaceOp2);
    call_op!(call_copy2d, InplaceOp2);
    call_op!(call_const_set, InplaceOp1);
    call_op!(call_transfer, TransferOp<'_>);
}
