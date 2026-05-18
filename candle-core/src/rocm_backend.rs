//! ROCm backend.
//!
//! This backend stores tensor data in ROCm buffers and dispatches supported
//! operations through `candle-rocm-kernels`. Unsupported operations still use
//! the CPU fallback bridge while native kernel coverage is expanded.

use candle_rocm_kernels as kernels;
use float8::F8E4M3;
use half::{bf16, f16};
use std::hash::{Hash, Hasher};
use std::mem;
use std::sync::Arc;

use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};

#[derive(Debug, Clone)]
pub struct RocmDevice {
    inner: Arc<kernels::Device>,
}

impl RocmDevice {
    fn ordinal(&self) -> usize {
        self.inner.ordinal()
    }

    pub(crate) fn kernel_device(&self) -> &kernels::Device {
        &self.inner
    }
}

impl PartialEq for RocmDevice {
    fn eq(&self, rhs: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &rhs.inner)
    }
}

impl Eq for RocmDevice {}

impl Hash for RocmDevice {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.inner).hash(state)
    }
}

#[derive(Debug, Clone)]
pub struct RocmStorage {
    buffer: kernels::Buffer,
    device: RocmDevice,
    dtype: DType,
    elem_count: usize,
}

impl RocmStorage {
    fn wrap(storage: CpuStorage, device: RocmDevice, op: &'static str) -> Result<Self> {
        ensure_supported_storage_dtype(storage.dtype(), op)?;
        Self::from_cpu_storage_owned(storage, device)
    }

    fn from_cpu_storage_owned(storage: CpuStorage, device: RocmDevice) -> Result<Self> {
        let (dtype, elem_count, bytes) = cpu_storage_into_bytes(storage)?;
        let buffer = device.inner.copy_from_host(&bytes)?;
        Self::from_buffer(buffer, device, dtype, elem_count, "from_cpu_storage_owned")
    }

    #[cfg(feature = "rocm")]
    pub(crate) fn from_raw_bytes(
        device: &RocmDevice,
        dtype: DType,
        elem_count: usize,
        bytes: &[u8],
        op: &'static str,
    ) -> Result<Self> {
        ensure_supported_storage_dtype(dtype, op)?;
        let expected = storage_size_in_bytes(dtype, elem_count)?;
        if bytes.len() != expected {
            crate::bail!(
                "invalid ROCm raw buffer size for {op}: expected {expected} bytes, got {}",
                bytes.len()
            )
        }
        let buffer = device.inner.copy_from_host(bytes)?;
        Self::from_buffer(buffer, device.clone(), dtype, elem_count, op)
    }

    pub(crate) fn from_buffer(
        buffer: kernels::Buffer,
        device: RocmDevice,
        dtype: DType,
        elem_count: usize,
        op: &'static str,
    ) -> Result<Self> {
        ensure_supported_storage_dtype(dtype, op)?;
        let expected = storage_size_in_bytes(dtype, elem_count)?;
        if buffer.size_in_bytes() != expected {
            crate::bail!(
                "invalid ROCm buffer size for {op}: expected {expected} bytes, got {}",
                buffer.size_in_bytes()
            )
        }
        Ok(Self {
            buffer,
            device,
            dtype,
            elem_count,
        })
    }

    fn size_in_bytes(&self) -> usize {
        storage_size_in_bytes(self.dtype, self.elem_count)
            .unwrap_or_else(|_| self.buffer.size_in_bytes())
    }

    fn to_host_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = vec![0; self.size_in_bytes()];
        self.device.inner.copy_to_host(&self.buffer, &mut bytes)?;
        Ok(bytes)
    }

    fn to_cpu_storage_impl(&self) -> Result<CpuStorage> {
        cpu_storage_from_bytes(self.dtype, self.elem_count, self.to_host_bytes()?)
    }

    fn set_cpu_storage_owned(&mut self, storage: CpuStorage, op: &'static str) -> Result<()> {
        let storage = Self::wrap(storage, self.device.clone(), op)?;
        *self = storage;
        Ok(())
    }

    pub(crate) fn tensor_arg(&self) -> Result<kernels::TensorArg> {
        kernels::TensorArg::new(&self.buffer, kernel_dtype(self.dtype)?, self.elem_count)
            .map_err(Error::from)
    }

    fn op1(
        &self,
        name: &'static str,
        output: Option<kernels::TensorOutput>,
    ) -> Result<kernels::tensor::Op1> {
        Ok(kernels::tensor::Op1 {
            name,
            input: self.tensor_arg()?,
            input_layout: None,
            output,
        })
    }

    fn op2(
        &self,
        name: &'static str,
        rhs: &Self,
        output: Option<kernels::TensorOutput>,
    ) -> Result<kernels::tensor::Op2> {
        Ok(kernels::tensor::Op2 {
            name,
            lhs: self.tensor_arg()?,
            rhs: rhs.tensor_arg()?,
            lhs_layout: None,
            rhs_layout: None,
            output,
        })
    }

    fn op3(
        &self,
        name: &'static str,
        second: &Self,
        third: &Self,
        output: Option<kernels::TensorOutput>,
    ) -> Result<kernels::tensor::Op3> {
        Ok(kernels::tensor::Op3 {
            name,
            first: self.tensor_arg()?,
            second: second.tensor_arg()?,
            third: third.tensor_arg()?,
            output,
        })
    }

    pub fn transfer_to_device(&self, dst: &RocmDevice) -> Result<Self> {
        let op = kernels::tensor::TransferOp {
            name: "transfer",
            src: self.tensor_arg()?,
            dst_device: dst.kernel_device(),
            output: kernel_output(self.dtype, self.elem_count)?,
        };
        let buffer = kernels::tensor::call_transfer(op, || {
            let bytes = self.to_host_bytes()?;
            dst.inner.copy_from_host(&bytes).map_err(Error::from)
        })?;
        Self::from_buffer(buffer, dst.clone(), self.dtype, self.elem_count, "transfer")
    }

    pub fn softmax_last_dim(&self, layout: &Layout) -> Result<(Self, Shape)> {
        if !matches!(self.dtype, DType::F32 | DType::BF16 | DType::F8E4M3) {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "softmax-last-dim").bt());
        }
        let mut op = self.op1(
            "softmax-last-dim",
            Some(kernel_output_for_layout(layout, self.dtype)?),
        )?;
        op.input_layout = Some(kernel_layout(layout)?);
        if let Some(buffer) = kernels::tensor::try_softmax_last_dim(&op).map_err(Error::from)? {
            let storage = Self::from_buffer(
                buffer,
                self.device.clone(),
                self.dtype,
                layout.shape().elem_count(),
                "softmax-last-dim",
            )?;
            return Ok((storage, layout.shape().clone()));
        }
        Err(Error::UnsupportedDTypeForOp(self.dtype, "softmax-last-dim").bt())
    }

    pub fn sigmoid(&self, layout: &Layout) -> Result<(Self, Shape)> {
        if !matches!(self.dtype, DType::F32 | DType::BF16 | DType::F8E4M3) {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "sigmoid").bt());
        }
        let mut op = self.op1(
            "sigmoid",
            Some(kernel_output_for_layout(layout, self.dtype)?),
        )?;
        op.input_layout = Some(kernel_layout(layout)?);
        if let Some(buffer) = kernels::tensor::try_unary(&op).map_err(Error::from)? {
            let storage = Self::from_buffer(
                buffer,
                self.device.clone(),
                self.dtype,
                layout.shape().elem_count(),
                "sigmoid",
            )?;
            return Ok((storage, layout.shape().clone()));
        }
        Err(Error::UnsupportedDTypeForOp(self.dtype, "sigmoid").bt())
    }

    pub fn repeat_penalty(
        &self,
        layout: &Layout,
        token_ids: &Self,
        token_ids_layout: &Layout,
        penalty: f32,
    ) -> Result<(Self, Shape)> {
        let mut op = self.op2(
            "repeat-penalty",
            token_ids,
            Some(kernel_output_for_layout(layout, DType::F32)?),
        )?;
        op.lhs_layout = Some(kernel_layout(layout)?);
        op.rhs_layout = Some(kernel_layout(token_ids_layout)?);
        if let Some(buffer) =
            kernels::tensor::try_repeat_penalty(&op, penalty).map_err(Error::from)?
        {
            let storage = Self::from_buffer(
                buffer,
                self.device.clone(),
                DType::F32,
                layout.shape().elem_count(),
                "repeat-penalty",
            )?;
            return Ok((storage, layout.shape().clone()));
        }
        let logits = self.to_cpu_storage_impl()?;
        let token_ids = token_ids.to_cpu_storage_impl()?;
        let storage =
            repeat_penalty_cpu_storage(&logits, layout, &token_ids, token_ids_layout, penalty)?;
        let storage = Self::wrap(storage, self.device.clone(), "repeat-penalty")?;
        Ok((storage, layout.shape().clone()))
    }

    pub fn rms_norm(
        &self,
        layout: &Layout,
        alpha: &Self,
        alpha_layout: &Layout,
        eps: f32,
    ) -> Result<(Self, Shape)> {
        if self.dtype != alpha.dtype {
            crate::bail!(
                "rms-norm input and alpha dtypes differ: {:?} != {:?}",
                self.dtype,
                alpha.dtype
            )
        }
        let mut op = self.op2(
            "rms-norm",
            alpha,
            Some(kernel_output_for_layout(layout, self.dtype)?),
        )?;
        op.lhs_layout = Some(kernel_layout(layout)?);
        op.rhs_layout = Some(kernel_layout(alpha_layout)?);
        if let Some(buffer) = kernels::tensor::try_rms_norm(&op, eps).map_err(Error::from)? {
            let storage = Self::from_buffer(
                buffer,
                self.device.clone(),
                self.dtype,
                layout.shape().elem_count(),
                "rms-norm",
            )?;
            return Ok((storage, layout.shape().clone()));
        }
        Err(Error::UnsupportedDTypeForOp(self.dtype, "rms-norm").bt())
    }

    pub fn layer_norm(
        &self,
        layout: &Layout,
        alpha: &Self,
        alpha_layout: &Layout,
        beta: &Self,
        beta_layout: &Layout,
        eps: f32,
    ) -> Result<(Self, Shape)> {
        if self.dtype != alpha.dtype || self.dtype != beta.dtype {
            crate::bail!(
                "layer-norm input, alpha, and beta dtypes differ: {:?}, {:?}, {:?}",
                self.dtype,
                alpha.dtype,
                beta.dtype
            )
        }
        if !matches!(self.dtype, DType::F32 | DType::BF16 | DType::F8E4M3) {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "layer-norm").bt());
        }
        let op = self.op3(
            "layer-norm",
            alpha,
            beta,
            Some(kernel_output_for_layout(layout, self.dtype)?),
        )?;
        let src_l = kernel_layout(layout)?;
        let alpha_l = kernel_layout(alpha_layout)?;
        let beta_l = kernel_layout(beta_layout)?;
        if let Some(buffer) = kernels::tensor::try_layer_norm(&op, &src_l, &alpha_l, &beta_l, eps)
            .map_err(Error::from)?
        {
            let storage = Self::from_buffer(
                buffer,
                self.device.clone(),
                self.dtype,
                layout.shape().elem_count(),
                "layer-norm",
            )?;
            return Ok((storage, layout.shape().clone()));
        }
        Err(Error::UnsupportedDTypeForOp(self.dtype, "layer-norm").bt())
    }

    pub fn rope(
        &self,
        layout: &Layout,
        cos: &Self,
        cos_layout: &Layout,
        sin: &Self,
        sin_layout: &Layout,
        interleaved: bool,
    ) -> Result<(Self, Shape)> {
        if self.dtype != cos.dtype || self.dtype != sin.dtype {
            crate::bail!(
                "rotary-emb input, cos, and sin dtypes differ: {:?}, {:?}, {:?}",
                self.dtype,
                cos.dtype,
                sin.dtype
            )
        }
        if !matches!(self.dtype, DType::F32 | DType::BF16 | DType::F8E4M3) {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "rotary-emb").bt());
        }
        let op = self.op3(
            if interleaved {
                "rotary-emb-int"
            } else {
                "rotary-emb"
            },
            cos,
            sin,
            Some(kernel_output_for_layout(layout, self.dtype)?),
        )?;
        let src_l = kernel_layout(layout)?;
        let cos_l = kernel_layout(cos_layout)?;
        let sin_l = kernel_layout(sin_layout)?;
        if let Some(buffer) =
            kernels::tensor::try_rope(&op, &src_l, &cos_l, &sin_l, interleaved, false)
                .map_err(Error::from)?
        {
            let storage = Self::from_buffer(
                buffer,
                self.device.clone(),
                self.dtype,
                layout.shape().elem_count(),
                "rotary-emb",
            )?;
            return Ok((storage, layout.shape().clone()));
        }
        Err(Error::UnsupportedDTypeForOp(self.dtype, "rotary-emb").bt())
    }

    pub fn rope_thd(
        &self,
        layout: &Layout,
        cos: &Self,
        cos_layout: &Layout,
        sin: &Self,
        sin_layout: &Layout,
    ) -> Result<(Self, Shape)> {
        if self.dtype != cos.dtype || self.dtype != sin.dtype {
            crate::bail!(
                "rotary-emb input, cos, and sin dtypes differ: {:?}, {:?}, {:?}",
                self.dtype,
                cos.dtype,
                sin.dtype
            )
        }
        if !matches!(self.dtype, DType::F32 | DType::BF16 | DType::F8E4M3) {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "rotary-emb").bt());
        }
        let op = self.op3(
            "rotary-emb",
            cos,
            sin,
            Some(kernel_output_for_layout(layout, self.dtype)?),
        )?;
        let src_l = kernel_layout(layout)?;
        let cos_l = kernel_layout(cos_layout)?;
        let sin_l = kernel_layout(sin_layout)?;
        if let Some(buffer) = kernels::tensor::try_rope(&op, &src_l, &cos_l, &sin_l, false, true)
            .map_err(Error::from)?
        {
            let storage = Self::from_buffer(
                buffer,
                self.device.clone(),
                self.dtype,
                layout.shape().elem_count(),
                "rotary-emb",
            )?;
            return Ok((storage, layout.shape().clone()));
        }
        Err(Error::UnsupportedDTypeForOp(self.dtype, "rotary-emb").bt())
    }

    pub(crate) fn try_arg_sort(
        &self,
        layout: &Layout,
        asc: bool,
        last_dim: usize,
    ) -> Result<Option<(Self, Shape)>> {
        if !matches!(self.dtype, DType::F32 | DType::BF16 | DType::F8E4M3) {
            return Ok(None);
        }
        let mut op = self.op1(
            "argsort",
            Some(kernel_output_for_layout(layout, DType::U32)?),
        )?;
        op.input_layout = Some(kernel_layout(layout)?);
        let Some(buffer) =
            kernels::tensor::try_arg_sort(&op, asc, last_dim).map_err(Error::from)?
        else {
            return Ok(None);
        };
        let storage = Self::from_buffer(
            buffer,
            self.device.clone(),
            DType::U32,
            layout.shape().elem_count(),
            "argsort",
        )?;
        Ok(Some((storage, layout.shape().clone())))
    }

    pub(crate) fn custom_op1<F>(
        &self,
        l: &Layout,
        name: &'static str,
        cpu_fwd: F,
    ) -> Result<(Self, Shape)>
    where
        F: FnOnce(&CpuStorage, &Layout) -> Result<(CpuStorage, Shape)>,
    {
        let (storage, shape) = if name == "argsort" {
            let op = kernels::custom::Op1 {
                name,
                input: self.tensor_arg()?,
                output: Some(kernel_output_for_layout(l, DType::U32)?),
            };
            kernels::custom::call_arg_sort(op, || {
                let storage = self.to_cpu_storage_impl()?;
                cpu_fwd(&storage, l)
            })?
        } else {
            let op = kernels::custom::Op1 {
                name,
                input: self.tensor_arg()?,
                output: None,
            };
            kernels::custom::call_apply_op1(op, || {
                let storage = self.to_cpu_storage_impl()?;
                cpu_fwd(&storage, l)
            })?
        };
        let storage = Self::wrap(storage, self.device.clone(), name)?;
        Ok((storage, shape))
    }

    pub(crate) fn custom_op2<F>(
        &self,
        l1: &Layout,
        rhs: &Self,
        l2: &Layout,
        name: &'static str,
        cpu_fwd: F,
    ) -> Result<(Self, Shape)>
    where
        F: FnOnce(&CpuStorage, &Layout, &CpuStorage, &Layout) -> Result<(CpuStorage, Shape)>,
    {
        let op = kernels::custom::Op2 {
            name,
            lhs: self.tensor_arg()?,
            rhs: rhs.tensor_arg()?,
            output: None,
        };
        let (storage, shape) = kernels::custom::call_apply_op2(op, || {
            let lhs = self.to_cpu_storage_impl()?;
            let rhs = rhs.to_cpu_storage_impl()?;
            cpu_fwd(&lhs, l1, &rhs, l2)
        })?;
        let storage = Self::wrap(storage, self.device.clone(), name)?;
        Ok((storage, shape))
    }

    pub(crate) fn custom_op3<F>(
        &self,
        l1: &Layout,
        t2: (&Self, &Layout),
        t3: (&Self, &Layout),
        name: &'static str,
        cpu_fwd: F,
    ) -> Result<(Self, Shape)>
    where
        F: FnOnce(
            &CpuStorage,
            &Layout,
            &CpuStorage,
            &Layout,
            &CpuStorage,
            &Layout,
        ) -> Result<(CpuStorage, Shape)>,
    {
        let (t2, l2) = t2;
        let (t3, l3) = t3;
        let op = kernels::custom::Op3 {
            name,
            lhs: self.tensor_arg()?,
            rhs2: t2.tensor_arg()?,
            rhs3: t3.tensor_arg()?,
            output: None,
        };
        let (storage, shape) = kernels::custom::call_apply_op3(op, || {
            let t1 = self.to_cpu_storage_impl()?;
            let t2 = t2.to_cpu_storage_impl()?;
            let t3 = t3.to_cpu_storage_impl()?;
            cpu_fwd(&t1, l1, &t2, l2, &t3, l3)
        })?;
        let storage = Self::wrap(storage, self.device.clone(), name)?;
        Ok((storage, shape))
    }

    pub(crate) fn inplace_custom_op1<F>(
        &mut self,
        l: &Layout,
        name: &'static str,
        cpu_fwd: F,
    ) -> Result<()>
    where
        F: FnOnce(&mut CpuStorage, &Layout) -> Result<()>,
    {
        let op = kernels::custom::InplaceOp1 {
            name,
            dst: self.tensor_arg()?,
        };
        let storage = kernels::custom::call_inplace_op1(op, || {
            let mut storage = self.to_cpu_storage_impl()?;
            cpu_fwd(&mut storage, l)?;
            Ok::<CpuStorage, Error>(storage)
        })?;
        self.set_cpu_storage_owned(storage, name)
    }

    pub(crate) fn inplace_custom_op2<F>(
        &mut self,
        l1: &Layout,
        rhs: &Self,
        l2: &Layout,
        name: &'static str,
        cpu_fwd: F,
    ) -> Result<()>
    where
        F: FnOnce(&mut CpuStorage, &Layout, &CpuStorage, &Layout) -> Result<()>,
    {
        let op = kernels::custom::InplaceOp2 {
            name,
            dst: self.tensor_arg()?,
            rhs: rhs.tensor_arg()?,
        };
        let storage = kernels::custom::call_inplace_op2(op, || {
            let mut lhs = self.to_cpu_storage_impl()?;
            let rhs = rhs.to_cpu_storage_impl()?;
            cpu_fwd(&mut lhs, l1, &rhs, l2)?;
            Ok::<CpuStorage, Error>(lhs)
        })?;
        self.set_cpu_storage_owned(storage, name)
    }

    pub(crate) fn inplace_custom_op3<F>(
        &mut self,
        l1: &Layout,
        t2: (&Self, &Layout),
        t3: (&Self, &Layout),
        name: &'static str,
        cpu_fwd: F,
    ) -> Result<()>
    where
        F: FnOnce(
            &mut CpuStorage,
            &Layout,
            &CpuStorage,
            &Layout,
            &CpuStorage,
            &Layout,
        ) -> Result<()>,
    {
        let (t2, l2) = t2;
        let (t3, l3) = t3;
        let op = kernels::custom::InplaceOp3 {
            name,
            dst: self.tensor_arg()?,
            rhs2: t2.tensor_arg()?,
            rhs3: t3.tensor_arg()?,
        };
        let storage = kernels::custom::call_inplace_op3(op, || {
            let mut t1 = self.to_cpu_storage_impl()?;
            let t2 = t2.to_cpu_storage_impl()?;
            let t3 = t3.to_cpu_storage_impl()?;
            cpu_fwd(&mut t1, l1, &t2, l2, &t3, l3)?;
            Ok::<CpuStorage, Error>(t1)
        })?;
        self.set_cpu_storage_owned(storage, name)
    }
}

fn ensure_supported_storage_dtype(dtype: DType, _op: &'static str) -> Result<()> {
    let _ = kernel_dtype(dtype)?;
    Ok(())
}

fn kernel_dtype(dtype: DType) -> Result<kernels::KernelDType> {
    let dtype = match dtype {
        DType::U8 => kernels::KernelDType::U8,
        DType::U32 => kernels::KernelDType::U32,
        DType::I16 => kernels::KernelDType::I16,
        DType::I32 => kernels::KernelDType::I32,
        DType::I64 => kernels::KernelDType::I64,
        DType::BF16 => kernels::KernelDType::BF16,
        DType::F16 => kernels::KernelDType::F16,
        DType::F32 => kernels::KernelDType::F32,
        DType::F64 => kernels::KernelDType::F64,
        DType::F8E4M3 => kernels::KernelDType::F8E4M3,
        DType::F6E2M3 => kernels::KernelDType::F6E2M3,
        DType::F6E3M2 => kernels::KernelDType::F6E3M2,
        DType::F4 => kernels::KernelDType::F4,
        DType::F8E8M0 => kernels::KernelDType::F8E8M0,
    };
    Ok(dtype)
}

fn kernel_output(dtype: DType, elem_count: usize) -> Result<kernels::TensorOutput> {
    Ok(kernels::TensorOutput::new(kernel_dtype(dtype)?, elem_count))
}

fn storage_size_in_bytes(dtype: DType, elem_count: usize) -> Result<usize> {
    Ok(kernel_dtype(dtype)?.storage_size_in_bytes(elem_count))
}

fn kernel_output_for_layout(layout: &Layout, dtype: DType) -> Result<kernels::TensorOutput> {
    kernel_output(dtype, layout.shape().elem_count())
}

fn kernel_layout(layout: &Layout) -> Result<kernels::LayoutArg> {
    kernels::LayoutArg::new(
        layout.shape().dims().to_vec(),
        layout.stride().to_vec(),
        layout.start_offset(),
    )
    .map_err(Error::from)
}

fn reduce_output_elem_count(layout: &Layout, dims: &[usize]) -> usize {
    let mut out_dims = layout.shape().dims().to_vec();
    for &dim in dims {
        if dim < out_dims.len() {
            out_dims[dim] = 1;
        }
    }
    out_dims.iter().product()
}

fn index_select_output_elem_count(layout: &Layout, ids_l: &Layout, dim: usize) -> usize {
    let mut out_dims = layout.shape().dims().to_vec();
    if dim < out_dims.len() {
        out_dims[dim] = ids_l.shape().elem_count();
    }
    out_dims.iter().product()
}

fn kernel_scalar(scalar: crate::scalar::Scalar) -> Result<kernels::KernelScalar> {
    match scalar {
        crate::scalar::Scalar::F32(value) => Ok(kernels::KernelScalar::F32(value)),
        crate::scalar::Scalar::U8(value) => Ok(kernels::KernelScalar::U8(value)),
        crate::scalar::Scalar::U32(value) => Ok(kernels::KernelScalar::U32(value)),
        crate::scalar::Scalar::I16(value) => Ok(kernels::KernelScalar::I16(value)),
        crate::scalar::Scalar::I32(value) => Ok(kernels::KernelScalar::I32(value)),
        crate::scalar::Scalar::I64(value) => Ok(kernels::KernelScalar::I64(value)),
        crate::scalar::Scalar::BF16(value) => Ok(kernels::KernelScalar::BF16(value.to_bits())),
        crate::scalar::Scalar::F16(value) => Ok(kernels::KernelScalar::F16(value.to_bits())),
        crate::scalar::Scalar::F64(value) => Ok(kernels::KernelScalar::F64(value)),
        crate::scalar::Scalar::F8E4M3(value) => Ok(kernels::KernelScalar::F8E4M3(value.to_bits())),
    }
}

fn native_float_output(dtype: DType) -> DType {
    match dtype {
        DType::BF16 | DType::F8E4M3 => dtype,
        _ => DType::F32,
    }
}

fn same_native_float_output(lhs: DType, rhs: DType) -> DType {
    if matches!(lhs, DType::BF16 | DType::F8E4M3) && lhs == rhs {
        lhs
    } else {
        DType::F32
    }
}

fn cmp_op_name(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Eq => "eq",
        CmpOp::Ne => "ne",
        CmpOp::Le => "le",
        CmpOp::Ge => "ge",
        CmpOp::Lt => "lt",
        CmpOp::Gt => "gt",
    }
}

fn slice_to_bytes<T>(data: &[T]) -> Vec<u8> {
    let len = mem::size_of_val(data);
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, len) }.to_vec()
}

fn vec_from_bytes<T>(bytes: &[u8], elem_count: usize, dtype: DType) -> Result<Vec<T>> {
    let expected = elem_count * mem::size_of::<T>();
    if bytes.len() != expected {
        crate::bail!(
            "invalid ROCm buffer size for {dtype:?}: expected {expected} bytes, got {}",
            bytes.len()
        )
    }
    let mut data = Vec::<T>::with_capacity(elem_count);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), data.as_mut_ptr() as *mut u8, bytes.len());
        data.set_len(elem_count);
    }
    Ok(data)
}

fn check_raw_storage_bytes(dtype: DType, elem_count: usize, bytes: Vec<u8>) -> Result<Vec<u8>> {
    let expected = storage_size_in_bytes(dtype, elem_count)?;
    if bytes.len() != expected {
        crate::bail!(
            "invalid ROCm buffer size for {dtype:?}: expected {expected} bytes, got {}",
            bytes.len()
        )
    }
    Ok(bytes)
}

fn packed_elem_count(dtype: DType, byte_count: usize) -> usize {
    match dtype {
        DType::F6E2M3 | DType::F6E3M2 => byte_count * 8 / 6,
        DType::F4 => byte_count * 2,
        _ => byte_count,
    }
}

fn cpu_storage_into_bytes(storage: CpuStorage) -> Result<(DType, usize, Vec<u8>)> {
    match storage {
        CpuStorage::U8(data) => {
            let elem_count = data.len();
            Ok((DType::U8, elem_count, data))
        }
        CpuStorage::U32(data) => {
            let elem_count = data.len();
            Ok((DType::U32, elem_count, slice_to_bytes(&data)))
        }
        CpuStorage::I16(data) => {
            let elem_count = data.len();
            Ok((DType::I16, elem_count, slice_to_bytes(&data)))
        }
        CpuStorage::I32(data) => {
            let elem_count = data.len();
            Ok((DType::I32, elem_count, slice_to_bytes(&data)))
        }
        CpuStorage::I64(data) => {
            let elem_count = data.len();
            Ok((DType::I64, elem_count, slice_to_bytes(&data)))
        }
        CpuStorage::BF16(data) => {
            let elem_count = data.len();
            Ok((DType::BF16, elem_count, slice_to_bytes(&data)))
        }
        CpuStorage::F16(data) => {
            let elem_count = data.len();
            Ok((DType::F16, elem_count, slice_to_bytes(&data)))
        }
        CpuStorage::F32(data) => {
            let elem_count = data.len();
            Ok((DType::F32, elem_count, slice_to_bytes(&data)))
        }
        CpuStorage::F64(data) => {
            let elem_count = data.len();
            Ok((DType::F64, elem_count, slice_to_bytes(&data)))
        }
        CpuStorage::F8E4M3(data) => {
            let elem_count = data.len();
            Ok((DType::F8E4M3, elem_count, slice_to_bytes(&data)))
        }
        CpuStorage::F6E2M3(data) => {
            let elem_count = packed_elem_count(DType::F6E2M3, data.len());
            Ok((DType::F6E2M3, elem_count, data))
        }
        CpuStorage::F6E3M2(data) => {
            let elem_count = packed_elem_count(DType::F6E3M2, data.len());
            Ok((DType::F6E3M2, elem_count, data))
        }
        CpuStorage::F4(data) => {
            let elem_count = packed_elem_count(DType::F4, data.len());
            Ok((DType::F4, elem_count, data))
        }
        CpuStorage::F8E8M0(data) => {
            let elem_count = data.len();
            Ok((DType::F8E8M0, elem_count, data))
        }
    }
}

fn cpu_storage_meta(storage: &CpuStorage) -> (DType, usize) {
    match storage {
        CpuStorage::U8(data) => (DType::U8, data.len()),
        CpuStorage::U32(data) => (DType::U32, data.len()),
        CpuStorage::I16(data) => (DType::I16, data.len()),
        CpuStorage::I32(data) => (DType::I32, data.len()),
        CpuStorage::I64(data) => (DType::I64, data.len()),
        CpuStorage::BF16(data) => (DType::BF16, data.len()),
        CpuStorage::F16(data) => (DType::F16, data.len()),
        CpuStorage::F32(data) => (DType::F32, data.len()),
        CpuStorage::F64(data) => (DType::F64, data.len()),
        CpuStorage::F8E4M3(data) => (DType::F8E4M3, data.len()),
        CpuStorage::F6E2M3(data) => (DType::F6E2M3, data.len()),
        CpuStorage::F6E3M2(data) => (DType::F6E3M2, data.len()),
        CpuStorage::F4(data) => (DType::F4, data.len()),
        CpuStorage::F8E8M0(data) => (DType::F8E8M0, data.len()),
    }
}

fn cpu_storage_from_bytes(dtype: DType, elem_count: usize, bytes: Vec<u8>) -> Result<CpuStorage> {
    match dtype {
        DType::U8 => {
            if bytes.len() != elem_count {
                crate::bail!(
                    "invalid ROCm buffer size for {dtype:?}: expected {elem_count} bytes, got {}",
                    bytes.len()
                )
            }
            Ok(CpuStorage::U8(bytes))
        }
        DType::U32 => Ok(CpuStorage::U32(vec_from_bytes::<u32>(
            &bytes, elem_count, dtype,
        )?)),
        DType::I16 => Ok(CpuStorage::I16(vec_from_bytes::<i16>(
            &bytes, elem_count, dtype,
        )?)),
        DType::I32 => Ok(CpuStorage::I32(vec_from_bytes::<i32>(
            &bytes, elem_count, dtype,
        )?)),
        DType::I64 => Ok(CpuStorage::I64(vec_from_bytes::<i64>(
            &bytes, elem_count, dtype,
        )?)),
        DType::BF16 => Ok(CpuStorage::BF16(vec_from_bytes::<bf16>(
            &bytes, elem_count, dtype,
        )?)),
        DType::F16 => Ok(CpuStorage::F16(vec_from_bytes::<f16>(
            &bytes, elem_count, dtype,
        )?)),
        DType::F32 => Ok(CpuStorage::F32(vec_from_bytes::<f32>(
            &bytes, elem_count, dtype,
        )?)),
        DType::F64 => Ok(CpuStorage::F64(vec_from_bytes::<f64>(
            &bytes, elem_count, dtype,
        )?)),
        DType::F8E4M3 => Ok(CpuStorage::F8E4M3(vec_from_bytes::<F8E4M3>(
            &bytes, elem_count, dtype,
        )?)),
        DType::F6E2M3 => Ok(CpuStorage::F6E2M3(check_raw_storage_bytes(
            dtype, elem_count, bytes,
        )?)),
        DType::F6E3M2 => Ok(CpuStorage::F6E3M2(check_raw_storage_bytes(
            dtype, elem_count, bytes,
        )?)),
        DType::F4 => Ok(CpuStorage::F4(check_raw_storage_bytes(
            dtype, elem_count, bytes,
        )?)),
        DType::F8E8M0 => Ok(CpuStorage::F8E8M0(check_raw_storage_bytes(
            dtype, elem_count, bytes,
        )?)),
    }
}

fn repeat_penalty_cpu_storage(
    logits: &CpuStorage,
    logits_layout: &Layout,
    token_ids: &CpuStorage,
    token_ids_layout: &Layout,
    penalty: f32,
) -> Result<CpuStorage> {
    if logits_layout.dims().len() != 1 {
        crate::bail!(
            "repeat-penalty expects rank-1 logits, got {:?}",
            logits_layout.shape()
        )
    }
    if token_ids_layout.dims().len() != 1 {
        crate::bail!(
            "repeat-penalty expects rank-1 token ids, got {:?}",
            token_ids_layout.shape()
        )
    }
    let logits = logits.as_slice::<f32>()?;
    let token_ids = token_ids.as_slice::<u32>()?;
    let mut logits = logits_layout
        .strided_index()
        .map(|index| logits[index])
        .collect::<Vec<_>>();
    for token_id_index in token_ids_layout.strided_index() {
        let token_id = token_ids[token_id_index] as usize;
        if let Some(logit) = logits.get_mut(token_id) {
            if *logit >= 0. {
                *logit /= penalty
            } else {
                *logit *= penalty
            }
        }
    }
    Ok(CpuStorage::F32(logits))
}

impl BackendStorage for RocmStorage {
    type Device = RocmDevice;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        let mut op = self.op1(
            "try_clone",
            Some(kernel_output_for_layout(layout, self.dtype)?),
        )?;
        op.input_layout = Some(kernel_layout(layout)?);
        if let Some(buffer) = kernels::tensor::try_clone(&op).map_err(Error::from)? {
            return Self::from_buffer(
                buffer,
                self.device.clone(),
                self.dtype,
                layout.shape().elem_count(),
                "try_clone",
            );
        }
        let storage = kernels::tensor::call_try_clone(op, || {
            let storage = self.to_cpu_storage_impl()?;
            storage.try_clone(layout)
        })?;
        Self::wrap(storage, self.device.clone(), "try_clone")
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        let op = self.op1(
            "transfer_to_cpu",
            Some(kernel_output(self.dtype, self.elem_count)?),
        )?;
        kernels::tensor::call_transfer_to_cpu(op, || self.to_cpu_storage_impl())
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let output_dtype = native_float_output(self.dtype);
        let mut op = self.op1(
            "affine",
            Some(kernel_output_for_layout(layout, output_dtype)?),
        )?;
        op.input_layout = Some(kernel_layout(layout)?);
        if let Some(buffer) =
            kernels::tensor::try_affine(&op, mul as f32, add as f32).map_err(Error::from)?
        {
            return Self::from_buffer(
                buffer,
                self.device.clone(),
                output_dtype,
                layout.shape().elem_count(),
                "affine",
            );
        }
        let storage = kernels::tensor::call_affine(op, || {
            let storage = self.to_cpu_storage_impl()?;
            storage.affine(layout, mul, add)
        })?;
        Self::wrap(storage, self.device.clone(), "affine")
    }

    fn powf(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let output_dtype = native_float_output(self.dtype);
        let mut op = self.op1(
            "powf",
            Some(kernel_output_for_layout(layout, output_dtype)?),
        )?;
        op.input_layout = Some(kernel_layout(layout)?);
        if let Some(buffer) = kernels::tensor::try_powf(&op, alpha as f32).map_err(Error::from)? {
            return Self::from_buffer(
                buffer,
                self.device.clone(),
                output_dtype,
                layout.shape().elem_count(),
                "powf",
            );
        }
        let storage = kernels::tensor::call_powf(op, || {
            let storage = self.to_cpu_storage_impl()?;
            storage.powf(layout, alpha)
        })?;
        Self::wrap(storage, self.device.clone(), "powf")
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let output_dtype = native_float_output(self.dtype);
        let mut op = self.op1("elu", Some(kernel_output_for_layout(layout, output_dtype)?))?;
        op.input_layout = Some(kernel_layout(layout)?);
        if let Some(buffer) = kernels::tensor::try_elu(&op, alpha as f32).map_err(Error::from)? {
            return Self::from_buffer(
                buffer,
                self.device.clone(),
                output_dtype,
                layout.shape().elem_count(),
                "elu",
            );
        }
        let storage = kernels::tensor::call_elu(op, || {
            let storage = self.to_cpu_storage_impl()?;
            storage.elu(layout, alpha)
        })?;
        Self::wrap(storage, self.device.clone(), "elu")
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, dims: &[usize]) -> Result<Self> {
        let output_dtype = match op {
            ReduceOp::ArgMin | ReduceOp::ArgMax => DType::U32,
            ReduceOp::Sum | ReduceOp::Min | ReduceOp::Max => native_float_output(self.dtype),
        };
        let mut kernel_op = self.op1(
            op.name(),
            Some(kernel_output(
                output_dtype,
                reduce_output_elem_count(layout, dims),
            )?),
        )?;
        kernel_op.input_layout = Some(kernel_layout(layout)?);
        if let Some(buffer) = kernels::tensor::try_reduce(&kernel_op, dims).map_err(Error::from)? {
            return Self::from_buffer(
                buffer,
                self.device.clone(),
                output_dtype,
                reduce_output_elem_count(layout, dims),
                op.name(),
            );
        }
        let storage = kernels::tensor::call_reduce(kernel_op, || {
            let storage = self.to_cpu_storage_impl()?;
            storage.reduce_op(op, layout, dims)
        })?;
        Self::wrap(storage, self.device.clone(), op.name())
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let mut kernel_op = self.op2(
            cmp_op_name(op),
            rhs,
            Some(kernel_output_for_layout(lhs_l, DType::U8)?),
        )?;
        kernel_op.lhs_layout = Some(kernel_layout(lhs_l)?);
        kernel_op.rhs_layout = Some(kernel_layout(rhs_l)?);
        if let Some(buffer) = kernels::tensor::try_cmp(&kernel_op).map_err(Error::from)? {
            return Self::from_buffer(
                buffer,
                self.device.clone(),
                DType::U8,
                lhs_l.shape().elem_count(),
                "cmp",
            );
        }
        let storage = kernels::tensor::call_cmp(kernel_op, || {
            let lhs = self.to_cpu_storage_impl()?;
            let rhs = rhs.to_cpu_storage_impl()?;
            lhs.cmp(op, &rhs, lhs_l, rhs_l)
        })?;
        Self::wrap(storage, self.device.clone(), "cmp")
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        let mut op = self.op1("to_dtype", Some(kernel_output_for_layout(layout, dtype)?))?;
        op.input_layout = Some(kernel_layout(layout)?);
        if let Some(buffer) = kernels::tensor::try_to_dtype(&op).map_err(Error::from)? {
            return Self::from_buffer(
                buffer,
                self.device.clone(),
                dtype,
                layout.shape().elem_count(),
                "to_dtype",
            );
        }
        let storage = kernels::tensor::call_to_dtype(op, || {
            let storage = self.to_cpu_storage_impl()?;
            storage.to_dtype(layout, dtype)
        })?;
        Self::wrap(storage, self.device.clone(), "to_dtype")
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let output_dtype = match self.dtype {
            DType::BF16 | DType::F8E4M3 => self.dtype,
            _ => DType::F32,
        };
        let mut op = self.op1(
            B::NAME,
            Some(kernel_output_for_layout(layout, output_dtype)?),
        )?;
        op.input_layout = Some(kernel_layout(layout)?);
        if let Some(buffer) = kernels::tensor::try_unary(&op).map_err(Error::from)? {
            return Self::from_buffer(
                buffer,
                self.device.clone(),
                output_dtype,
                layout.shape().elem_count(),
                B::NAME,
            );
        }
        let storage = kernels::tensor::call_unary(op, || {
            let storage = self.to_cpu_storage_impl()?;
            storage.unary_impl::<B>(layout)
        })?;
        Self::wrap(storage, self.device.clone(), B::NAME)
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let output_dtype = same_native_float_output(self.dtype, rhs.dtype);
        let mut op = self.op2(
            B::NAME,
            rhs,
            Some(kernel_output_for_layout(lhs_l, output_dtype)?),
        )?;
        op.lhs_layout = Some(kernel_layout(lhs_l)?);
        op.rhs_layout = Some(kernel_layout(rhs_l)?);
        if let Some(buffer) = kernels::tensor::try_binary(&op).map_err(Error::from)? {
            return Self::from_buffer(
                buffer,
                self.device.clone(),
                output_dtype,
                lhs_l.shape().elem_count(),
                B::NAME,
            );
        }
        let storage = kernels::tensor::call_binary(op, || {
            let lhs = self.to_cpu_storage_impl()?;
            let rhs = rhs.to_cpu_storage_impl()?;
            lhs.binary_impl::<B>(&rhs, lhs_l, rhs_l)
        })?;
        Self::wrap(storage, self.device.clone(), B::NAME)
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        let output_dtype = t.dtype;
        let op = self.op3(
            "where",
            t,
            f,
            Some(kernel_output_for_layout(layout, output_dtype)?),
        )?;
        let cond_l = kernel_layout(layout)?;
        let true_l = kernel_layout(t_l)?;
        let false_l = kernel_layout(f_l)?;
        if let Some(buffer) =
            kernels::tensor::try_where_cond(&op, &cond_l, &true_l, &false_l).map_err(Error::from)?
        {
            return Self::from_buffer(
                buffer,
                self.device.clone(),
                output_dtype,
                layout.shape().elem_count(),
                "where",
            );
        }
        let storage = kernels::tensor::call_where_cond(op, || {
            let cond = self.to_cpu_storage_impl()?;
            let t = t.to_cpu_storage_impl()?;
            let f = f.to_cpu_storage_impl()?;
            cond.where_cond(layout, &t, t_l, &f, f_l)
        })?;
        Self::wrap(storage, self.device.clone(), "where")
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        let l_out = params.l_out();
        let elem_count = params.b_size * params.c_out * l_out;
        let output_dtype = same_native_float_output(self.dtype, kernel.dtype);
        let mut op = self.op2(
            "conv1d",
            kernel,
            Some(kernel_output(output_dtype, elem_count)?),
        )?;
        op.lhs_layout = Some(kernel_layout(l)?);
        op.rhs_layout = Some(kernel_layout(kernel_l)?);
        let kernel_params = kernels::tensor::Conv1dParams {
            padding: params.padding,
            stride: params.stride,
            dilation: params.dilation,
            l_out,
            elem_count,
        };
        if let Some(buffer) =
            kernels::tensor::try_conv1d(&op, kernel_params).map_err(Error::from)?
        {
            return Self::from_buffer(
                buffer,
                self.device.clone(),
                output_dtype,
                elem_count,
                "conv1d",
            );
        }
        let storage = kernels::tensor::call_conv1d(op, || {
            let lhs = self.to_cpu_storage_impl()?;
            let kernel = kernel.to_cpu_storage_impl()?;
            lhs.conv1d(l, &kernel, kernel_l, params)
        })?;
        Self::wrap(storage, self.device.clone(), "conv1d")
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        let l_out = params.l_out();
        let elem_count = params.b_size * params.c_out * l_out;
        let output_dtype = same_native_float_output(self.dtype, kernel.dtype);
        let mut op = self.op2(
            "conv_transpose1d",
            kernel,
            Some(kernel_output(output_dtype, elem_count)?),
        )?;
        op.lhs_layout = Some(kernel_layout(l)?);
        op.rhs_layout = Some(kernel_layout(kernel_l)?);
        let kernel_params = kernels::tensor::Conv1dParams {
            padding: params.padding,
            stride: params.stride,
            dilation: params.dilation,
            l_out,
            elem_count,
        };
        if let Some(buffer) =
            kernels::tensor::try_conv_transpose1d(&op, kernel_params).map_err(Error::from)?
        {
            return Self::from_buffer(
                buffer,
                self.device.clone(),
                output_dtype,
                elem_count,
                "conv_transpose1d",
            );
        }
        let storage = kernels::tensor::call_conv_transpose1d(op, || {
            let lhs = self.to_cpu_storage_impl()?;
            let kernel = kernel.to_cpu_storage_impl()?;
            lhs.conv_transpose1d(l, &kernel, kernel_l, params)
        })?;
        Self::wrap(storage, self.device.clone(), "conv_transpose1d")
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        let (out_h, out_w) = (params.out_h(), params.out_w());
        let elem_count = params.b_size * params.c_out * out_h * out_w;
        let output_dtype = same_native_float_output(self.dtype, kernel.dtype);
        let mut op = self.op2(
            "conv2d",
            kernel,
            Some(kernel_output(output_dtype, elem_count)?),
        )?;
        op.lhs_layout = Some(kernel_layout(l)?);
        op.rhs_layout = Some(kernel_layout(kernel_l)?);
        let kernel_params = kernels::tensor::Conv2dParams {
            padding: params.padding,
            stride: params.stride,
            dilation: params.dilation,
            out_h,
            out_w,
            elem_count,
        };
        if let Some(buffer) =
            kernels::tensor::try_conv2d(&op, kernel_params).map_err(Error::from)?
        {
            return Self::from_buffer(
                buffer,
                self.device.clone(),
                output_dtype,
                elem_count,
                "conv2d",
            );
        }
        let storage = kernels::tensor::call_conv2d(op, || {
            let lhs = self.to_cpu_storage_impl()?;
            let kernel = kernel.to_cpu_storage_impl()?;
            lhs.conv2d(l, &kernel, kernel_l, params)
        })?;
        Self::wrap(storage, self.device.clone(), "conv2d")
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        let (out_h, out_w) = (params.out_h(), params.out_w());
        let elem_count = params.b_size * params.c_out * out_h * out_w;
        let output_dtype = same_native_float_output(self.dtype, kernel.dtype);
        let mut op = self.op2(
            "conv_transpose2d",
            kernel,
            Some(kernel_output(output_dtype, elem_count)?),
        )?;
        op.lhs_layout = Some(kernel_layout(l)?);
        op.rhs_layout = Some(kernel_layout(kernel_l)?);
        let kernel_params = kernels::tensor::Conv2dParams {
            padding: params.padding,
            stride: params.stride,
            dilation: params.dilation,
            out_h,
            out_w,
            elem_count,
        };
        if let Some(buffer) =
            kernels::tensor::try_conv_transpose2d(&op, kernel_params).map_err(Error::from)?
        {
            return Self::from_buffer(
                buffer,
                self.device.clone(),
                output_dtype,
                elem_count,
                "conv_transpose2d",
            );
        }
        let storage = kernels::tensor::call_conv_transpose2d(op, || {
            let lhs = self.to_cpu_storage_impl()?;
            let kernel = kernel.to_cpu_storage_impl()?;
            lhs.conv_transpose2d(l, &kernel, kernel_l, params)
        })?;
        Self::wrap(storage, self.device.clone(), "conv_transpose2d")
    }

    fn avg_pool2d(
        &self,
        l: &Layout,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        let dims = l.dims();
        let elem_count = dims[0]
            * dims[1]
            * ((dims[2] - kernel.0) / stride.0 + 1)
            * ((dims[3] - kernel.1) / stride.1 + 1);
        let output_dtype = native_float_output(self.dtype);
        let mut op = self.op1("avg_pool2d", Some(kernel_output(output_dtype, elem_count)?))?;
        op.input_layout = Some(kernel_layout(l)?);
        if let Some(buffer) =
            kernels::tensor::try_pool2d(&op, kernel, stride, false).map_err(Error::from)?
        {
            return Self::from_buffer(
                buffer,
                self.device.clone(),
                output_dtype,
                elem_count,
                "avg_pool2d",
            );
        }
        let storage = kernels::tensor::call_avg_pool2d(op, || {
            let storage = self.to_cpu_storage_impl()?;
            storage.avg_pool2d(l, kernel, stride)
        })?;
        Self::wrap(storage, self.device.clone(), "avg_pool2d")
    }

    fn max_pool2d(
        &self,
        l: &Layout,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        let dims = l.dims();
        let elem_count = dims[0]
            * dims[1]
            * ((dims[2] - kernel.0) / stride.0 + 1)
            * ((dims[3] - kernel.1) / stride.1 + 1);
        let output_dtype = native_float_output(self.dtype);
        let mut op = self.op1("max_pool2d", Some(kernel_output(output_dtype, elem_count)?))?;
        op.input_layout = Some(kernel_layout(l)?);
        if let Some(buffer) =
            kernels::tensor::try_pool2d(&op, kernel, stride, true).map_err(Error::from)?
        {
            return Self::from_buffer(
                buffer,
                self.device.clone(),
                output_dtype,
                elem_count,
                "max_pool2d",
            );
        }
        let storage = kernels::tensor::call_max_pool2d(op, || {
            let storage = self.to_cpu_storage_impl()?;
            storage.max_pool2d(l, kernel, stride)
        })?;
        Self::wrap(storage, self.device.clone(), "max_pool2d")
    }

    fn upsample_nearest1d(&self, l: &Layout, size: usize) -> Result<Self> {
        let dims = l.dims();
        let elem_count = dims[0] * dims[1] * size;
        let output_dtype = native_float_output(self.dtype);
        let mut op = self.op1(
            "upsample_nearest1d",
            Some(kernel_output(output_dtype, elem_count)?),
        )?;
        op.input_layout = Some(kernel_layout(l)?);
        if let Some(buffer) =
            kernels::tensor::try_upsample_nearest1d(&op, size).map_err(Error::from)?
        {
            return Self::from_buffer(
                buffer,
                self.device.clone(),
                output_dtype,
                elem_count,
                "upsample_nearest1d",
            );
        }
        let storage = kernels::tensor::call_upsample_nearest1d(op, || {
            let storage = self.to_cpu_storage_impl()?;
            storage.upsample_nearest1d(l, size)
        })?;
        Self::wrap(storage, self.device.clone(), "upsample_nearest1d")
    }

    fn upsample_nearest2d(&self, l: &Layout, h: usize, w: usize) -> Result<Self> {
        let dims = l.dims();
        let elem_count = dims[0] * dims[1] * h * w;
        let output_dtype = native_float_output(self.dtype);
        let mut op = self.op1(
            "upsample_nearest2d",
            Some(kernel_output(output_dtype, elem_count)?),
        )?;
        op.input_layout = Some(kernel_layout(l)?);
        if let Some(buffer) =
            kernels::tensor::try_upsample_nearest2d(&op, h, w).map_err(Error::from)?
        {
            return Self::from_buffer(
                buffer,
                self.device.clone(),
                output_dtype,
                elem_count,
                "upsample_nearest2d",
            );
        }
        let storage = kernels::tensor::call_upsample_nearest2d(op, || {
            let storage = self.to_cpu_storage_impl()?;
            storage.upsample_nearest2d(l, h, w)
        })?;
        Self::wrap(storage, self.device.clone(), "upsample_nearest2d")
    }

    fn upsample_bilinear2d(
        &self,
        l: &Layout,
        h: usize,
        w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Self> {
        let dims = l.dims();
        let elem_count = dims[0] * dims[1] * h * w;
        let output_dtype = native_float_output(self.dtype);
        let mut op = self.op1(
            "upsample_bilinear2d",
            Some(kernel_output(output_dtype, elem_count)?),
        )?;
        op.input_layout = Some(kernel_layout(l)?);
        if let Some(buffer) =
            kernels::tensor::try_upsample_bilinear2d(&op, h, w, align_corners, scale_h, scale_w)
                .map_err(Error::from)?
        {
            return Self::from_buffer(
                buffer,
                self.device.clone(),
                output_dtype,
                elem_count,
                "upsample_bilinear2d",
            );
        }
        let storage = kernels::tensor::call_upsample_bilinear2d(op, || {
            let storage = self.to_cpu_storage_impl()?;
            storage.upsample_bilinear2d(l, h, w, align_corners, scale_h, scale_w)
        })?;
        Self::wrap(storage, self.device.clone(), "upsample_bilinear2d")
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        let output_dtype = self.dtype;
        let mut op = self.op2(
            "gather",
            ids,
            Some(kernel_output_for_layout(ids_l, output_dtype)?),
        )?;
        op.lhs_layout = Some(kernel_layout(l)?);
        op.rhs_layout = Some(kernel_layout(ids_l)?);
        if let Some(buffer) = kernels::tensor::try_gather(&op, dim).map_err(Error::from)? {
            return Self::from_buffer(
                buffer,
                self.device.clone(),
                output_dtype,
                ids_l.shape().elem_count(),
                "gather",
            );
        }
        let storage = kernels::tensor::call_gather(op, || {
            let storage = self.to_cpu_storage_impl()?;
            let ids = ids.to_cpu_storage_impl()?;
            storage.gather(l, &ids, ids_l, dim)
        })?;
        Self::wrap(storage, self.device.clone(), "gather")
    }

    fn scatter_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let op = kernels::tensor::InplaceOp3 {
            name: "scatter_set",
            dst: self.tensor_arg()?,
            second: ids.tensor_arg()?,
            third: src.tensor_arg()?,
        };
        let dst_kernel_l = kernel_layout(l)?;
        let ids_kernel_l = kernel_layout(ids_l)?;
        let src_kernel_l = kernel_layout(src_l)?;
        if kernels::tensor::try_scatter(
            &op,
            &dst_kernel_l,
            &ids_kernel_l,
            &src_kernel_l,
            dim,
            false,
        )
        .map_err(Error::from)?
        {
            return Ok(());
        }
        kernels::tensor::call_scatter_set(op, || {
            let mut storage = self.to_cpu_storage_impl()?;
            let ids = ids.to_cpu_storage_impl()?;
            let src = src.to_cpu_storage_impl()?;
            storage.scatter_set(l, &ids, ids_l, &src, src_l, dim)?;
            self.set_cpu_storage_owned(storage, "scatter_set")
        })
    }

    fn scatter_add_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let op = kernels::tensor::InplaceOp3 {
            name: "scatter_add",
            dst: self.tensor_arg()?,
            second: ids.tensor_arg()?,
            third: src.tensor_arg()?,
        };
        let dst_kernel_l = kernel_layout(l)?;
        let ids_kernel_l = kernel_layout(ids_l)?;
        let src_kernel_l = kernel_layout(src_l)?;
        if kernels::tensor::try_scatter(&op, &dst_kernel_l, &ids_kernel_l, &src_kernel_l, dim, true)
            .map_err(Error::from)?
        {
            return Ok(());
        }
        kernels::tensor::call_scatter_add_set(op, || {
            let mut storage = self.to_cpu_storage_impl()?;
            let ids = ids.to_cpu_storage_impl()?;
            let src = src.to_cpu_storage_impl()?;
            storage.scatter_add_set(l, &ids, ids_l, &src, src_l, dim)?;
            self.set_cpu_storage_owned(storage, "scatter_add")
        })
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        let output_elem_count = index_select_output_elem_count(l, ids_l, dim);
        let mut op = self.op2(
            "index_select",
            ids,
            Some(kernel_output(self.dtype, output_elem_count)?),
        )?;
        op.lhs_layout = Some(kernel_layout(l)?);
        op.rhs_layout = Some(kernel_layout(ids_l)?);
        if let Some(buffer) = kernels::tensor::try_index_select(&op, dim).map_err(Error::from)? {
            return Self::from_buffer(
                buffer,
                self.device.clone(),
                self.dtype,
                output_elem_count,
                "index_select",
            );
        }
        let storage = kernels::tensor::call_index_select(op, || {
            let storage = self.to_cpu_storage_impl()?;
            let ids = ids.to_cpu_storage_impl()?;
            storage.index_select(&ids, l, ids_l, dim)
        })?;
        Self::wrap(storage, self.device.clone(), "index_select")
    }

    fn index_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let output_dtype = same_native_float_output(self.dtype, src.dtype);
        let op = self.op3(
            "index_add",
            ids,
            src,
            Some(kernel_output_for_layout(l, output_dtype)?),
        )?;
        let input_kernel_l = kernel_layout(l)?;
        let ids_kernel_l = kernel_layout(ids_l)?;
        let src_kernel_l = kernel_layout(src_l)?;
        if let Some(buffer) =
            kernels::tensor::try_index_add(&op, &input_kernel_l, &ids_kernel_l, &src_kernel_l, dim)
                .map_err(Error::from)?
        {
            return Self::from_buffer(
                buffer,
                self.device.clone(),
                output_dtype,
                l.shape().elem_count(),
                "index_add",
            );
        }
        let storage = kernels::tensor::call_index_add(op, || {
            let storage = self.to_cpu_storage_impl()?;
            let ids = ids.to_cpu_storage_impl()?;
            let src = src.to_cpu_storage_impl()?;
            storage.index_add(l, &ids, ids_l, &src, src_l, dim)
        })?;
        Self::wrap(storage, self.device.clone(), "index_add")
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let (b, m, n, _) = bmnk;
        let output_dtype = same_native_float_output(self.dtype, rhs.dtype);
        let mut op = self.op2("matmul", rhs, Some(kernel_output(output_dtype, b * m * n)?))?;
        op.lhs_layout = Some(kernel_layout(lhs_l)?);
        op.rhs_layout = Some(kernel_layout(rhs_l)?);
        if let Some(buffer) = kernels::tensor::try_matmul(&op, bmnk).map_err(Error::from)? {
            return Self::from_buffer(
                buffer,
                self.device.clone(),
                output_dtype,
                b * m * n,
                "matmul",
            );
        }
        let storage = kernels::tensor::call_matmul(op, || {
            let lhs = self.to_cpu_storage_impl()?;
            let rhs = rhs.to_cpu_storage_impl()?;
            lhs.matmul(&rhs, bmnk, lhs_l, rhs_l)
        })?;
        Self::wrap(storage, self.device.clone(), "matmul")
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let op = kernels::tensor::InplaceOp2 {
            name: "copy_strided_src",
            dst: dst.tensor_arg()?,
            src: self.tensor_arg()?,
            copy: Some(kernels::tensor::CopySpec::StridedSrc {
                dst_offset,
                src_layout: kernel_layout(src_l)?,
            }),
        };
        kernels::tensor::call_copy_strided_src(op, || {
            let storage = self.to_cpu_storage_impl()?;
            let mut dst_storage = dst.to_cpu_storage_impl()?;
            storage.copy_strided_src(&mut dst_storage, dst_offset, src_l)?;
            dst.set_cpu_storage_owned(dst_storage, "copy_strided_src")
        })
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()> {
        let op = kernels::tensor::InplaceOp2 {
            name: "copy2d",
            dst: dst.tensor_arg()?,
            src: self.tensor_arg()?,
            copy: Some(kernels::tensor::CopySpec::Copy2d {
                d1,
                d2,
                src_stride1,
                dst_stride1,
                src_offset,
                dst_offset,
            }),
        };
        kernels::tensor::call_copy2d(op, || {
            let storage = self.to_cpu_storage_impl()?;
            let mut dst_storage = dst.to_cpu_storage_impl()?;
            storage.copy2d(
                &mut dst_storage,
                d1,
                d2,
                src_stride1,
                dst_stride1,
                src_offset,
                dst_offset,
            )?;
            dst.set_cpu_storage_owned(dst_storage, "copy2d")
        })
    }

    fn const_set(&mut self, scalar: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        let kernel_scalar = kernel_scalar(scalar)?;
        let op = kernels::tensor::InplaceOp1 {
            name: "const_set",
            dst: self.tensor_arg()?,
            dst_layout: Some(kernel_layout(layout)?),
            scalar: Some(kernel_scalar),
        };
        kernels::tensor::call_const_set(op, || {
            let mut storage = self.to_cpu_storage_impl()?;
            storage.const_set(scalar, layout)?;
            self.set_cpu_storage_owned(storage, "const_set")
        })
    }
}

impl BackendDevice for RocmDevice {
    type Storage = RocmStorage;

    fn new(gpu_id: usize) -> Result<Self> {
        let inner = Arc::new(kernels::Device::new(gpu_id)?);
        Ok(Self { inner })
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Rocm {
            gpu_id: self.ordinal(),
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &rhs.inner)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        ensure_supported_storage_dtype(dtype, "zeros")?;
        let op = kernels::device::AllocOp {
            name: "zeros",
            device: self.kernel_device(),
            output: kernel_output(dtype, shape.elem_count())?,
        };
        let buffer = kernels::device::call_zeros(op)?;
        RocmStorage::from_buffer(buffer, self.clone(), dtype, shape.elem_count(), "zeros")
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        ensure_supported_storage_dtype(dtype, "alloc_uninit")?;
        let op = kernels::device::AllocOp {
            name: "alloc_uninit",
            device: self.kernel_device(),
            output: kernel_output(dtype, shape.elem_count())?,
        };
        let buffer = kernels::device::call_alloc_uninit(op)?;
        RocmStorage::from_buffer(
            buffer,
            self.clone(),
            dtype,
            shape.elem_count(),
            "alloc_uninit",
        )
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data: &[T]) -> Result<Self::Storage> {
        ensure_supported_storage_dtype(T::DTYPE, "storage_from_slice")?;
        let op = kernels::device::SliceLoadOp {
            name: "storage_from_slice",
            device: self.kernel_device(),
            dtype: kernel_dtype(T::DTYPE)?,
            elem_count: data.len(),
        };
        let bytes = slice_to_bytes(data);
        let buffer = kernels::device::call_storage_from_host_bytes(op, &bytes)?;
        RocmStorage::from_buffer(
            buffer,
            self.clone(),
            T::DTYPE,
            data.len(),
            "storage_from_slice",
        )
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        let (dtype, elem_count) = cpu_storage_meta(storage);
        let op = kernels::device::SliceLoadOp {
            name: "storage_from_cpu_storage",
            device: self.kernel_device(),
            dtype: kernel_dtype(dtype)?,
            elem_count,
        };
        let (_, _, bytes) = cpu_storage_into_bytes(storage.clone())?;
        let buffer = kernels::device::call_storage_from_host_bytes(op, &bytes)?;
        RocmStorage::from_buffer(
            buffer,
            self.clone(),
            dtype,
            elem_count,
            "storage_from_cpu_storage",
        )
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        let (dtype, elem_count) = cpu_storage_meta(&storage);
        let op = kernels::device::SliceLoadOp {
            name: "storage_from_cpu_storage_owned",
            device: self.kernel_device(),
            dtype: kernel_dtype(dtype)?,
            elem_count,
        };
        let (_, _, bytes) = cpu_storage_into_bytes(storage)?;
        let buffer = kernels::device::call_storage_from_host_bytes(op, &bytes)?;
        RocmStorage::from_buffer(
            buffer,
            self.clone(),
            dtype,
            elem_count,
            "storage_from_cpu_storage_owned",
        )
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<Self::Storage> {
        if !matches!(dtype, DType::F32 | DType::BF16 | DType::F8E4M3) {
            return Err(Error::UnsupportedDTypeForOp(dtype, "rand_uniform").bt());
        }
        let op = kernels::device::AllocOp {
            name: "rand_uniform",
            device: self.kernel_device(),
            output: kernel_output(dtype, shape.elem_count())?,
        };
        let seed = self.inner.next_seed();
        if let Some(buffer) = kernels::device::try_rand_uniform(op, seed, lo as f32, up as f32)
            .map_err(Error::from)?
        {
            return RocmStorage::from_buffer(
                buffer,
                self.clone(),
                dtype,
                shape.elem_count(),
                "rand_uniform",
            );
        }
        let storage = kernels::device::call_rand_uniform(op, || {
            crate::cpu_backend::CpuDevice.rand_uniform(shape, dtype, lo, up)
        })?;
        RocmStorage::wrap(storage, self.clone(), "rand_uniform")
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Self::Storage> {
        if !matches!(dtype, DType::F32 | DType::BF16 | DType::F8E4M3) {
            return Err(Error::UnsupportedDTypeForOp(dtype, "rand_normal").bt());
        }
        let op = kernels::device::AllocOp {
            name: "rand_normal",
            device: self.kernel_device(),
            output: kernel_output(dtype, shape.elem_count())?,
        };
        let seed = self.inner.next_seed();
        if let Some(buffer) = kernels::device::try_rand_normal(op, seed, mean as f32, std as f32)
            .map_err(Error::from)?
        {
            return RocmStorage::from_buffer(
                buffer,
                self.clone(),
                dtype,
                shape.elem_count(),
                "rand_normal",
            );
        }
        let storage = kernels::device::call_rand_normal(op, || {
            crate::cpu_backend::CpuDevice.rand_normal(shape, dtype, mean, std)
        })?;
        RocmStorage::wrap(storage, self.clone(), "rand_normal")
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        let op = kernels::device::DeviceOp {
            name: "set_seed",
            device: self.kernel_device(),
        };
        kernels::device::call_set_seed(op, || {
            self.inner.set_seed(seed);
            Ok::<(), Error>(())
        })
    }

    fn get_current_seed(&self) -> Result<u64> {
        let op = kernels::device::DeviceOp {
            name: "get_current_seed",
            device: self.kernel_device(),
        };
        kernels::device::call_get_current_seed(op, || {
            Ok::<u64, Error>(self.inner.get_current_seed())
        })
    }

    fn synchronize(&self) -> Result<()> {
        self.inner.synchronize()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{CpuStorage, DType, Device, Storage, Tensor, WithDType};

    fn assert_rocm_round_trip<T>(device: &Device, data: &[T]) -> crate::Result<()>
    where
        T: WithDType + std::fmt::Debug + PartialEq,
    {
        let tensor = Tensor::from_slice(data, data.len(), device)?;
        assert!(tensor.device().is_rocm());
        assert_eq!(tensor.dtype(), T::DTYPE);
        assert_eq!(tensor.to_vec1::<T>()?, data);

        let cpu = tensor.to_device(&Device::Cpu)?;
        assert!(cpu.device().is_cpu());
        assert_eq!(cpu.dtype(), T::DTYPE);
        assert_eq!(cpu.to_vec1::<T>()?, data);
        Ok(())
    }

    #[test]
    fn dummy_rocm_runs_f32_ops_on_cpu_storage() -> crate::Result<()> {
        let device = Device::new_rocm(0)?;
        let lhs = Tensor::from_slice(&[1f32, 2., 3., 4.], (2, 2), &device)?;
        let rhs = Tensor::from_slice(&[5f32, 6., 7., 8.], (2, 2), &device)?;
        let out = lhs.matmul(&rhs)?;

        assert!(out.device().is_rocm());
        assert_eq!(out.to_vec2::<f32>()?, vec![vec![19., 22.], vec![43., 50.]]);

        let cpu = out.to_device(&Device::Cpu)?;
        assert!(cpu.device().is_cpu());
        assert_eq!(cpu.to_vec2::<f32>()?, vec![vec![19., 22.], vec![43., 50.]]);
        Ok(())
    }

    #[test]
    fn dummy_rocm_round_trips_non_f32_storage_dtypes() -> crate::Result<()> {
        let device = Device::new_rocm(0)?;
        assert_rocm_round_trip(&device, &[1u8, 2, 3])?;
        assert_rocm_round_trip(&device, &[1u32, 2, 3])?;
        assert_rocm_round_trip(&device, &[-1i16, 2, 3])?;
        assert_rocm_round_trip(&device, &[-1i32, 2, 3])?;
        assert_rocm_round_trip(&device, &[-1i64, 2, 3])?;
        assert_rocm_round_trip(
            &device,
            &[half::bf16::from_f32(1.25), half::bf16::from_f32(-2.5)],
        )?;
        assert_rocm_round_trip(
            &device,
            &[half::f16::from_f32(1.25), half::f16::from_f32(-2.5)],
        )?;
        assert_rocm_round_trip(&device, &[1f32, -2.5, 3.25])?;
        assert_rocm_round_trip(&device, &[1f64, -2.5, 3.25])?;
        assert_rocm_round_trip(
            &device,
            &[
                float8::F8E4M3::from_f64(1.0),
                float8::F8E4M3::from_f64(-2.0),
            ],
        )?;
        assert!(Tensor::zeros((3,), DType::BF16, &device).is_ok());
        assert!(Tensor::zeros((3,), DType::F64, &device).is_ok());
        assert!(Tensor::zeros((3,), DType::F4, &device).is_ok());
        Ok(())
    }

    #[test]
    fn dummy_rocm_preserves_raw_packed_safetensor_dtypes() -> crate::Result<()> {
        let device = Device::new_rocm(0)?;
        let raw = [0b1010_0101, 0b0000_0111];
        let tensor = Tensor::from_raw_buffer(&raw, DType::F4, &[3], &device)?;

        assert!(tensor.device().is_rocm());
        assert_eq!(tensor.dtype(), DType::F4);

        let cpu = tensor.to_device(&Device::Cpu)?;
        let storage = cpu.storage();
        match &*storage {
            Storage::Cpu(CpuStorage::F4(bytes)) => assert_eq!(bytes, &raw),
            storage => panic!("expected CPU F4 storage, got {storage:?}"),
        }
        Ok(())
    }

    #[test]
    fn dummy_rocm_allows_u32_index_tensors() -> crate::Result<()> {
        let device = Device::new_rocm(0)?;
        let values = Tensor::from_slice(&[10f32, 11., 12., 13., 14., 15.], (3, 2), &device)?;
        let indexes = Tensor::from_slice(&[2u32, 0], 2, &device)?;
        let out = values.index_select(&indexes, 0)?;

        assert!(indexes.device().is_rocm());
        assert!(out.device().is_rocm());
        assert_eq!(out.to_vec2::<f32>()?, vec![vec![14., 15.], vec![10., 11.]]);

        let arange = Tensor::arange(0u32, 3u32, &device)?.to_dtype(DType::F32)?;
        assert_eq!(arange.to_vec1::<f32>()?, vec![0., 1., 2.]);
        Ok(())
    }

    #[test]
    fn dummy_rocm_routes_argsort_through_kernel_wrapper() -> crate::Result<()> {
        let device = Device::new_rocm(0)?;
        let values = Tensor::from_slice(&[3f32, 1., 2., 4.], (2, 2), &device)?;
        let indexes = values.arg_sort_last_dim(true)?;

        assert!(indexes.device().is_rocm());
        assert_eq!(indexes.dtype(), DType::U32);
        assert_eq!(indexes.to_vec2::<u32>()?, vec![vec![1, 0], vec![0, 1]]);
        Ok(())
    }

    #[test]
    fn rocm_device_identity_is_exact() -> crate::Result<()> {
        let d1 = Device::new_rocm(0)?;
        let d2 = d1.clone();
        let d3 = Device::new_rocm(0)?;

        assert!(d1.same_device(&d2));
        assert!(!d1.same_device(&d3));
        assert_eq!(d1.location(), d3.location());

        let lhs = Tensor::from_slice(&[1f32, 2.], 2, &d1)?;
        let rhs = Tensor::from_slice(&[3f32, 4.], 2, &d3)?;
        assert!(lhs.add(&rhs).is_err());
        Ok(())
    }

    #[test]
    fn rocm_device_seed_and_sync_use_kernel_device() -> crate::Result<()> {
        let device = Device::new_rocm(0)?;
        device.set_seed(42)?;
        assert_eq!(device.get_current_seed()?, 42);
        device.synchronize()?;
        Ok(())
    }
}
