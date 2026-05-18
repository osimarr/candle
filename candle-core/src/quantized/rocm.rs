use super::{k_quants, GgmlDType, QStorage, QuantizedType};
use crate::backend::{BackendDevice, BackendStorage};
use crate::{CpuStorage, DType, Layout, Result, RocmDevice, RocmStorage, Shape};
use candle_rocm_kernels as kernels;
use half::{bf16, f16, slice::HalfFloatSliceExt};
use std::borrow::Cow;
use std::mem;

pub struct QRocmStorage {
    dtype: GgmlDType,
    device: RocmDevice,
    storage: QRocmStorageData,
}

enum QRocmStorageData {
    Raw {
        buffer: kernels::Buffer,
        elem_count: usize,
    },
    NativeFloat16 {
        buffer: kernels::Buffer,
        elem_count: usize,
    },
}

impl QRocmStorage {
    pub fn zeros(device: &RocmDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        if supports_native_float16(dtype) {
            let buffer = device
                .kernel_device()
                .allocate_zeroed(dtype.type_size() * elem_count)?;
            return Ok(Self {
                dtype,
                device: device.clone(),
                storage: QRocmStorageData::NativeFloat16 { buffer, elem_count },
            });
        }
        let size_in_bytes = quantized_storage_size_in_bytes(dtype, elem_count)?;
        let buffer = kernels::quantized::zeros(device.kernel_device(), size_in_bytes)?;
        Ok(Self {
            dtype,
            device: device.clone(),
            storage: QRocmStorageData::Raw { buffer, elem_count },
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &RocmDevice {
        &self.device
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<RocmStorage> {
        if let QRocmStorageData::NativeFloat16 {
            buffer,
            elem_count: storage_elem_count,
        } = &self.storage
        {
            if elem_count != *storage_elem_count {
                crate::bail!(
                    "invalid {:?} dequantize elem count: storage has {storage_elem_count}, requested {elem_count}",
                    self.dtype
                )
            }
            if let Some(buffer) = try_dequantize_native_float16_to_f32(
                self.dtype,
                self.device.kernel_device(),
                buffer,
                elem_count,
            )
            .map_err(crate::Error::from)?
            {
                return RocmStorage::from_buffer(
                    buffer,
                    self.device.clone(),
                    DType::F32,
                    elem_count,
                    native_float16_op_name("dequantize", self.dtype),
                );
            }
        }
        let op = self.op("dequantize")?;
        let storage = kernels::quantized::call_dequantize(op, || {
            let storage = self.raw_cpu_storage(elem_count, "dequantize")?;
            storage.dequantize(elem_count)
        })?;
        self.device.storage_from_cpu_storage_owned(storage)
    }

    pub fn quantize(&mut self, src: &RocmStorage) -> Result<()> {
        if let QRocmStorageData::NativeFloat16 { buffer, elem_count } = &mut self.storage {
            let src = src.tensor_arg()?;
            if src.elem_count() != *elem_count {
                crate::bail!(
                    "invalid {:?} quantize elem count: storage has {elem_count}, input has {}",
                    self.dtype,
                    src.elem_count(),
                )
            }
            if try_quantize_native_float16(self.dtype, &src, buffer).map_err(crate::Error::from)? {
                return Ok(());
            }
        }
        let op = self.op_with_input("quantize", src)?;
        let src = kernels::quantized::call_quantize(op, || src.to_cpu_storage())?;
        self.quantize_onto(&src)
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &RocmStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        let op = self.op_with_input("quantize_imatrix", src)?;
        let src = kernels::quantized::call_quantize_imatrix(op, || src.to_cpu_storage())?;
        self.quantize_imatrix_onto(&src, imatrix_weights, n_per_row)
    }

    pub fn quantize_onto(&mut self, src: &CpuStorage) -> Result<()> {
        if let QRocmStorageData::NativeFloat16 { buffer, elem_count } = &mut self.storage {
            let bytes = cpu_storage_to_native_float16_bytes(self.dtype, src, *elem_count)?;
            *buffer = self.device.kernel_device().copy_from_host(&bytes)?;
            return Ok(());
        }
        let device = self.device.clone();
        let dtype = self.dtype;
        let QRocmStorageData::Raw { buffer, elem_count } = &mut self.storage else {
            crate::bail!(
                "native {:?} quantize_onto is unavailable without HIP",
                self.dtype
            )
        };
        let op = kernels::quantized::QuantizedOp {
            name: "quantize_onto",
            device: Some(device.kernel_device()),
            input: None,
        };
        kernels::quantized::call_quantize_onto(op, || {
            let bytes =
                cpu_storage_to_raw_quantized_bytes(dtype, src, *elem_count, None, "quantize_onto")?;
            *buffer = kernels::quantized::load_quantized(device.kernel_device(), &bytes)?;
            Ok(())
        })
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        src: &CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        let device = self.device.clone();
        let dtype = self.dtype;
        let QRocmStorageData::Raw { buffer, elem_count } = &mut self.storage else {
            crate::bail!(
                "native {:?} quantize_imatrix_onto is unavailable without HIP",
                self.dtype
            )
        };
        let op = kernels::quantized::QuantizedOp {
            name: "quantize_imatrix_onto",
            device: Some(device.kernel_device()),
            input: None,
        };
        kernels::quantized::call_quantize_imatrix_onto(op, || {
            let bytes = cpu_storage_to_raw_quantized_bytes(
                dtype,
                src,
                *elem_count,
                Some((imatrix_weights, n_per_row)),
                "quantize_imatrix_onto",
            )?;
            *buffer = kernels::quantized::load_quantized(device.kernel_device(), &bytes)?;
            Ok(())
        })
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        match &self.storage {
            QRocmStorageData::Raw { buffer, .. } => buffer.size_in_bytes(),
            QRocmStorageData::NativeFloat16 { buffer, .. } => buffer.size_in_bytes(),
        }
    }

    pub fn fwd_cpu(
        &self,
        self_shape: &Shape,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        let (n, k) = self_shape.dims2()?;
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {self_shape:?}")
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let op = self.op("matmul_t")?;
        let dst_storage = kernels::quantized::call_matmul_t(op, || {
            let qstorage = self.raw_cpu_storage(self_shape.elem_count(), "matmul_t")?;
            match storage.dtype() {
                DType::F32 => {
                    let slice = storage.as_slice::<f32>()?;
                    let slice = &slice
                        [layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
                    let mut dst_storage = vec![0f32; dst_shape.elem_count()];
                    qstorage.matmul_t(
                        (dst_shape.elem_count() / n, k, n),
                        slice,
                        &mut dst_storage,
                    )?;
                    Ok(CpuStorage::F32(dst_storage))
                }
                DType::F16 => {
                    let slice = storage.as_slice::<f16>()?;
                    let slice = &slice
                        [layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
                    let mut dst_storage = vec![f16::ZERO; dst_shape.elem_count()];
                    qstorage.matmul_t_f16(
                        (dst_shape.elem_count() / n, k, n),
                        slice,
                        &mut dst_storage,
                    )?;
                    Ok(CpuStorage::F16(dst_storage))
                }
                _ => crate::bail!("Expected f32/f16"),
            }
        })?;
        Ok((dst_storage, dst_shape))
    }

    pub fn fwd(
        &self,
        self_shape: &Shape,
        storage: &RocmStorage,
        layout: &Layout,
    ) -> Result<(RocmStorage, Shape)> {
        let src_shape = layout.shape();
        let (n, k) = self_shape.dims2()?;
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {self_shape:?}")
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);

        if let Some(buffer) = self.try_fwd(self_shape, storage, layout, n, k, &dst_shape)? {
            return RocmStorage::from_buffer(
                buffer,
                self.device.clone(),
                DType::F32,
                dst_shape.elem_count(),
                "qmatmul",
            )
            .map(|storage| (storage, dst_shape));
        }

        let storage = storage.to_cpu_storage()?;
        let (storage, shape) = self.fwd_cpu(self_shape, &storage, layout)?;
        self.device
            .storage_from_cpu_storage_owned(storage)
            .map(|storage| (storage, shape))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_gemm_gguf(
        &self,
        self_shape: &Shape,
        input: &RocmStorage,
        input_layout: &Layout,
        topk_weights: Option<(&RocmStorage, &Layout)>,
        sorted_token_ids: &RocmStorage,
        sorted_token_ids_layout: &Layout,
        expert_ids: &RocmStorage,
        expert_ids_layout: &Layout,
        topk: usize,
    ) -> Result<(RocmStorage, Shape)> {
        let (num_experts, size_n, size_k) = self_shape.dims3()?;
        let (input_rows, input_k) = input_layout.shape().dims2()?;
        if input_k != size_k {
            crate::bail!(
                "input and GGUF MoE weights last dim mismatch: input {input_k}, weights {size_k}"
            )
        }
        if input.dtype() != DType::F32 {
            crate::bail!("moe_gemm_gguf only accepts f32 inputs")
        }
        if topk == 0 {
            crate::bail!("moe_gemm_gguf requires topk > 0")
        }
        check_contiguous_zero_offset(input_layout, "moe_gemm_gguf input")?;
        check_contiguous_zero_offset(sorted_token_ids_layout, "moe_gemm_gguf sorted_token_ids")?;
        check_contiguous_zero_offset(expert_ids_layout, "moe_gemm_gguf expert_ids")?;

        let size_m = if let Some((topk_weights, topk_weights_layout)) = topk_weights {
            if topk_weights.dtype() != DType::F32 {
                crate::bail!("moe_gemm_gguf topk_weights must be f32")
            }
            check_contiguous_zero_offset(topk_weights_layout, "moe_gemm_gguf topk_weights")?;
            let topk_weights_elems = topk_weights_layout.shape().elem_count();
            if topk_weights_elems != input_rows {
                crate::bail!(
                    "invalid moe_gemm_gguf topk_weights shape: expected {input_rows} elems, got {topk_weights_elems}"
                )
            }
            input_rows
        } else {
            input_rows
                .checked_mul(topk)
                .ok_or_else(|| crate::Error::Msg("moe_gemm_gguf M dimension overflow".into()))?
        };
        let sorted_elems = sorted_token_ids_layout.shape().elem_count();
        let expert_elems = expert_ids_layout.shape().elem_count();
        if sorted_token_ids.dtype() != DType::U32 || expert_ids.dtype() != DType::U32 {
            crate::bail!("moe_gemm_gguf routing tensors must be u32")
        }
        if sorted_elems != size_m || expert_elems != size_m {
            crate::bail!(
                "invalid moe_gemm_gguf routing shape: expected {size_m} elems, got sorted {sorted_elems}, experts {expert_elems}"
            )
        }

        let QRocmStorageData::Raw {
            buffer,
            elem_count: storage_elem_count,
        } = &self.storage
        else {
            crate::bail!(
                "moe_gemm_gguf requires raw quantized ROCm storage, got {:?}",
                self.dtype
            )
        };
        if *storage_elem_count != self_shape.elem_count() {
            crate::bail!(
                "invalid {:?} moe_gemm_gguf elem count: storage has {storage_elem_count}, requested {}",
                self.dtype,
                self_shape.elem_count()
            )
        }
        let Some(dtype) = kernel_quantized_dtype(self.dtype) else {
            crate::bail!(
                "moe_gemm_gguf does not support {:?} ROCm weights",
                self.dtype
            )
        };

        let input = input.tensor_arg()?;
        let sorted_token_ids = sorted_token_ids.tensor_arg()?;
        let expert_ids = expert_ids.tensor_arg()?;
        let topk_weights = match topk_weights {
            Some((weights, _)) => Some(weights.tensor_arg()?),
            None => None,
        };
        let op = kernels::quantized::MoeGemmGgufOp {
            name: "moe_gemm_gguf",
            device: self.device.kernel_device(),
            weights: buffer,
            input,
            sorted_token_ids,
            expert_ids,
            topk_weights,
            output: kernels::TensorOutput::new(kernels::KernelDType::F32, size_m * size_n),
            num_experts,
            topk,
            size_m,
            size_n,
            size_k,
        };
        let Some(buffer) =
            kernels::quantized::try_moe_gemm_gguf(dtype, &op).map_err(crate::Error::from)?
        else {
            crate::bail!(
                "moe_gemm_gguf does not support {:?} ROCm weights",
                self.dtype
            )
        };
        let out_shape = Shape::from((size_m, size_n));
        RocmStorage::from_buffer(
            buffer,
            self.device.clone(),
            DType::F32,
            size_m * size_n,
            "moe_gemm_gguf",
        )
        .map(|storage| (storage, out_shape))
    }

    pub fn data(&self) -> Result<Cow<'_, [u8]>> {
        let op = self.op("data")?;
        kernels::quantized::call_data(op, || Ok(Cow::Owned(self.raw_data()?)))
    }

    fn op(&self, name: &'static str) -> Result<kernels::quantized::QuantizedOp<'_>> {
        Ok(kernels::quantized::QuantizedOp {
            name,
            device: Some(self.device.kernel_device()),
            input: None,
        })
    }

    fn op_with_input(
        &self,
        name: &'static str,
        src: &RocmStorage,
    ) -> Result<kernels::quantized::QuantizedOp<'_>> {
        Ok(kernels::quantized::QuantizedOp {
            name,
            device: Some(self.device.kernel_device()),
            input: Some(src.tensor_arg()?),
        })
    }

    fn raw_data(&self) -> Result<Vec<u8>> {
        let buffer = match &self.storage {
            QRocmStorageData::Raw { buffer, .. } => buffer,
            QRocmStorageData::NativeFloat16 { buffer, .. } => buffer,
        };
        Ok(kernels::quantized::data(
            self.device.kernel_device(),
            buffer,
        )?)
    }

    fn try_fwd(
        &self,
        self_shape: &Shape,
        storage: &RocmStorage,
        layout: &Layout,
        nrows: usize,
        ncols: usize,
        dst_shape: &Shape,
    ) -> Result<Option<kernels::Buffer>> {
        let QRocmStorageData::Raw {
            buffer,
            elem_count: storage_elem_count,
        } = &self.storage
        else {
            return Ok(None);
        };
        if *storage_elem_count != self_shape.elem_count() {
            crate::bail!(
                "invalid {:?} matmul_t elem count: storage has {storage_elem_count}, requested {}",
                self.dtype,
                self_shape.elem_count()
            )
        }
        let Some(dtype) = kernel_quantized_dtype(self.dtype) else {
            return Ok(None);
        };
        if storage.dtype() != DType::F32 {
            return Ok(None);
        }
        let rhs = storage.tensor_arg()?;
        let rhs_layout = kernels::LayoutArg::new(
            layout.shape().dims().to_vec(),
            layout.stride().to_vec(),
            layout.start_offset(),
        )
        .map_err(crate::Error::from)?;
        let op = kernels::quantized::MatMulOp {
            name: "qmatmul",
            device: self.device.kernel_device(),
            weights: buffer,
            rhs,
            rhs_layout,
            output: kernels::TensorOutput::new(kernels::KernelDType::F32, dst_shape.elem_count()),
            batch_size: dst_shape.elem_count() / nrows,
            nrows,
            ncols,
        };
        kernels::quantized::try_matmul_t(dtype, &op).map_err(crate::Error::from)
    }

    fn raw_cpu_storage(
        &self,
        elem_count: usize,
        op: &'static str,
    ) -> Result<Box<dyn QuantizedType>> {
        match &self.storage {
            QRocmStorageData::Raw {
                buffer,
                elem_count: storage_elem_count,
            } => {
                if elem_count != *storage_elem_count {
                    crate::bail!(
                        "invalid {:?} {op} elem count: storage has {storage_elem_count}, requested {elem_count}",
                        self.dtype
                    )
                }
                let data = kernels::quantized::data(self.device.kernel_device(), buffer)?;
                qstorage_from_raw_bytes(self.dtype, &data)
            }
            QRocmStorageData::NativeFloat16 { .. } => {
                crate::bail!(
                    "native {:?} quantized storage should be dequantized before {op}",
                    self.dtype
                )
            }
        }
    }
}

fn kernel_quantized_dtype(dtype: GgmlDType) -> Option<kernels::quantized::QuantizedDType> {
    match dtype {
        GgmlDType::Q5_0 => Some(kernels::quantized::QuantizedDType::Q5_0),
        GgmlDType::Q8_0 => Some(kernels::quantized::QuantizedDType::Q8_0),
        GgmlDType::Q4K => Some(kernels::quantized::QuantizedDType::Q4K),
        GgmlDType::Q6K => Some(kernels::quantized::QuantizedDType::Q6K),
        _ => None,
    }
}

fn check_contiguous_zero_offset(layout: &Layout, op: &'static str) -> Result<()> {
    if !layout.is_contiguous() || layout.start_offset() != 0 {
        crate::bail!("{op} must be contiguous with zero start offset")
    }
    Ok(())
}

fn supports_native_float16(dtype: GgmlDType) -> bool {
    match dtype {
        GgmlDType::F16 => kernels::quantized::supports_native_f16(),
        GgmlDType::BF16 => kernels::quantized::supports_native_bf16(),
        _ => false,
    }
}

fn native_float16_op_name(op: &'static str, dtype: GgmlDType) -> &'static str {
    match (op, dtype) {
        ("dequantize", GgmlDType::F16) => "dequantize_f16",
        ("dequantize", GgmlDType::BF16) => "dequantize_bf16",
        _ => op,
    }
}

fn try_quantize_native_float16(
    dtype: GgmlDType,
    src: &kernels::TensorArg,
    dst: &kernels::Buffer,
) -> kernels::Result<bool> {
    match dtype {
        GgmlDType::F16 => kernels::quantized::try_quantize_f16(src, dst),
        GgmlDType::BF16 => kernels::quantized::try_quantize_bf16(src, dst),
        _ => Ok(false),
    }
}

fn try_dequantize_native_float16_to_f32(
    dtype: GgmlDType,
    device: &kernels::Device,
    src: &kernels::Buffer,
    elem_count: usize,
) -> kernels::Result<Option<kernels::Buffer>> {
    match dtype {
        GgmlDType::F16 => kernels::quantized::try_dequantize_f16_to_f32(device, src, elem_count),
        GgmlDType::BF16 => kernels::quantized::try_dequantize_bf16_to_f32(device, src, elem_count),
        _ => Ok(None),
    }
}

fn cpu_storage_to_native_float16_bytes(
    dtype: GgmlDType,
    src: &CpuStorage,
    elem_count: usize,
) -> Result<Vec<u8>> {
    match (dtype, src) {
        (GgmlDType::F16, CpuStorage::F32(values)) => {
            check_quantized_elem_count(dtype, elem_count, values.len(), "quantize_onto")?;
            let mut values_f16 = vec![f16::ZERO; values.len()];
            values_f16.convert_from_f32_slice(values);
            Ok(slice_to_bytes(&values_f16).to_vec())
        }
        (GgmlDType::F16, CpuStorage::F16(values)) => {
            check_quantized_elem_count(dtype, elem_count, values.len(), "quantize_onto")?;
            Ok(slice_to_bytes(values).to_vec())
        }
        (GgmlDType::BF16, CpuStorage::F32(values)) => {
            check_quantized_elem_count(dtype, elem_count, values.len(), "quantize_onto")?;
            let mut values_bf16 = vec![bf16::ZERO; values.len()];
            values_bf16.convert_from_f32_slice(values);
            Ok(slice_to_bytes(&values_bf16).to_vec())
        }
        (GgmlDType::BF16, CpuStorage::BF16(values)) => {
            check_quantized_elem_count(dtype, elem_count, values.len(), "quantize_onto")?;
            Ok(slice_to_bytes(values).to_vec())
        }
        (GgmlDType::F16, _) => crate::bail!("only f32/f16 can be quantized into native f16"),
        (GgmlDType::BF16, _) => crate::bail!("only f32/bf16 can be quantized into native bf16"),
        _ => crate::bail!("native storage is only supported for f16/bf16, got {dtype:?}"),
    }
}

fn quantized_storage_size_in_bytes(dtype: GgmlDType, elem_count: usize) -> Result<usize> {
    let block_size = dtype.block_size();
    if !elem_count.is_multiple_of(block_size) {
        crate::bail!(
            "invalid {dtype:?} storage size: element count {elem_count} is not divisible by block size {block_size}"
        )
    }
    Ok(elem_count / block_size * dtype.type_size())
}

fn qstorage_from_raw_bytes(dtype: GgmlDType, data: &[u8]) -> Result<Box<dyn QuantizedType>> {
    match dtype {
        GgmlDType::F32 => raw_data_to_cpu_storage::<f32>(dtype, data),
        GgmlDType::F16 => raw_data_to_cpu_storage::<f16>(dtype, data),
        GgmlDType::BF16 => raw_data_to_cpu_storage::<bf16>(dtype, data),
        GgmlDType::Q4_0 => raw_data_to_cpu_storage::<k_quants::BlockQ4_0>(dtype, data),
        GgmlDType::Q4_1 => raw_data_to_cpu_storage::<k_quants::BlockQ4_1>(dtype, data),
        GgmlDType::Q5_0 => raw_data_to_cpu_storage::<k_quants::BlockQ5_0>(dtype, data),
        GgmlDType::Q5_1 => raw_data_to_cpu_storage::<k_quants::BlockQ5_1>(dtype, data),
        GgmlDType::Q8_0 => raw_data_to_cpu_storage::<k_quants::BlockQ8_0>(dtype, data),
        GgmlDType::Q8_1 => raw_data_to_cpu_storage::<k_quants::BlockQ8_1>(dtype, data),
        GgmlDType::Q2K => raw_data_to_cpu_storage::<k_quants::BlockQ2K>(dtype, data),
        GgmlDType::Q3K => raw_data_to_cpu_storage::<k_quants::BlockQ3K>(dtype, data),
        GgmlDType::Q4K => raw_data_to_cpu_storage::<k_quants::BlockQ4K>(dtype, data),
        GgmlDType::Q5K => raw_data_to_cpu_storage::<k_quants::BlockQ5K>(dtype, data),
        GgmlDType::Q6K => raw_data_to_cpu_storage::<k_quants::BlockQ6K>(dtype, data),
        GgmlDType::Q8K => raw_data_to_cpu_storage::<k_quants::BlockQ8K>(dtype, data),
    }
}

fn raw_data_to_cpu_storage<T: super::GgmlType + Send + Sync + 'static>(
    dtype: GgmlDType,
    data: &[u8],
) -> Result<Box<dyn QuantizedType>> {
    let size = mem::size_of::<T>();
    if data.len() % size != 0 {
        crate::bail!(
            "invalid {dtype:?} raw storage size: {} is not divisible by {size}",
            data.len()
        )
    }
    let mut storage = vec![T::zeros(); data.len() / size];
    let dst =
        unsafe { std::slice::from_raw_parts_mut(storage.as_mut_ptr().cast::<u8>(), data.len()) };
    dst.copy_from_slice(data);
    Ok(Box::new(storage))
}

fn cpu_storage_to_raw_quantized_bytes(
    dtype: GgmlDType,
    src: &CpuStorage,
    elem_count: usize,
    imatrix: Option<(&[f32], usize)>,
    op: &'static str,
) -> Result<Vec<u8>> {
    if matches!(dtype, GgmlDType::F16 | GgmlDType::BF16) {
        if imatrix.is_some() {
            crate::bail!("{dtype:?} imatrix quantize is not supported")
        }
        return cpu_storage_to_native_float16_bytes(dtype, src, elem_count);
    }
    let values = src.as_slice::<f32>()?;
    check_quantized_elem_count(dtype, elem_count, values.len(), op)?;
    let mut storage = dtype.cpu_zeros(elem_count);
    match imatrix {
        None => storage.from_float(values),
        Some((imatrix_weights, n_per_row)) => {
            storage.from_float_imatrix(values, imatrix_weights, n_per_row)
        }
    }
    let data_ptr = storage.as_ptr();
    let size_in_bytes = storage.storage_size_in_bytes();
    let data = unsafe { std::slice::from_raw_parts(data_ptr, size_in_bytes) };
    Ok(data.to_vec())
}

fn check_quantized_elem_count(
    dtype: GgmlDType,
    storage_elem_count: usize,
    input_elem_count: usize,
    op: &'static str,
) -> Result<()> {
    if input_elem_count != storage_elem_count {
        crate::bail!(
            "invalid {:?} {op} elem count: storage has {storage_elem_count}, input has {input_elem_count}",
            dtype
        )
    }
    Ok(())
}

fn slice_to_bytes<T>(data: &[T]) -> &[u8] {
    let len = std::mem::size_of_val(data);
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, len) }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &RocmDevice,
    data: &[T],
) -> Result<QStorage> {
    let bytes = slice_to_bytes(data);
    let op = kernels::quantized::QuantizedOp {
        name: "load_quantized",
        device: Some(device.kernel_device()),
        input: None,
    };
    if supports_native_float16(T::DTYPE) {
        let buffer = kernels::quantized::call_load_quantized(op, || {
            kernels::quantized::load_quantized(device.kernel_device(), bytes)
                .map_err(crate::Error::from)
        })?;
        return Ok(QStorage::Rocm(QRocmStorage {
            dtype: T::DTYPE,
            device: device.clone(),
            storage: QRocmStorageData::NativeFloat16 {
                buffer,
                elem_count: data.len(),
            },
        }));
    }
    let buffer = kernels::quantized::call_load_quantized(op, || {
        kernels::quantized::load_quantized(device.kernel_device(), bytes)
            .map_err(crate::Error::from)
    })?;
    Ok(QStorage::Rocm(QRocmStorage {
        dtype: T::DTYPE,
        device: device.clone(),
        storage: QRocmStorageData::Raw {
            buffer,
            elem_count: data.len() * T::BLCK_SIZE,
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::{super::GgmlType, *};

    #[test]
    fn load_quantized_non_native_uses_raw_buffer() -> Result<()> {
        let device = crate::Device::new_rocm(0)?;
        let device = device.as_rocm_device()?;
        let blocks = vec![k_quants::BlockQ4_0::zeros(); 2];
        let storage = load_quantized(device, &blocks)?;
        let QStorage::Rocm(storage) = storage else {
            unreachable!("load_quantized on a ROCm device returns ROCm storage")
        };

        assert!(matches!(&storage.storage, QRocmStorageData::Raw { .. }));
        assert_eq!(
            storage.storage_size_in_bytes(),
            std::mem::size_of_val(blocks.as_slice())
        );
        assert_eq!(
            storage.data()?.len(),
            std::mem::size_of_val(blocks.as_slice())
        );
        Ok(())
    }
}
