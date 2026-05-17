use super::{GgmlDType, QStorage, QuantizedType};
use crate::backend::{BackendDevice, BackendStorage};
use crate::{CpuStorage, DType, Layout, Result, RocmDevice, RocmStorage, Shape};
use candle_rocm_kernels as kernels;
use half::{bf16, f16, slice::HalfFloatSliceExt};
use std::borrow::Cow;

pub struct QRocmStorage {
    dtype: GgmlDType,
    device: RocmDevice,
    storage: QRocmStorageData,
}

enum QRocmStorageData {
    Cpu(Box<dyn QuantizedType>),
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
        let op = kernels::quantized::QuantizedOp {
            name: "zeros",
            device: Some(device.kernel_device()),
            input: None,
        };
        let storage = kernels::quantized::call_zeros(op, || {
            Ok::<Box<dyn QuantizedType>, crate::Error>(dtype.cpu_zeros(elem_count))
        })?;
        Ok(Self {
            dtype,
            device: device.clone(),
            storage: QRocmStorageData::Cpu(storage),
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
        let op = kernels::quantized::QuantizedOp {
            name: "dequantize",
            device: Some(self.device.kernel_device()),
            input: None,
        };
        let storage = kernels::quantized::call_dequantize(op, || match &self.storage {
            QRocmStorageData::Cpu(storage) => storage.dequantize(elem_count),
            QRocmStorageData::NativeFloat16 { .. } => {
                crate::bail!(
                    "native {:?} dequantize is unavailable without HIP",
                    self.dtype
                )
            }
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
        let op = kernels::quantized::QuantizedOp {
            name: "quantize",
            device: Some(self.device.kernel_device()),
            input: Some(src.tensor_arg()?),
        };
        let src = kernels::quantized::call_quantize(op, || src.to_cpu_storage())?;
        self.quantize_onto(&src)
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &RocmStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        let op = kernels::quantized::QuantizedOp {
            name: "quantize_imatrix",
            device: Some(self.device.kernel_device()),
            input: Some(src.tensor_arg()?),
        };
        let src = kernels::quantized::call_quantize_imatrix(op, || src.to_cpu_storage())?;
        self.quantize_imatrix_onto(&src, imatrix_weights, n_per_row)
    }

    pub fn quantize_onto(&mut self, src: &CpuStorage) -> Result<()> {
        if let QRocmStorageData::NativeFloat16 { buffer, elem_count } = &mut self.storage {
            let bytes = cpu_storage_to_native_float16_bytes(self.dtype, src, *elem_count)?;
            *buffer = self.device.kernel_device().copy_from_host(&bytes)?;
            return Ok(());
        }
        let op = kernels::quantized::QuantizedOp {
            name: "quantize_onto",
            device: Some(self.device.kernel_device()),
            input: None,
        };
        kernels::quantized::call_quantize_onto(op, || {
            let QRocmStorageData::Cpu(storage) = &mut self.storage else {
                crate::bail!(
                    "native {:?} quantize_onto is unavailable without HIP",
                    self.dtype
                )
            };
            storage.from_float(src.as_slice::<f32>()?);
            Ok(())
        })
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        src: &CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        let op = kernels::quantized::QuantizedOp {
            name: "quantize_imatrix_onto",
            device: Some(self.device.kernel_device()),
            input: None,
        };
        kernels::quantized::call_quantize_imatrix_onto(op, || {
            let QRocmStorageData::Cpu(storage) = &mut self.storage else {
                crate::bail!("native {:?} imatrix quantize is not supported", self.dtype)
            };
            storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
            Ok(())
        })
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        match &self.storage {
            QRocmStorageData::Cpu(storage) => storage.storage_size_in_bytes(),
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
        let op = kernels::quantized::QuantizedOp {
            name: "matmul_t",
            device: Some(self.device.kernel_device()),
            input: None,
        };
        let dst_storage =
            kernels::quantized::call_matmul_t(op, || match (&self.storage, storage.dtype()) {
                (QRocmStorageData::NativeFloat16 { .. }, _) => {
                    crate::bail!(
                        "native {:?} quantized storage should be dequantized before matmul",
                        self.dtype
                    )
                }
                (QRocmStorageData::Cpu(qstorage), DType::F32) => {
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
                (QRocmStorageData::Cpu(qstorage), DType::F16) => {
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
            })?;
        Ok((dst_storage, dst_shape))
    }

    pub fn data(&self) -> Result<Cow<'_, [u8]>> {
        if let QRocmStorageData::NativeFloat16 { buffer, .. } = &self.storage {
            let mut data = vec![0; buffer.size_in_bytes()];
            self.device
                .kernel_device()
                .copy_to_host(buffer, &mut data)?;
            return Ok(Cow::Owned(data));
        }
        let op = kernels::quantized::QuantizedOp {
            name: "data",
            device: Some(self.device.kernel_device()),
            input: None,
        };
        kernels::quantized::call_data(op, || {
            let QRocmStorageData::Cpu(storage) = &self.storage else {
                crate::bail!("native {:?} data is unavailable without HIP", self.dtype)
            };
            let data_ptr = storage.as_ptr();
            let size_in_bytes = storage.storage_size_in_bytes();
            let data = unsafe { std::slice::from_raw_parts(data_ptr, size_in_bytes) };
            Ok(Cow::from(data))
        })
    }
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
            check_native_float16_len(dtype, elem_count, values.len(), "quantize_onto")?;
            let mut values_f16 = vec![f16::ZERO; values.len()];
            values_f16.convert_from_f32_slice(values);
            Ok(slice_to_bytes(&values_f16).to_vec())
        }
        (GgmlDType::F16, CpuStorage::F16(values)) => {
            check_native_float16_len(dtype, elem_count, values.len(), "quantize_onto")?;
            Ok(slice_to_bytes(values).to_vec())
        }
        (GgmlDType::BF16, CpuStorage::F32(values)) => {
            check_native_float16_len(dtype, elem_count, values.len(), "quantize_onto")?;
            let mut values_bf16 = vec![bf16::ZERO; values.len()];
            values_bf16.convert_from_f32_slice(values);
            Ok(slice_to_bytes(&values_bf16).to_vec())
        }
        (GgmlDType::BF16, CpuStorage::BF16(values)) => {
            check_native_float16_len(dtype, elem_count, values.len(), "quantize_onto")?;
            Ok(slice_to_bytes(values).to_vec())
        }
        (GgmlDType::F16, _) => crate::bail!("only f32/f16 can be quantized into native f16"),
        (GgmlDType::BF16, _) => crate::bail!("only f32/bf16 can be quantized into native bf16"),
        _ => crate::bail!("native storage is only supported for f16/bf16, got {dtype:?}"),
    }
}

fn check_native_float16_len(
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
    if supports_native_float16(T::DTYPE) {
        let buffer = device
            .kernel_device()
            .copy_from_host(slice_to_bytes(data))?;
        return Ok(QStorage::Rocm(QRocmStorage {
            dtype: T::DTYPE,
            device: device.clone(),
            storage: QRocmStorageData::NativeFloat16 {
                buffer,
                elem_count: data.len(),
            },
        }));
    }
    let op = kernels::quantized::QuantizedOp {
        name: "load_quantized",
        device: Some(device.kernel_device()),
        input: None,
    };
    kernels::quantized::call_load_quantized(op, || {
        Ok(QStorage::Rocm(QRocmStorage {
            dtype: T::DTYPE,
            device: device.clone(),
            storage: QRocmStorageData::Cpu(Box::new(data.to_vec())),
        }))
    })
}
