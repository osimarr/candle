//! Dummy ROCm backend.
//!
//! This backend advertises itself as ROCm to Candle while storing data in CPU
//! memory and dispatching operations through the CPU backend. It is intended as
//! a bring-up shim and intentionally supports `f32` tensors only.

use candle_rocm_kernels as kernels;

use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{
    CpuStorage, CustomOp1, CustomOp2, CustomOp3, DType, Error, InplaceOp1, InplaceOp2, InplaceOp3,
    Layout, Result, Shape,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RocmDevice {
    gpu_id: usize,
}

#[derive(Debug, Clone)]
pub struct RocmStorage {
    storage: CpuStorage,
    device: RocmDevice,
}

impl RocmStorage {
    fn wrap(storage: CpuStorage, device: RocmDevice, op: &'static str) -> Result<Self> {
        ensure_supported_storage_dtype(storage.dtype(), op)?;
        Ok(Self { storage, device })
    }

    fn wrap_f32(storage: CpuStorage, device: RocmDevice, op: &'static str) -> Result<Self> {
        ensure_f32(storage.dtype(), op)?;
        Ok(Self { storage, device })
    }

    fn to_cpu_storage_ref(&self) -> &CpuStorage {
        &self.storage
    }

    pub fn transfer_to_device(&self, dst: &RocmDevice) -> Result<Self> {
        let storage = kernels::tensor::transfer(|| Ok::<CpuStorage, Error>(self.storage.clone()))?;
        Self::wrap(storage, dst.clone(), "transfer")
    }

    pub(crate) fn apply_op1(&self, l: &Layout, c: &dyn CustomOp1) -> Result<(Self, Shape)> {
        let (storage, shape) =
            kernels::custom::apply_op1(|| c.cpu_fwd(self.to_cpu_storage_ref(), l))?;
        let storage = Self::wrap_f32(storage, self.device.clone(), c.name())?;
        Ok((storage, shape))
    }

    pub(crate) fn apply_op2(
        &self,
        l1: &Layout,
        rhs: &Self,
        l2: &Layout,
        c: &dyn CustomOp2,
    ) -> Result<(Self, Shape)> {
        let (storage, shape) = kernels::custom::apply_op2(|| {
            c.cpu_fwd(self.to_cpu_storage_ref(), l1, rhs.to_cpu_storage_ref(), l2)
        })?;
        let storage = Self::wrap_f32(storage, self.device.clone(), c.name())?;
        Ok((storage, shape))
    }

    pub(crate) fn apply_op3(
        &self,
        l1: &Layout,
        t2: &Self,
        l2: &Layout,
        t3: &Self,
        l3: &Layout,
        c: &dyn CustomOp3,
    ) -> Result<(Self, Shape)> {
        let (storage, shape) = kernels::custom::apply_op3(|| {
            c.cpu_fwd(
                self.to_cpu_storage_ref(),
                l1,
                t2.to_cpu_storage_ref(),
                l2,
                t3.to_cpu_storage_ref(),
                l3,
            )
        })?;
        let storage = Self::wrap_f32(storage, self.device.clone(), c.name())?;
        Ok((storage, shape))
    }

    pub(crate) fn inplace_op1(&mut self, l: &Layout, c: &dyn InplaceOp1) -> Result<()> {
        kernels::custom::inplace_op1(|| c.cpu_fwd(&mut self.storage, l))?;
        ensure_f32(self.storage.dtype(), c.name())
    }

    pub(crate) fn inplace_op2(
        &mut self,
        l1: &Layout,
        rhs: &Self,
        l2: &Layout,
        c: &dyn InplaceOp2,
    ) -> Result<()> {
        kernels::custom::inplace_op2(|| {
            c.cpu_fwd(&mut self.storage, l1, rhs.to_cpu_storage_ref(), l2)
        })?;
        ensure_f32(self.storage.dtype(), c.name())
    }

    pub(crate) fn inplace_op3(
        &mut self,
        l1: &Layout,
        t2: &Self,
        l2: &Layout,
        t3: &Self,
        l3: &Layout,
        c: &dyn InplaceOp3,
    ) -> Result<()> {
        kernels::custom::inplace_op3(|| {
            c.cpu_fwd(
                &mut self.storage,
                l1,
                t2.to_cpu_storage_ref(),
                l2,
                t3.to_cpu_storage_ref(),
                l3,
            )
        })?;
        ensure_f32(self.storage.dtype(), c.name())
    }
}

fn ensure_f32(dtype: DType, op: &'static str) -> Result<()> {
    if dtype == DType::F32 {
        Ok(())
    } else {
        Err(Error::UnsupportedDTypeForOp(dtype, op).bt())
    }
}

fn ensure_supported_storage_dtype(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::F32 | DType::U8 | DType::U32 => Ok(()),
        dtype => Err(Error::UnsupportedDTypeForOp(dtype, op).bt()),
    }
}

impl BackendStorage for RocmStorage {
    type Device = RocmDevice;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        let storage = kernels::tensor::try_clone(|| self.storage.try_clone(layout))?;
        Self::wrap(storage, self.device.clone(), "try_clone")
    }

    fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        kernels::tensor::transfer(|| self.storage.to_cpu_storage())
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let storage = kernels::tensor::affine(|| self.storage.affine(layout, mul, add))?;
        Self::wrap_f32(storage, self.device.clone(), "affine")
    }

    fn powf(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let storage = kernels::tensor::powf(|| self.storage.powf(layout, alpha))?;
        Self::wrap_f32(storage, self.device.clone(), "powf")
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let storage = kernels::tensor::elu(|| self.storage.elu(layout, alpha))?;
        Self::wrap_f32(storage, self.device.clone(), "elu")
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, dims: &[usize]) -> Result<Self> {
        let storage = kernels::tensor::reduce(|| self.storage.reduce_op(op, layout, dims))?;
        Self::wrap_f32(storage, self.device.clone(), op.name())
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let storage = kernels::tensor::cmp(|| self.storage.cmp(op, &rhs.storage, lhs_l, rhs_l))?;
        Self::wrap(storage, self.device.clone(), "cmp")
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        ensure_f32(dtype, "to_dtype")?;
        let storage = kernels::tensor::to_dtype(|| self.storage.to_dtype(layout, dtype))?;
        Self::wrap_f32(storage, self.device.clone(), "to_dtype")
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let storage = kernels::tensor::unary(|| self.storage.unary_impl::<B>(layout))?;
        Self::wrap_f32(storage, self.device.clone(), B::NAME)
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let storage =
            kernels::tensor::binary(|| self.storage.binary_impl::<B>(&rhs.storage, lhs_l, rhs_l))?;
        Self::wrap_f32(storage, self.device.clone(), B::NAME)
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        let storage = kernels::tensor::where_cond(|| {
            self.storage
                .where_cond(layout, &t.storage, t_l, &f.storage, f_l)
        })?;
        Self::wrap_f32(storage, self.device.clone(), "where")
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        let storage =
            kernels::tensor::conv1d(|| self.storage.conv1d(l, &kernel.storage, kernel_l, params))?;
        Self::wrap_f32(storage, self.device.clone(), "conv1d")
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        let storage = kernels::tensor::conv_transpose1d(|| {
            self.storage
                .conv_transpose1d(l, &kernel.storage, kernel_l, params)
        })?;
        Self::wrap_f32(storage, self.device.clone(), "conv_transpose1d")
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        let storage =
            kernels::tensor::conv2d(|| self.storage.conv2d(l, &kernel.storage, kernel_l, params))?;
        Self::wrap_f32(storage, self.device.clone(), "conv2d")
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        let storage = kernels::tensor::conv_transpose2d(|| {
            self.storage
                .conv_transpose2d(l, &kernel.storage, kernel_l, params)
        })?;
        Self::wrap_f32(storage, self.device.clone(), "conv_transpose2d")
    }

    fn avg_pool2d(
        &self,
        l: &Layout,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        let storage = kernels::tensor::avg_pool2d(|| self.storage.avg_pool2d(l, kernel, stride))?;
        Self::wrap_f32(storage, self.device.clone(), "avg_pool2d")
    }

    fn max_pool2d(
        &self,
        l: &Layout,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        let storage = kernels::tensor::max_pool2d(|| self.storage.max_pool2d(l, kernel, stride))?;
        Self::wrap_f32(storage, self.device.clone(), "max_pool2d")
    }

    fn upsample_nearest1d(&self, l: &Layout, size: usize) -> Result<Self> {
        let storage =
            kernels::tensor::upsample_nearest1d(|| self.storage.upsample_nearest1d(l, size))?;
        Self::wrap_f32(storage, self.device.clone(), "upsample_nearest1d")
    }

    fn upsample_nearest2d(&self, l: &Layout, h: usize, w: usize) -> Result<Self> {
        let storage =
            kernels::tensor::upsample_nearest2d(|| self.storage.upsample_nearest2d(l, h, w))?;
        Self::wrap_f32(storage, self.device.clone(), "upsample_nearest2d")
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
        let storage = kernels::tensor::upsample_bilinear2d(|| {
            self.storage
                .upsample_bilinear2d(l, h, w, align_corners, scale_h, scale_w)
        })?;
        Self::wrap_f32(storage, self.device.clone(), "upsample_bilinear2d")
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        let storage = kernels::tensor::gather(|| self.storage.gather(l, &ids.storage, ids_l, dim))?;
        Self::wrap_f32(storage, self.device.clone(), "gather")
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
        kernels::tensor::scatter_set(|| {
            self.storage
                .scatter_set(l, &ids.storage, ids_l, &src.storage, src_l, dim)
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
        kernels::tensor::scatter_add_set(|| {
            self.storage
                .scatter_add_set(l, &ids.storage, ids_l, &src.storage, src_l, dim)
        })
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        let storage = kernels::tensor::index_select(|| {
            self.storage.index_select(&ids.storage, l, ids_l, dim)
        })?;
        Self::wrap_f32(storage, self.device.clone(), "index_select")
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
        let storage = kernels::tensor::index_add(|| {
            self.storage
                .index_add(l, &ids.storage, ids_l, &src.storage, src_l, dim)
        })?;
        Self::wrap_f32(storage, self.device.clone(), "index_add")
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let storage =
            kernels::tensor::matmul(|| self.storage.matmul(&rhs.storage, bmnk, lhs_l, rhs_l))?;
        Self::wrap_f32(storage, self.device.clone(), "matmul")
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        kernels::tensor::copy_strided_src(|| {
            self.storage
                .copy_strided_src(&mut dst.storage, dst_offset, src_l)
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
        kernels::tensor::copy2d(|| {
            self.storage.copy2d(
                &mut dst.storage,
                d1,
                d2,
                src_stride1,
                dst_stride1,
                src_offset,
                dst_offset,
            )
        })
    }

    fn const_set(&mut self, scalar: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        ensure_f32(scalar.dtype(), "const_set")?;
        kernels::tensor::const_set(|| self.storage.const_set(scalar, layout))
    }
}

impl BackendDevice for RocmDevice {
    type Storage = RocmStorage;

    fn new(gpu_id: usize) -> Result<Self> {
        Ok(Self { gpu_id })
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Rocm {
            gpu_id: self.gpu_id,
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.gpu_id == rhs.gpu_id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        ensure_f32(dtype, "zeros")?;
        let storage =
            kernels::device::zeros(|| crate::cpu_backend::CpuDevice.zeros_impl(shape, dtype))?;
        RocmStorage::wrap(storage, self.clone(), "zeros")
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        ensure_supported_storage_dtype(dtype, "alloc_uninit")?;
        let storage = kernels::device::alloc_uninit(|| {
            crate::cpu_backend::CpuDevice.alloc_uninit(shape, dtype)
        })?;
        RocmStorage::wrap(storage, self.clone(), "alloc_uninit")
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data: &[T]) -> Result<Self::Storage> {
        ensure_supported_storage_dtype(T::DTYPE, "storage_from_slice")?;
        let storage = kernels::device::storage_from_slice(|| {
            Ok::<CpuStorage, Error>(T::to_cpu_storage(data))
        })?;
        RocmStorage::wrap(storage, self.clone(), "storage_from_slice")
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        let storage =
            kernels::device::storage_from_cpu_storage(|| Ok::<CpuStorage, Error>(storage.clone()))?;
        RocmStorage::wrap(storage, self.clone(), "storage_from_cpu_storage")
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        let storage =
            kernels::device::storage_from_cpu_storage_owned(|| Ok::<CpuStorage, Error>(storage))?;
        RocmStorage::wrap(storage, self.clone(), "storage_from_cpu_storage_owned")
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<Self::Storage> {
        ensure_f32(dtype, "rand_uniform")?;
        let storage = kernels::device::rand_uniform(|| {
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
        ensure_f32(dtype, "rand_normal")?;
        let storage = kernels::device::rand_normal(|| {
            crate::cpu_backend::CpuDevice.rand_normal(shape, dtype, mean, std)
        })?;
        RocmStorage::wrap(storage, self.clone(), "rand_normal")
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        kernels::device::set_seed(|| crate::cpu_backend::CpuDevice.set_seed(seed))
    }

    fn get_current_seed(&self) -> Result<u64> {
        kernels::device::get_current_seed(|| crate::cpu_backend::CpuDevice.get_current_seed())
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{DType, Device, Tensor};

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
    fn dummy_rocm_rejects_non_f32_tensors() -> crate::Result<()> {
        let device = Device::new_rocm(0)?;
        assert!(Tensor::zeros((2,), DType::F64, &device).is_err());
        assert!(Tensor::zeros((2,), DType::U32, &device).is_err());
        assert!(Tensor::zeros((2,), DType::F32, &device).is_ok());
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
}
