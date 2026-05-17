#![allow(unused)]
use super::GgmlDType;
use crate::{CpuStorage, Error, Layout, Result, RocmDevice, RocmStorage, Shape};
use std::borrow::Cow;

pub struct QRocmStorage {
    dtype: GgmlDType,
    device: RocmDevice,
}

impl QRocmStorage {
    pub fn zeros(_: &RocmDevice, _: usize, _: GgmlDType) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &RocmDevice {
        &self.device
    }

    pub fn dequantize(&self, _elem_count: usize) -> Result<RocmStorage> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn quantize(&mut self, _src: &RocmStorage) -> Result<()> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn quantize_imatrix(
        &mut self,
        _src: &RocmStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        _src: &CpuStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn quantize_onto(&mut self, _src: &CpuStorage) -> Result<()> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        0
    }

    pub fn fwd_cpu(
        &self,
        _self_shape: &Shape,
        _storage: &CpuStorage,
        _layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn data(&self) -> Result<Cow<'_, [u8]>> {
        Err(Error::NotCompiledWithRocmSupport)
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    _device: &RocmDevice,
    _data: &[T],
) -> Result<super::QStorage> {
    Err(Error::NotCompiledWithRocmSupport)
}
