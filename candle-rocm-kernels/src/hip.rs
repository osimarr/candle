use crate::{Buffer, LayoutArg, Result, RocmError};
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

unsafe extern "C" {
    fn hip_last_error() -> *const c_char;
    fn hip_device_count(count: *mut c_int) -> c_int;
    fn hip_set_device(ordinal: c_int) -> c_int;
    fn hip_malloc(ordinal: c_int, bytes: usize, ptr: *mut *mut c_void) -> c_int;
    fn hip_free(ordinal: c_int, ptr: *mut c_void) -> c_int;
    fn hip_memset(ordinal: c_int, ptr: *mut c_void, value: c_int, bytes: usize) -> c_int;
    fn hip_copy_h2d(ordinal: c_int, dst: *mut c_void, src: *const u8, bytes: usize) -> c_int;
    fn hip_copy_d2h(ordinal: c_int, src: *const c_void, dst: *mut u8, bytes: usize) -> c_int;
    fn hip_copy_d2d(
        dst_ordinal: c_int,
        dst: *mut c_void,
        src_ordinal: c_int,
        src: *const c_void,
        bytes: usize,
    ) -> c_int;
    fn hip_synchronize(ordinal: c_int) -> c_int;
    fn hip_unary_f32(
        ordinal: c_int,
        op: c_int,
        src: *const f32,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut f32,
        elem_count: usize,
    ) -> c_int;
    fn hip_binary_f32(
        ordinal: c_int,
        op: c_int,
        lhs: *const f32,
        lhs_dims: *const usize,
        lhs_strides: *const usize,
        lhs_rank: usize,
        lhs_start_offset: usize,
        rhs: *const f32,
        rhs_dims: *const usize,
        rhs_strides: *const usize,
        rhs_rank: usize,
        rhs_start_offset: usize,
        dst: *mut f32,
        elem_count: usize,
    ) -> c_int;
    fn hip_cmp_f32(
        ordinal: c_int,
        op: c_int,
        lhs: *const f32,
        lhs_dims: *const usize,
        lhs_strides: *const usize,
        lhs_rank: usize,
        lhs_start_offset: usize,
        rhs: *const f32,
        rhs_dims: *const usize,
        rhs_strides: *const usize,
        rhs_rank: usize,
        rhs_start_offset: usize,
        dst: *mut u8,
        elem_count: usize,
    ) -> c_int;
    fn hip_const_set_f32(
        ordinal: c_int,
        dst: *mut f32,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        value: f32,
        elem_count: usize,
    ) -> c_int;
    fn hip_const_set_u8(
        ordinal: c_int,
        dst: *mut u8,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        value: u8,
        elem_count: usize,
    ) -> c_int;
    fn hip_const_set_u32(
        ordinal: c_int,
        dst: *mut u32,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        value: u32,
        elem_count: usize,
    ) -> c_int;
    fn hip_copy_strided_src(
        ordinal: c_int,
        src: *const u8,
        src_dims: *const usize,
        src_strides: *const usize,
        src_rank: usize,
        src_start_offset: usize,
        dst: *mut u8,
        dst_offset: usize,
        elem_size: usize,
        elem_count: usize,
    ) -> c_int;
    fn hip_copy2d(
        ordinal: c_int,
        src: *const u8,
        dst: *mut u8,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
        elem_size: usize,
    ) -> c_int;
}

#[derive(Debug)]
pub(crate) struct DeviceMemory {
    ordinal: usize,
    bytes: usize,
    ptr: *mut c_void,
}

unsafe impl Send for DeviceMemory {}
unsafe impl Sync for DeviceMemory {}

impl DeviceMemory {
    pub(crate) fn allocate(ordinal: usize, bytes: usize) -> Result<Self> {
        let mut ptr = ptr::null_mut();
        check(
            unsafe { hip_malloc(ordinal_to_c_int(ordinal)?, bytes, &mut ptr) },
            "hipMalloc",
        )?;
        Ok(Self {
            ordinal,
            bytes,
            ptr,
        })
    }

    pub(crate) fn ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn ordinal(&self) -> usize {
        self.ordinal
    }
}

impl Drop for DeviceMemory {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        let _ = unsafe { hip_free(self.ordinal as c_int, self.ptr) };
    }
}

pub(crate) fn device_count() -> Result<usize> {
    let mut count = 0;
    check(unsafe { hip_device_count(&mut count) }, "hipGetDeviceCount")?;
    Ok(count as usize)
}

pub(crate) fn set_device(ordinal: usize) -> Result<()> {
    check(
        unsafe { hip_set_device(ordinal_to_c_int(ordinal)?) },
        "hipSetDevice",
    )
}

pub(crate) fn memset(memory: &DeviceMemory, value: i32) -> Result<()> {
    check(
        unsafe {
            hip_memset(
                ordinal_to_c_int(memory.ordinal())?,
                memory.ptr(),
                value,
                memory.bytes(),
            )
        },
        "hipMemset",
    )
}

pub(crate) fn copy_h2d(memory: &DeviceMemory, src: &[u8]) -> Result<()> {
    if src.len() > memory.bytes() {
        return Err(RocmError::BufferOutOfBounds {
            buffer_bytes: memory.bytes(),
            offset: 0,
            requested: src.len(),
        });
    }
    check(
        unsafe {
            hip_copy_h2d(
                ordinal_to_c_int(memory.ordinal())?,
                memory.ptr(),
                src.as_ptr(),
                src.len(),
            )
        },
        "hipMemcpyHostToDevice",
    )
}

pub(crate) fn copy_d2h(memory: &DeviceMemory, dst: &mut [u8]) -> Result<()> {
    if dst.len() > memory.bytes() {
        return Err(RocmError::BufferOutOfBounds {
            buffer_bytes: memory.bytes(),
            offset: 0,
            requested: dst.len(),
        });
    }
    check(
        unsafe {
            hip_copy_d2h(
                ordinal_to_c_int(memory.ordinal())?,
                memory.ptr(),
                dst.as_mut_ptr(),
                dst.len(),
            )
        },
        "hipMemcpyDeviceToHost",
    )
}

pub(crate) fn copy_d2d(src: &Buffer, dst: &Buffer, bytes: usize) -> Result<()> {
    if bytes > src.size_in_bytes() {
        return Err(RocmError::BufferOutOfBounds {
            buffer_bytes: src.size_in_bytes(),
            offset: 0,
            requested: bytes,
        });
    }
    if bytes > dst.size_in_bytes() {
        return Err(RocmError::BufferOutOfBounds {
            buffer_bytes: dst.size_in_bytes(),
            offset: 0,
            requested: bytes,
        });
    }
    check(
        unsafe {
            hip_copy_d2d(
                ordinal_to_c_int(dst.device_ordinal())?,
                dst.device_ptr(),
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr(),
                bytes,
            )
        },
        "hipMemcpyDeviceToDevice",
    )
}

pub(crate) fn synchronize(ordinal: usize) -> Result<()> {
    check(
        unsafe { hip_synchronize(ordinal_to_c_int(ordinal)?) },
        "hipDeviceSynchronize",
    )
}

pub(crate) fn unary_f32(op: i32, src: &Buffer, layout: &LayoutArg, dst: &Buffer) -> Result<()> {
    check(
        unsafe {
            hip_unary_f32(
                ordinal_to_c_int(src.device_ordinal())?,
                op,
                src.device_ptr().cast::<f32>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<f32>(),
                layout.elem_count(),
            )
        },
        "unary_f32",
    )
}

pub(crate) fn binary_f32(
    op: i32,
    lhs: &Buffer,
    lhs_layout: &LayoutArg,
    rhs: &Buffer,
    rhs_layout: &LayoutArg,
    dst: &Buffer,
) -> Result<()> {
    check(
        unsafe {
            hip_binary_f32(
                ordinal_to_c_int(lhs.device_ordinal())?,
                op,
                lhs.device_ptr().cast::<f32>(),
                lhs_layout.dims().as_ptr(),
                lhs_layout.stride().as_ptr(),
                lhs_layout.dims().len(),
                lhs_layout.start_offset(),
                rhs.device_ptr().cast::<f32>(),
                rhs_layout.dims().as_ptr(),
                rhs_layout.stride().as_ptr(),
                rhs_layout.dims().len(),
                rhs_layout.start_offset(),
                dst.device_ptr().cast::<f32>(),
                lhs_layout.elem_count(),
            )
        },
        "binary_f32",
    )
}

pub(crate) fn cmp_f32(
    op: i32,
    lhs: &Buffer,
    lhs_layout: &LayoutArg,
    rhs: &Buffer,
    rhs_layout: &LayoutArg,
    dst: &Buffer,
) -> Result<()> {
    check(
        unsafe {
            hip_cmp_f32(
                ordinal_to_c_int(lhs.device_ordinal())?,
                op,
                lhs.device_ptr().cast::<f32>(),
                lhs_layout.dims().as_ptr(),
                lhs_layout.stride().as_ptr(),
                lhs_layout.dims().len(),
                lhs_layout.start_offset(),
                rhs.device_ptr().cast::<f32>(),
                rhs_layout.dims().as_ptr(),
                rhs_layout.stride().as_ptr(),
                rhs_layout.dims().len(),
                rhs_layout.start_offset(),
                dst.device_ptr().cast::<u8>(),
                lhs_layout.elem_count(),
            )
        },
        "cmp_f32",
    )
}

pub(crate) fn const_set_f32(dst: &Buffer, layout: &LayoutArg, value: f32) -> Result<()> {
    check(
        unsafe {
            hip_const_set_f32(
                ordinal_to_c_int(dst.device_ordinal())?,
                dst.device_ptr().cast::<f32>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                value,
                layout.elem_count(),
            )
        },
        "const_set_f32",
    )
}

pub(crate) fn const_set_u8(dst: &Buffer, layout: &LayoutArg, value: u8) -> Result<()> {
    check(
        unsafe {
            hip_const_set_u8(
                ordinal_to_c_int(dst.device_ordinal())?,
                dst.device_ptr().cast::<u8>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                value,
                layout.elem_count(),
            )
        },
        "const_set_u8",
    )
}

pub(crate) fn const_set_u32(dst: &Buffer, layout: &LayoutArg, value: u32) -> Result<()> {
    check(
        unsafe {
            hip_const_set_u32(
                ordinal_to_c_int(dst.device_ordinal())?,
                dst.device_ptr().cast::<u32>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                value,
                layout.elem_count(),
            )
        },
        "const_set_u32",
    )
}

pub(crate) fn copy_strided_src(
    src: &Buffer,
    src_layout: &LayoutArg,
    dst: &Buffer,
    dst_offset: usize,
    elem_size: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_copy_strided_src(
                ordinal_to_c_int(dst.device_ordinal())?,
                src.device_ptr().cast::<u8>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                dst.device_ptr().cast::<u8>(),
                dst_offset,
                elem_size,
                src_layout.elem_count(),
            )
        },
        "copy_strided_src",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn copy2d(
    src: &Buffer,
    dst: &Buffer,
    d1: usize,
    d2: usize,
    src_stride1: usize,
    dst_stride1: usize,
    src_offset: usize,
    dst_offset: usize,
    elem_size: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_copy2d(
                ordinal_to_c_int(dst.device_ordinal())?,
                src.device_ptr().cast::<u8>(),
                dst.device_ptr().cast::<u8>(),
                d1,
                d2,
                src_stride1,
                dst_stride1,
                src_offset,
                dst_offset,
                elem_size,
            )
        },
        "copy2d",
    )
}

fn ordinal_to_c_int(ordinal: usize) -> Result<c_int> {
    c_int::try_from(ordinal)
        .map_err(|_| RocmError::Runtime(format!("invalid device ordinal {ordinal}")))
}

fn check(code: c_int, op: &'static str) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    Err(RocmError::Runtime(format!("{op}: {}", last_error())))
}

fn last_error() -> String {
    let ptr = unsafe { hip_last_error() };
    if ptr.is_null() {
        return "unknown HIP error".to_string();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}
