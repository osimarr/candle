//! ROCm kernel dispatch surface for Candle.
//!
//! The current implementation is a CPU fallback shim. `candle-core` calls into
//! this crate for ROCm operations first, and this crate executes the provided
//! fallback closure until real ROCm kernels are implemented behind these entry
//! points.

#[inline]
pub fn cpu_fallback<T, E, F>(_op: &'static str, fallback: F) -> Result<T, E>
where
    F: FnOnce() -> Result<T, E>,
{
    fallback()
}

macro_rules! fallback_fns {
    ($($name:ident),* $(,)?) => {
        $(
            #[inline]
            pub fn $name<T, E, F>(fallback: F) -> Result<T, E>
            where
                F: FnOnce() -> Result<T, E>,
            {
                super::cpu_fallback(stringify!($name), fallback)
            }
        )*
    };
}

pub mod custom {
    fallback_fns!(
        apply_op1,
        apply_op2,
        apply_op3,
        inplace_op1,
        inplace_op2,
        inplace_op3
    );
}

pub mod device {
    fallback_fns!(
        alloc_uninit,
        get_current_seed,
        rand_normal,
        rand_uniform,
        set_seed,
        storage_from_cpu_storage,
        storage_from_cpu_storage_owned,
        storage_from_slice,
        zeros
    );
}

pub mod quantized {
    fallback_fns!(
        data,
        dequantize,
        load_quantized,
        matmul_t,
        quantize,
        quantize_imatrix,
        quantize_imatrix_onto,
        quantize_onto,
        zeros
    );
}

pub mod tensor {
    fallback_fns!(
        affine,
        avg_pool2d,
        binary,
        cmp,
        const_set,
        conv1d,
        conv2d,
        conv_transpose1d,
        conv_transpose2d,
        copy2d,
        copy_strided_src,
        elu,
        gather,
        index_add,
        index_select,
        matmul,
        max_pool2d,
        powf,
        reduce,
        scatter_add_set,
        scatter_set,
        to_dtype,
        transfer,
        try_clone,
        unary,
        upsample_bilinear2d,
        upsample_nearest1d,
        upsample_nearest2d,
        where_cond
    );
}
