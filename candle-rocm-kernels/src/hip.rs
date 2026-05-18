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
    fn hip_cast_f32_to_bf16(
        ordinal: c_int,
        src: *const f32,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut u16,
        elem_count: usize,
    ) -> c_int;
    fn hip_cast_bf16_to_f32(
        ordinal: c_int,
        src: *const u16,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut f32,
        elem_count: usize,
    ) -> c_int;
    fn hip_cast_f32_to_f16(
        ordinal: c_int,
        src: *const f32,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut u16,
        elem_count: usize,
    ) -> c_int;
    fn hip_cast_f16_to_f32(
        ordinal: c_int,
        src: *const u16,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut f32,
        elem_count: usize,
    ) -> c_int;
    fn hip_cast_f32_to_f8e4m3(
        ordinal: c_int,
        src: *const f32,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut u8,
        elem_count: usize,
    ) -> c_int;
    fn hip_cast_f8e4m3_to_f32(
        ordinal: c_int,
        src: *const u8,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut f32,
        elem_count: usize,
    ) -> c_int;
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
    fn hip_unary_bf16(
        ordinal: c_int,
        op: c_int,
        src: *const u16,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut u16,
        elem_count: usize,
    ) -> c_int;
    fn hip_unary_f8e4m3(
        ordinal: c_int,
        op: c_int,
        src: *const u8,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut u8,
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
    fn hip_binary_bf16(
        ordinal: c_int,
        op: c_int,
        lhs: *const u16,
        lhs_dims: *const usize,
        lhs_strides: *const usize,
        lhs_rank: usize,
        lhs_start_offset: usize,
        rhs: *const u16,
        rhs_dims: *const usize,
        rhs_strides: *const usize,
        rhs_rank: usize,
        rhs_start_offset: usize,
        dst: *mut u16,
        elem_count: usize,
    ) -> c_int;
    fn hip_binary_f8e4m3(
        ordinal: c_int,
        op: c_int,
        lhs: *const u8,
        lhs_dims: *const usize,
        lhs_strides: *const usize,
        lhs_rank: usize,
        lhs_start_offset: usize,
        rhs: *const u8,
        rhs_dims: *const usize,
        rhs_strides: *const usize,
        rhs_rank: usize,
        rhs_start_offset: usize,
        dst: *mut u8,
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
    fn hip_cmp_bf16(
        ordinal: c_int,
        op: c_int,
        lhs: *const u16,
        lhs_dims: *const usize,
        lhs_strides: *const usize,
        lhs_rank: usize,
        lhs_start_offset: usize,
        rhs: *const u16,
        rhs_dims: *const usize,
        rhs_strides: *const usize,
        rhs_rank: usize,
        rhs_start_offset: usize,
        dst: *mut u8,
        elem_count: usize,
    ) -> c_int;
    fn hip_cmp_f8e4m3(
        ordinal: c_int,
        op: c_int,
        lhs: *const u8,
        lhs_dims: *const usize,
        lhs_strides: *const usize,
        lhs_rank: usize,
        lhs_start_offset: usize,
        rhs: *const u8,
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
    fn hip_const_set_i16(
        ordinal: c_int,
        dst: *mut i16,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        value: i16,
        elem_count: usize,
    ) -> c_int;
    fn hip_const_set_i32(
        ordinal: c_int,
        dst: *mut i32,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        value: i32,
        elem_count: usize,
    ) -> c_int;
    fn hip_const_set_i64(
        ordinal: c_int,
        dst: *mut i64,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        value: i64,
        elem_count: usize,
    ) -> c_int;
    fn hip_const_set_bf16(
        ordinal: c_int,
        dst: *mut u16,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        value: u16,
        elem_count: usize,
    ) -> c_int;
    fn hip_const_set_f16(
        ordinal: c_int,
        dst: *mut u16,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        value: u16,
        elem_count: usize,
    ) -> c_int;
    fn hip_const_set_f64(
        ordinal: c_int,
        dst: *mut f64,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        value: f64,
        elem_count: usize,
    ) -> c_int;
    fn hip_const_set_f8e4m3(
        ordinal: c_int,
        dst: *mut u8,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        value: u8,
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
    fn hip_affine_f32(
        ordinal: c_int,
        src: *const f32,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut f32,
        elem_count: usize,
        mul: f32,
        add: f32,
    ) -> c_int;
    fn hip_affine_bf16(
        ordinal: c_int,
        src: *const u16,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut u16,
        elem_count: usize,
        mul: f32,
        add: f32,
    ) -> c_int;
    fn hip_affine_f8e4m3(
        ordinal: c_int,
        src: *const u8,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut u8,
        elem_count: usize,
        mul: f32,
        add: f32,
    ) -> c_int;
    fn hip_powf_f32(
        ordinal: c_int,
        src: *const f32,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut f32,
        elem_count: usize,
        value: f32,
    ) -> c_int;
    fn hip_powf_bf16(
        ordinal: c_int,
        src: *const u16,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut u16,
        elem_count: usize,
        value: f32,
    ) -> c_int;
    fn hip_powf_f8e4m3(
        ordinal: c_int,
        src: *const u8,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut u8,
        elem_count: usize,
        value: f32,
    ) -> c_int;
    fn hip_elu_f32(
        ordinal: c_int,
        src: *const f32,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut f32,
        elem_count: usize,
        alpha: f32,
    ) -> c_int;
    fn hip_elu_bf16(
        ordinal: c_int,
        src: *const u16,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut u16,
        elem_count: usize,
        alpha: f32,
    ) -> c_int;
    fn hip_elu_f8e4m3(
        ordinal: c_int,
        src: *const u8,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut u8,
        elem_count: usize,
        alpha: f32,
    ) -> c_int;
    fn hip_reduce_f32(
        ordinal: c_int,
        op: c_int,
        src: *const f32,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        reduce_mask: u64,
        reduce_count: usize,
        dst: *mut c_void,
        elem_count: usize,
    ) -> c_int;
    fn hip_reduce_bf16(
        ordinal: c_int,
        op: c_int,
        src: *const u16,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        reduce_mask: u64,
        reduce_count: usize,
        dst: *mut c_void,
        elem_count: usize,
    ) -> c_int;
    fn hip_reduce_f8e4m3(
        ordinal: c_int,
        op: c_int,
        src: *const u8,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        reduce_mask: u64,
        reduce_count: usize,
        dst: *mut c_void,
        elem_count: usize,
    ) -> c_int;
    fn hip_random_uniform_f32(
        ordinal: c_int,
        dst: *mut f32,
        elem_count: usize,
        seed: u64,
        lo: f32,
        up: f32,
    ) -> c_int;
    fn hip_random_uniform_bf16(
        ordinal: c_int,
        dst: *mut u16,
        elem_count: usize,
        seed: u64,
        lo: f32,
        up: f32,
    ) -> c_int;
    fn hip_random_uniform_f8e4m3(
        ordinal: c_int,
        dst: *mut u8,
        elem_count: usize,
        seed: u64,
        lo: f32,
        up: f32,
    ) -> c_int;
    fn hip_random_normal_f32(
        ordinal: c_int,
        dst: *mut f32,
        elem_count: usize,
        seed: u64,
        mean: f32,
        std: f32,
    ) -> c_int;
    fn hip_random_normal_bf16(
        ordinal: c_int,
        dst: *mut u16,
        elem_count: usize,
        seed: u64,
        mean: f32,
        std: f32,
    ) -> c_int;
    fn hip_random_normal_f8e4m3(
        ordinal: c_int,
        dst: *mut u8,
        elem_count: usize,
        seed: u64,
        mean: f32,
        std: f32,
    ) -> c_int;
    fn hip_matmul_f32(
        ordinal: c_int,
        lhs: *const f32,
        rhs: *const f32,
        dst: *mut f32,
        b: usize,
        m: usize,
        n: usize,
        k: usize,
        lhs_start_offset: usize,
        rhs_start_offset: usize,
        lhs_batch_stride: usize,
        rhs_batch_stride: usize,
        lhs_row_stride: usize,
        lhs_col_stride: usize,
        rhs_row_stride: usize,
        rhs_col_stride: usize,
    ) -> c_int;
    fn hip_matmul_bf16(
        ordinal: c_int,
        lhs: *const u16,
        rhs: *const u16,
        dst: *mut u16,
        b: usize,
        m: usize,
        n: usize,
        k: usize,
        lhs_start_offset: usize,
        rhs_start_offset: usize,
        lhs_batch_stride: usize,
        rhs_batch_stride: usize,
        lhs_row_stride: usize,
        lhs_col_stride: usize,
        rhs_row_stride: usize,
        rhs_col_stride: usize,
    ) -> c_int;
    fn hip_matmul_f8e4m3(
        ordinal: c_int,
        lhs: *const u8,
        rhs: *const u8,
        dst: *mut u8,
        b: usize,
        m: usize,
        n: usize,
        k: usize,
        lhs_start_offset: usize,
        rhs_start_offset: usize,
        lhs_batch_stride: usize,
        rhs_batch_stride: usize,
        lhs_row_stride: usize,
        lhs_col_stride: usize,
        rhs_row_stride: usize,
        rhs_col_stride: usize,
    ) -> c_int;
    fn hip_qmatmul_t_q5_0_f32(
        ordinal: c_int,
        weights: *const u8,
        rhs: *const f32,
        rhs_dims: *const usize,
        rhs_strides: *const usize,
        rhs_rank: usize,
        rhs_start_offset: usize,
        dst: *mut f32,
        batch_size: usize,
        nrows: usize,
        ncols: usize,
    ) -> c_int;
    fn hip_qmatmul_t_q8_0_f32(
        ordinal: c_int,
        weights: *const u8,
        rhs: *const f32,
        rhs_dims: *const usize,
        rhs_strides: *const usize,
        rhs_rank: usize,
        rhs_start_offset: usize,
        dst: *mut f32,
        batch_size: usize,
        nrows: usize,
        ncols: usize,
    ) -> c_int;
    fn hip_qmatmul_t_q4k_f32(
        ordinal: c_int,
        weights: *const u8,
        rhs: *const f32,
        rhs_dims: *const usize,
        rhs_strides: *const usize,
        rhs_rank: usize,
        rhs_start_offset: usize,
        dst: *mut f32,
        batch_size: usize,
        nrows: usize,
        ncols: usize,
    ) -> c_int;
    fn hip_qmatmul_t_q6k_f32(
        ordinal: c_int,
        weights: *const u8,
        rhs: *const f32,
        rhs_dims: *const usize,
        rhs_strides: *const usize,
        rhs_rank: usize,
        rhs_start_offset: usize,
        dst: *mut f32,
        batch_size: usize,
        nrows: usize,
        ncols: usize,
    ) -> c_int;
    fn hip_moe_gemm_gguf_q8_0_f32(
        ordinal: c_int,
        input: *const f32,
        weights: *const u8,
        sorted_token_ids: *const u32,
        expert_ids: *const u32,
        topk_weights: *const f32,
        dst: *mut f32,
        num_experts: usize,
        topk: usize,
        size_m: usize,
        size_n: usize,
        size_k: usize,
    ) -> c_int;
    fn hip_moe_gemm_gguf_q4k_f32(
        ordinal: c_int,
        input: *const f32,
        weights: *const u8,
        sorted_token_ids: *const u32,
        expert_ids: *const u32,
        topk_weights: *const f32,
        dst: *mut f32,
        num_experts: usize,
        topk: usize,
        size_m: usize,
        size_n: usize,
        size_k: usize,
    ) -> c_int;
    fn hip_moe_gemm_gguf_q6k_f32(
        ordinal: c_int,
        input: *const f32,
        weights: *const u8,
        sorted_token_ids: *const u32,
        expert_ids: *const u32,
        topk_weights: *const f32,
        dst: *mut f32,
        num_experts: usize,
        topk: usize,
        size_m: usize,
        size_n: usize,
        size_k: usize,
    ) -> c_int;
    fn hip_index_select_u32_f32(
        ordinal: c_int,
        src: *const f32,
        src_dims: *const usize,
        src_strides: *const usize,
        rank: usize,
        src_start_offset: usize,
        ids: *const u32,
        ids_start_offset: usize,
        ids_stride: usize,
        dim: usize,
        n_ids: usize,
        dst: *mut f32,
        elem_count: usize,
    ) -> c_int;
    fn hip_index_select_i64_f32(
        ordinal: c_int,
        src: *const f32,
        src_dims: *const usize,
        src_strides: *const usize,
        rank: usize,
        src_start_offset: usize,
        ids: *const i64,
        ids_start_offset: usize,
        ids_stride: usize,
        dim: usize,
        n_ids: usize,
        dst: *mut f32,
        elem_count: usize,
    ) -> c_int;
    fn hip_index_select_u32_bf16(
        ordinal: c_int,
        src: *const u16,
        src_dims: *const usize,
        src_strides: *const usize,
        rank: usize,
        src_start_offset: usize,
        ids: *const u32,
        ids_start_offset: usize,
        ids_stride: usize,
        dim: usize,
        n_ids: usize,
        dst: *mut u16,
        elem_count: usize,
    ) -> c_int;
    fn hip_index_select_i64_bf16(
        ordinal: c_int,
        src: *const u16,
        src_dims: *const usize,
        src_strides: *const usize,
        rank: usize,
        src_start_offset: usize,
        ids: *const i64,
        ids_start_offset: usize,
        ids_stride: usize,
        dim: usize,
        n_ids: usize,
        dst: *mut u16,
        elem_count: usize,
    ) -> c_int;
    fn hip_index_select_u32_f8e4m3(
        ordinal: c_int,
        src: *const u8,
        src_dims: *const usize,
        src_strides: *const usize,
        rank: usize,
        src_start_offset: usize,
        ids: *const u32,
        ids_start_offset: usize,
        ids_stride: usize,
        dim: usize,
        n_ids: usize,
        dst: *mut u8,
        elem_count: usize,
    ) -> c_int;
    fn hip_index_select_i64_f8e4m3(
        ordinal: c_int,
        src: *const u8,
        src_dims: *const usize,
        src_strides: *const usize,
        rank: usize,
        src_start_offset: usize,
        ids: *const i64,
        ids_start_offset: usize,
        ids_stride: usize,
        dim: usize,
        n_ids: usize,
        dst: *mut u8,
        elem_count: usize,
    ) -> c_int;
    fn hip_gather_u32_f32(
        ordinal: c_int,
        src: *const f32,
        src_dims: *const usize,
        src_strides: *const usize,
        src_rank: usize,
        src_start_offset: usize,
        ids: *const u32,
        ids_dims: *const usize,
        ids_strides: *const usize,
        ids_rank: usize,
        ids_start_offset: usize,
        dim: usize,
        dst: *mut f32,
        elem_count: usize,
    ) -> c_int;
    fn hip_gather_u32_bf16(
        ordinal: c_int,
        src: *const u16,
        src_dims: *const usize,
        src_strides: *const usize,
        src_rank: usize,
        src_start_offset: usize,
        ids: *const u32,
        ids_dims: *const usize,
        ids_strides: *const usize,
        ids_rank: usize,
        ids_start_offset: usize,
        dim: usize,
        dst: *mut u16,
        elem_count: usize,
    ) -> c_int;
    fn hip_gather_u32_f8e4m3(
        ordinal: c_int,
        src: *const u8,
        src_dims: *const usize,
        src_strides: *const usize,
        src_rank: usize,
        src_start_offset: usize,
        ids: *const u32,
        ids_dims: *const usize,
        ids_strides: *const usize,
        ids_rank: usize,
        ids_start_offset: usize,
        dim: usize,
        dst: *mut u8,
        elem_count: usize,
    ) -> c_int;
    fn hip_scatter_u32_f32(
        ordinal: c_int,
        add: c_int,
        dst: *mut f32,
        dst_dims: *const usize,
        dst_strides: *const usize,
        dst_rank: usize,
        dst_start_offset: usize,
        ids: *const u32,
        ids_dims: *const usize,
        ids_strides: *const usize,
        ids_rank: usize,
        ids_start_offset: usize,
        src: *const f32,
        src_dims: *const usize,
        src_strides: *const usize,
        src_rank: usize,
        src_start_offset: usize,
        dim: usize,
        src_elem_count: usize,
        dst_elem_count: usize,
    ) -> c_int;
    fn hip_scatter_u32_bf16(
        ordinal: c_int,
        add: c_int,
        dst: *mut u16,
        dst_dims: *const usize,
        dst_strides: *const usize,
        dst_rank: usize,
        dst_start_offset: usize,
        ids: *const u32,
        ids_dims: *const usize,
        ids_strides: *const usize,
        ids_rank: usize,
        ids_start_offset: usize,
        src: *const u16,
        src_dims: *const usize,
        src_strides: *const usize,
        src_rank: usize,
        src_start_offset: usize,
        dim: usize,
        src_elem_count: usize,
        dst_elem_count: usize,
    ) -> c_int;
    fn hip_scatter_u32_f8e4m3(
        ordinal: c_int,
        add: c_int,
        dst: *mut u8,
        dst_dims: *const usize,
        dst_strides: *const usize,
        dst_rank: usize,
        dst_start_offset: usize,
        ids: *const u32,
        ids_dims: *const usize,
        ids_strides: *const usize,
        ids_rank: usize,
        ids_start_offset: usize,
        src: *const u8,
        src_dims: *const usize,
        src_strides: *const usize,
        src_rank: usize,
        src_start_offset: usize,
        dim: usize,
        src_elem_count: usize,
        dst_elem_count: usize,
    ) -> c_int;
    fn hip_index_add_u32_f32(
        ordinal: c_int,
        input: *const f32,
        input_dims: *const usize,
        input_strides: *const usize,
        input_rank: usize,
        input_start_offset: usize,
        ids: *const u32,
        ids_start_offset: usize,
        ids_stride: usize,
        ids_len: usize,
        src: *const f32,
        src_dims: *const usize,
        src_strides: *const usize,
        src_rank: usize,
        src_start_offset: usize,
        dim: usize,
        dst: *mut f32,
        elem_count: usize,
    ) -> c_int;
    fn hip_index_add_u32_bf16(
        ordinal: c_int,
        input: *const u16,
        input_dims: *const usize,
        input_strides: *const usize,
        input_rank: usize,
        input_start_offset: usize,
        ids: *const u32,
        ids_start_offset: usize,
        ids_stride: usize,
        ids_len: usize,
        src: *const u16,
        src_dims: *const usize,
        src_strides: *const usize,
        src_rank: usize,
        src_start_offset: usize,
        dim: usize,
        dst: *mut u16,
        elem_count: usize,
    ) -> c_int;
    fn hip_index_add_u32_f8e4m3(
        ordinal: c_int,
        input: *const u8,
        input_dims: *const usize,
        input_strides: *const usize,
        input_rank: usize,
        input_start_offset: usize,
        ids: *const u32,
        ids_start_offset: usize,
        ids_stride: usize,
        ids_len: usize,
        src: *const u8,
        src_dims: *const usize,
        src_strides: *const usize,
        src_rank: usize,
        src_start_offset: usize,
        dim: usize,
        dst: *mut u8,
        elem_count: usize,
    ) -> c_int;
    fn hip_where_u8_f32(
        ordinal: c_int,
        cond: *const u8,
        cond_dims: *const usize,
        cond_strides: *const usize,
        cond_rank: usize,
        cond_start_offset: usize,
        on_true: *const f32,
        true_dims: *const usize,
        true_strides: *const usize,
        true_rank: usize,
        true_start_offset: usize,
        on_false: *const f32,
        false_dims: *const usize,
        false_strides: *const usize,
        false_rank: usize,
        false_start_offset: usize,
        dst: *mut f32,
        elem_count: usize,
    ) -> c_int;
    fn hip_where_u8_bf16(
        ordinal: c_int,
        cond: *const u8,
        cond_dims: *const usize,
        cond_strides: *const usize,
        cond_rank: usize,
        cond_start_offset: usize,
        on_true: *const u16,
        true_dims: *const usize,
        true_strides: *const usize,
        true_rank: usize,
        true_start_offset: usize,
        on_false: *const u16,
        false_dims: *const usize,
        false_strides: *const usize,
        false_rank: usize,
        false_start_offset: usize,
        dst: *mut u16,
        elem_count: usize,
    ) -> c_int;
    fn hip_where_u8_f8e4m3(
        ordinal: c_int,
        cond: *const u8,
        cond_dims: *const usize,
        cond_strides: *const usize,
        cond_rank: usize,
        cond_start_offset: usize,
        on_true: *const u8,
        true_dims: *const usize,
        true_strides: *const usize,
        true_rank: usize,
        true_start_offset: usize,
        on_false: *const u8,
        false_dims: *const usize,
        false_strides: *const usize,
        false_rank: usize,
        false_start_offset: usize,
        dst: *mut u8,
        elem_count: usize,
    ) -> c_int;
    fn hip_pool2d_f32(
        ordinal: c_int,
        op: c_int,
        src: *const f32,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut f32,
        k_h: usize,
        k_w: usize,
        s_h: usize,
        s_w: usize,
        out_h: usize,
        out_w: usize,
        elem_count: usize,
    ) -> c_int;
    fn hip_pool2d_bf16(
        ordinal: c_int,
        op: c_int,
        src: *const u16,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut u16,
        k_h: usize,
        k_w: usize,
        s_h: usize,
        s_w: usize,
        out_h: usize,
        out_w: usize,
        elem_count: usize,
    ) -> c_int;
    fn hip_pool2d_f8e4m3(
        ordinal: c_int,
        op: c_int,
        src: *const u8,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut u8,
        k_h: usize,
        k_w: usize,
        s_h: usize,
        s_w: usize,
        out_h: usize,
        out_w: usize,
        elem_count: usize,
    ) -> c_int;
    fn hip_upsample_nearest1d_f32(
        ordinal: c_int,
        src: *const f32,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut f32,
        out_size: usize,
        elem_count: usize,
    ) -> c_int;
    fn hip_upsample_nearest1d_bf16(
        ordinal: c_int,
        src: *const u16,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut u16,
        out_size: usize,
        elem_count: usize,
    ) -> c_int;
    fn hip_upsample_nearest1d_f8e4m3(
        ordinal: c_int,
        src: *const u8,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut u8,
        out_size: usize,
        elem_count: usize,
    ) -> c_int;
    fn hip_upsample_nearest2d_f32(
        ordinal: c_int,
        src: *const f32,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut f32,
        out_h: usize,
        out_w: usize,
        elem_count: usize,
    ) -> c_int;
    fn hip_upsample_nearest2d_bf16(
        ordinal: c_int,
        src: *const u16,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut u16,
        out_h: usize,
        out_w: usize,
        elem_count: usize,
    ) -> c_int;
    fn hip_upsample_nearest2d_f8e4m3(
        ordinal: c_int,
        src: *const u8,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut u8,
        out_h: usize,
        out_w: usize,
        elem_count: usize,
    ) -> c_int;
    fn hip_upsample_bilinear2d_f32(
        ordinal: c_int,
        src: *const f32,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut f32,
        out_h: usize,
        out_w: usize,
        scale_h: f64,
        scale_w: f64,
        align_corners: c_int,
        elem_count: usize,
    ) -> c_int;
    fn hip_upsample_bilinear2d_bf16(
        ordinal: c_int,
        src: *const u16,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut u16,
        out_h: usize,
        out_w: usize,
        scale_h: f64,
        scale_w: f64,
        align_corners: c_int,
        elem_count: usize,
    ) -> c_int;
    fn hip_upsample_bilinear2d_f8e4m3(
        ordinal: c_int,
        src: *const u8,
        dims: *const usize,
        strides: *const usize,
        rank: usize,
        start_offset: usize,
        dst: *mut u8,
        out_h: usize,
        out_w: usize,
        scale_h: f64,
        scale_w: f64,
        align_corners: c_int,
        elem_count: usize,
    ) -> c_int;
    fn hip_conv1d_f32(
        ordinal: c_int,
        src: *const f32,
        src_dims: *const usize,
        src_strides: *const usize,
        src_rank: usize,
        src_start_offset: usize,
        kernel: *const f32,
        kernel_dims: *const usize,
        kernel_strides: *const usize,
        kernel_rank: usize,
        kernel_start_offset: usize,
        dst: *mut f32,
        padding: usize,
        stride: usize,
        dilation: usize,
        l_out: usize,
        elem_count: usize,
    ) -> c_int;
    fn hip_conv1d_bf16(
        ordinal: c_int,
        src: *const u16,
        src_dims: *const usize,
        src_strides: *const usize,
        src_rank: usize,
        src_start_offset: usize,
        kernel: *const u16,
        kernel_dims: *const usize,
        kernel_strides: *const usize,
        kernel_rank: usize,
        kernel_start_offset: usize,
        dst: *mut u16,
        padding: usize,
        stride: usize,
        dilation: usize,
        l_out: usize,
        elem_count: usize,
    ) -> c_int;
    fn hip_conv1d_f8e4m3(
        ordinal: c_int,
        src: *const u8,
        src_dims: *const usize,
        src_strides: *const usize,
        src_rank: usize,
        src_start_offset: usize,
        kernel: *const u8,
        kernel_dims: *const usize,
        kernel_strides: *const usize,
        kernel_rank: usize,
        kernel_start_offset: usize,
        dst: *mut u8,
        padding: usize,
        stride: usize,
        dilation: usize,
        l_out: usize,
        elem_count: usize,
    ) -> c_int;
    fn hip_conv_transpose1d_f32(
        ordinal: c_int,
        src: *const f32,
        src_dims: *const usize,
        src_strides: *const usize,
        src_rank: usize,
        src_start_offset: usize,
        kernel: *const f32,
        kernel_dims: *const usize,
        kernel_strides: *const usize,
        kernel_rank: usize,
        kernel_start_offset: usize,
        dst: *mut f32,
        padding: usize,
        stride: usize,
        dilation: usize,
        l_out: usize,
        elem_count: usize,
    ) -> c_int;
    fn hip_conv_transpose1d_bf16(
        ordinal: c_int,
        src: *const u16,
        src_dims: *const usize,
        src_strides: *const usize,
        src_rank: usize,
        src_start_offset: usize,
        kernel: *const u16,
        kernel_dims: *const usize,
        kernel_strides: *const usize,
        kernel_rank: usize,
        kernel_start_offset: usize,
        dst: *mut u16,
        padding: usize,
        stride: usize,
        dilation: usize,
        l_out: usize,
        elem_count: usize,
    ) -> c_int;
    fn hip_conv_transpose1d_f8e4m3(
        ordinal: c_int,
        src: *const u8,
        src_dims: *const usize,
        src_strides: *const usize,
        src_rank: usize,
        src_start_offset: usize,
        kernel: *const u8,
        kernel_dims: *const usize,
        kernel_strides: *const usize,
        kernel_rank: usize,
        kernel_start_offset: usize,
        dst: *mut u8,
        padding: usize,
        stride: usize,
        dilation: usize,
        l_out: usize,
        elem_count: usize,
    ) -> c_int;
    fn hip_conv2d_f32(
        ordinal: c_int,
        src: *const f32,
        src_dims: *const usize,
        src_strides: *const usize,
        src_rank: usize,
        src_start_offset: usize,
        kernel: *const f32,
        kernel_dims: *const usize,
        kernel_strides: *const usize,
        kernel_rank: usize,
        kernel_start_offset: usize,
        dst: *mut f32,
        padding: usize,
        stride: usize,
        dilation: usize,
        out_h: usize,
        out_w: usize,
        elem_count: usize,
    ) -> c_int;
    fn hip_conv2d_bf16(
        ordinal: c_int,
        src: *const u16,
        src_dims: *const usize,
        src_strides: *const usize,
        src_rank: usize,
        src_start_offset: usize,
        kernel: *const u16,
        kernel_dims: *const usize,
        kernel_strides: *const usize,
        kernel_rank: usize,
        kernel_start_offset: usize,
        dst: *mut u16,
        padding: usize,
        stride: usize,
        dilation: usize,
        out_h: usize,
        out_w: usize,
        elem_count: usize,
    ) -> c_int;
    fn hip_conv2d_f8e4m3(
        ordinal: c_int,
        src: *const u8,
        src_dims: *const usize,
        src_strides: *const usize,
        src_rank: usize,
        src_start_offset: usize,
        kernel: *const u8,
        kernel_dims: *const usize,
        kernel_strides: *const usize,
        kernel_rank: usize,
        kernel_start_offset: usize,
        dst: *mut u8,
        padding: usize,
        stride: usize,
        dilation: usize,
        out_h: usize,
        out_w: usize,
        elem_count: usize,
    ) -> c_int;
    fn hip_conv_transpose2d_f32(
        ordinal: c_int,
        src: *const f32,
        src_dims: *const usize,
        src_strides: *const usize,
        src_rank: usize,
        src_start_offset: usize,
        kernel: *const f32,
        kernel_dims: *const usize,
        kernel_strides: *const usize,
        kernel_rank: usize,
        kernel_start_offset: usize,
        dst: *mut f32,
        padding: usize,
        stride: usize,
        dilation: usize,
        out_h: usize,
        out_w: usize,
        elem_count: usize,
    ) -> c_int;
    fn hip_conv_transpose2d_bf16(
        ordinal: c_int,
        src: *const u16,
        src_dims: *const usize,
        src_strides: *const usize,
        src_rank: usize,
        src_start_offset: usize,
        kernel: *const u16,
        kernel_dims: *const usize,
        kernel_strides: *const usize,
        kernel_rank: usize,
        kernel_start_offset: usize,
        dst: *mut u16,
        padding: usize,
        stride: usize,
        dilation: usize,
        out_h: usize,
        out_w: usize,
        elem_count: usize,
    ) -> c_int;
    fn hip_conv_transpose2d_f8e4m3(
        ordinal: c_int,
        src: *const u8,
        src_dims: *const usize,
        src_strides: *const usize,
        src_rank: usize,
        src_start_offset: usize,
        kernel: *const u8,
        kernel_dims: *const usize,
        kernel_strides: *const usize,
        kernel_rank: usize,
        kernel_start_offset: usize,
        dst: *mut u8,
        padding: usize,
        stride: usize,
        dilation: usize,
        out_h: usize,
        out_w: usize,
        elem_count: usize,
    ) -> c_int;
    fn hip_arg_sort_f32(
        ordinal: c_int,
        src: *const f32,
        start_offset: usize,
        dst: *mut u32,
        elem_count: usize,
        last_dim: usize,
        asc: c_int,
    ) -> c_int;
    fn hip_arg_sort_bf16(
        ordinal: c_int,
        src: *const u16,
        start_offset: usize,
        dst: *mut u32,
        elem_count: usize,
        last_dim: usize,
        asc: c_int,
    ) -> c_int;
    fn hip_arg_sort_f8e4m3(
        ordinal: c_int,
        src: *const u8,
        start_offset: usize,
        dst: *mut u32,
        elem_count: usize,
        last_dim: usize,
        asc: c_int,
    ) -> c_int;
    fn hip_softmax_last_dim_f32(
        ordinal: c_int,
        src: *const f32,
        start_offset: usize,
        dst: *mut f32,
        rows: usize,
        cols: usize,
    ) -> c_int;
    fn hip_softmax_last_dim_bf16(
        ordinal: c_int,
        src: *const u16,
        start_offset: usize,
        dst: *mut u16,
        rows: usize,
        cols: usize,
    ) -> c_int;
    fn hip_softmax_last_dim_f8e4m3(
        ordinal: c_int,
        src: *const u8,
        start_offset: usize,
        dst: *mut u8,
        rows: usize,
        cols: usize,
    ) -> c_int;
    fn hip_repeat_penalty_f32(
        ordinal: c_int,
        src: *const f32,
        src_start_offset: usize,
        src_stride: usize,
        token_ids: *const u32,
        token_ids_start_offset: usize,
        token_ids_stride: usize,
        dst: *mut f32,
        elem_count: usize,
        token_ids_count: usize,
        penalty: f32,
    ) -> c_int;
    fn hip_rms_norm_f32(
        ordinal: c_int,
        src: *const f32,
        src_start_offset: usize,
        alpha: *const f32,
        alpha_start_offset: usize,
        dst: *mut f32,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> c_int;
    fn hip_rms_norm_bf16(
        ordinal: c_int,
        src: *const u16,
        src_start_offset: usize,
        alpha: *const u16,
        alpha_start_offset: usize,
        dst: *mut u16,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> c_int;
    fn hip_rms_norm_f8e4m3(
        ordinal: c_int,
        src: *const u8,
        src_start_offset: usize,
        alpha: *const u8,
        alpha_start_offset: usize,
        dst: *mut u8,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> c_int;
    fn hip_layer_norm_f32(
        ordinal: c_int,
        src: *const f32,
        src_start_offset: usize,
        alpha: *const f32,
        alpha_start_offset: usize,
        beta: *const f32,
        beta_start_offset: usize,
        dst: *mut f32,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> c_int;
    fn hip_layer_norm_bf16(
        ordinal: c_int,
        src: *const u16,
        src_start_offset: usize,
        alpha: *const u16,
        alpha_start_offset: usize,
        beta: *const u16,
        beta_start_offset: usize,
        dst: *mut u16,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> c_int;
    fn hip_layer_norm_f8e4m3(
        ordinal: c_int,
        src: *const u8,
        src_start_offset: usize,
        alpha: *const u8,
        alpha_start_offset: usize,
        beta: *const u8,
        beta_start_offset: usize,
        dst: *mut u8,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> c_int;
    fn hip_rope_f32(
        ordinal: c_int,
        src: *const f32,
        src_start_offset: usize,
        cos: *const f32,
        cos_start_offset: usize,
        sin: *const f32,
        sin_start_offset: usize,
        dst: *mut f32,
        b: usize,
        h: usize,
        t: usize,
        d: usize,
        interleaved: c_int,
        unbatched_rope: c_int,
        thd: c_int,
    ) -> c_int;
    fn hip_rope_bf16(
        ordinal: c_int,
        src: *const u16,
        src_start_offset: usize,
        cos: *const u16,
        cos_start_offset: usize,
        sin: *const u16,
        sin_start_offset: usize,
        dst: *mut u16,
        b: usize,
        h: usize,
        t: usize,
        d: usize,
        interleaved: c_int,
        unbatched_rope: c_int,
        thd: c_int,
    ) -> c_int;
    fn hip_rope_f8e4m3(
        ordinal: c_int,
        src: *const u8,
        src_start_offset: usize,
        cos: *const u8,
        cos_start_offset: usize,
        sin: *const u8,
        sin_start_offset: usize,
        dst: *mut u8,
        b: usize,
        h: usize,
        t: usize,
        d: usize,
        interleaved: c_int,
        unbatched_rope: c_int,
        thd: c_int,
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

pub(crate) fn cast_f32_to_bf16(src: &Buffer, layout: &LayoutArg, dst: &Buffer) -> Result<()> {
    check(
        unsafe {
            hip_cast_f32_to_bf16(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<f32>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<u16>(),
                layout.elem_count(),
            )
        },
        "cast_f32_to_bf16",
    )
}

pub(crate) fn cast_bf16_to_f32(src: &Buffer, layout: &LayoutArg, dst: &Buffer) -> Result<()> {
    check(
        unsafe {
            hip_cast_bf16_to_f32(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u16>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<f32>(),
                layout.elem_count(),
            )
        },
        "cast_bf16_to_f32",
    )
}

pub(crate) fn cast_f32_to_f16(src: &Buffer, layout: &LayoutArg, dst: &Buffer) -> Result<()> {
    check(
        unsafe {
            hip_cast_f32_to_f16(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<f32>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<u16>(),
                layout.elem_count(),
            )
        },
        "cast_f32_to_f16",
    )
}

pub(crate) fn cast_f16_to_f32(src: &Buffer, layout: &LayoutArg, dst: &Buffer) -> Result<()> {
    check(
        unsafe {
            hip_cast_f16_to_f32(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u16>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<f32>(),
                layout.elem_count(),
            )
        },
        "cast_f16_to_f32",
    )
}

pub(crate) fn cast_f32_to_f8e4m3(src: &Buffer, layout: &LayoutArg, dst: &Buffer) -> Result<()> {
    check(
        unsafe {
            hip_cast_f32_to_f8e4m3(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<f32>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<u8>(),
                layout.elem_count(),
            )
        },
        "cast_f32_to_f8e4m3",
    )
}

pub(crate) fn cast_f8e4m3_to_f32(src: &Buffer, layout: &LayoutArg, dst: &Buffer) -> Result<()> {
    check(
        unsafe {
            hip_cast_f8e4m3_to_f32(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u8>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<f32>(),
                layout.elem_count(),
            )
        },
        "cast_f8e4m3_to_f32",
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

pub(crate) fn unary_bf16(op: i32, src: &Buffer, layout: &LayoutArg, dst: &Buffer) -> Result<()> {
    check(
        unsafe {
            hip_unary_bf16(
                ordinal_to_c_int(src.device_ordinal())?,
                op,
                src.device_ptr().cast::<u16>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<u16>(),
                layout.elem_count(),
            )
        },
        "unary_bf16",
    )
}

pub(crate) fn unary_f8e4m3(op: i32, src: &Buffer, layout: &LayoutArg, dst: &Buffer) -> Result<()> {
    check(
        unsafe {
            hip_unary_f8e4m3(
                ordinal_to_c_int(src.device_ordinal())?,
                op,
                src.device_ptr().cast::<u8>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<u8>(),
                layout.elem_count(),
            )
        },
        "unary_f8e4m3",
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

pub(crate) fn binary_bf16(
    op: i32,
    lhs: &Buffer,
    lhs_layout: &LayoutArg,
    rhs: &Buffer,
    rhs_layout: &LayoutArg,
    dst: &Buffer,
) -> Result<()> {
    check(
        unsafe {
            hip_binary_bf16(
                ordinal_to_c_int(lhs.device_ordinal())?,
                op,
                lhs.device_ptr().cast::<u16>(),
                lhs_layout.dims().as_ptr(),
                lhs_layout.stride().as_ptr(),
                lhs_layout.dims().len(),
                lhs_layout.start_offset(),
                rhs.device_ptr().cast::<u16>(),
                rhs_layout.dims().as_ptr(),
                rhs_layout.stride().as_ptr(),
                rhs_layout.dims().len(),
                rhs_layout.start_offset(),
                dst.device_ptr().cast::<u16>(),
                lhs_layout.elem_count(),
            )
        },
        "binary_bf16",
    )
}

pub(crate) fn binary_f8e4m3(
    op: i32,
    lhs: &Buffer,
    lhs_layout: &LayoutArg,
    rhs: &Buffer,
    rhs_layout: &LayoutArg,
    dst: &Buffer,
) -> Result<()> {
    check(
        unsafe {
            hip_binary_f8e4m3(
                ordinal_to_c_int(lhs.device_ordinal())?,
                op,
                lhs.device_ptr().cast::<u8>(),
                lhs_layout.dims().as_ptr(),
                lhs_layout.stride().as_ptr(),
                lhs_layout.dims().len(),
                lhs_layout.start_offset(),
                rhs.device_ptr().cast::<u8>(),
                rhs_layout.dims().as_ptr(),
                rhs_layout.stride().as_ptr(),
                rhs_layout.dims().len(),
                rhs_layout.start_offset(),
                dst.device_ptr().cast::<u8>(),
                lhs_layout.elem_count(),
            )
        },
        "binary_f8e4m3",
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

pub(crate) fn cmp_bf16(
    op: i32,
    lhs: &Buffer,
    lhs_layout: &LayoutArg,
    rhs: &Buffer,
    rhs_layout: &LayoutArg,
    dst: &Buffer,
) -> Result<()> {
    check(
        unsafe {
            hip_cmp_bf16(
                ordinal_to_c_int(lhs.device_ordinal())?,
                op,
                lhs.device_ptr().cast::<u16>(),
                lhs_layout.dims().as_ptr(),
                lhs_layout.stride().as_ptr(),
                lhs_layout.dims().len(),
                lhs_layout.start_offset(),
                rhs.device_ptr().cast::<u16>(),
                rhs_layout.dims().as_ptr(),
                rhs_layout.stride().as_ptr(),
                rhs_layout.dims().len(),
                rhs_layout.start_offset(),
                dst.device_ptr().cast::<u8>(),
                lhs_layout.elem_count(),
            )
        },
        "cmp_bf16",
    )
}

pub(crate) fn cmp_f8e4m3(
    op: i32,
    lhs: &Buffer,
    lhs_layout: &LayoutArg,
    rhs: &Buffer,
    rhs_layout: &LayoutArg,
    dst: &Buffer,
) -> Result<()> {
    check(
        unsafe {
            hip_cmp_f8e4m3(
                ordinal_to_c_int(lhs.device_ordinal())?,
                op,
                lhs.device_ptr().cast::<u8>(),
                lhs_layout.dims().as_ptr(),
                lhs_layout.stride().as_ptr(),
                lhs_layout.dims().len(),
                lhs_layout.start_offset(),
                rhs.device_ptr().cast::<u8>(),
                rhs_layout.dims().as_ptr(),
                rhs_layout.stride().as_ptr(),
                rhs_layout.dims().len(),
                rhs_layout.start_offset(),
                dst.device_ptr().cast::<u8>(),
                lhs_layout.elem_count(),
            )
        },
        "cmp_f8e4m3",
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

pub(crate) fn const_set_i16(dst: &Buffer, layout: &LayoutArg, value: i16) -> Result<()> {
    check(
        unsafe {
            hip_const_set_i16(
                ordinal_to_c_int(dst.device_ordinal())?,
                dst.device_ptr().cast::<i16>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                value,
                layout.elem_count(),
            )
        },
        "const_set_i16",
    )
}

pub(crate) fn const_set_i32(dst: &Buffer, layout: &LayoutArg, value: i32) -> Result<()> {
    check(
        unsafe {
            hip_const_set_i32(
                ordinal_to_c_int(dst.device_ordinal())?,
                dst.device_ptr().cast::<i32>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                value,
                layout.elem_count(),
            )
        },
        "const_set_i32",
    )
}

pub(crate) fn const_set_i64(dst: &Buffer, layout: &LayoutArg, value: i64) -> Result<()> {
    check(
        unsafe {
            hip_const_set_i64(
                ordinal_to_c_int(dst.device_ordinal())?,
                dst.device_ptr().cast::<i64>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                value,
                layout.elem_count(),
            )
        },
        "const_set_i64",
    )
}

pub(crate) fn const_set_bf16(dst: &Buffer, layout: &LayoutArg, value: u16) -> Result<()> {
    check(
        unsafe {
            hip_const_set_bf16(
                ordinal_to_c_int(dst.device_ordinal())?,
                dst.device_ptr().cast::<u16>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                value,
                layout.elem_count(),
            )
        },
        "const_set_bf16",
    )
}

pub(crate) fn const_set_f16(dst: &Buffer, layout: &LayoutArg, value: u16) -> Result<()> {
    check(
        unsafe {
            hip_const_set_f16(
                ordinal_to_c_int(dst.device_ordinal())?,
                dst.device_ptr().cast::<u16>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                value,
                layout.elem_count(),
            )
        },
        "const_set_f16",
    )
}

pub(crate) fn const_set_f64(dst: &Buffer, layout: &LayoutArg, value: f64) -> Result<()> {
    check(
        unsafe {
            hip_const_set_f64(
                ordinal_to_c_int(dst.device_ordinal())?,
                dst.device_ptr().cast::<f64>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                value,
                layout.elem_count(),
            )
        },
        "const_set_f64",
    )
}

pub(crate) fn const_set_f8e4m3(dst: &Buffer, layout: &LayoutArg, value: u8) -> Result<()> {
    check(
        unsafe {
            hip_const_set_f8e4m3(
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
        "const_set_f8e4m3",
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

pub(crate) fn affine_f32(
    src: &Buffer,
    layout: &LayoutArg,
    dst: &Buffer,
    mul: f32,
    add: f32,
) -> Result<()> {
    check(
        unsafe {
            hip_affine_f32(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<f32>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<f32>(),
                layout.elem_count(),
                mul,
                add,
            )
        },
        "affine_f32",
    )
}

pub(crate) fn affine_bf16(
    src: &Buffer,
    layout: &LayoutArg,
    dst: &Buffer,
    mul: f32,
    add: f32,
) -> Result<()> {
    check(
        unsafe {
            hip_affine_bf16(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u16>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<u16>(),
                layout.elem_count(),
                mul,
                add,
            )
        },
        "affine_bf16",
    )
}

pub(crate) fn affine_f8e4m3(
    src: &Buffer,
    layout: &LayoutArg,
    dst: &Buffer,
    mul: f32,
    add: f32,
) -> Result<()> {
    check(
        unsafe {
            hip_affine_f8e4m3(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u8>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<u8>(),
                layout.elem_count(),
                mul,
                add,
            )
        },
        "affine_f8e4m3",
    )
}

pub(crate) fn powf_f32(src: &Buffer, layout: &LayoutArg, dst: &Buffer, value: f32) -> Result<()> {
    check(
        unsafe {
            hip_powf_f32(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<f32>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<f32>(),
                layout.elem_count(),
                value,
            )
        },
        "powf_f32",
    )
}

pub(crate) fn powf_bf16(src: &Buffer, layout: &LayoutArg, dst: &Buffer, value: f32) -> Result<()> {
    check(
        unsafe {
            hip_powf_bf16(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u16>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<u16>(),
                layout.elem_count(),
                value,
            )
        },
        "powf_bf16",
    )
}

pub(crate) fn powf_f8e4m3(
    src: &Buffer,
    layout: &LayoutArg,
    dst: &Buffer,
    value: f32,
) -> Result<()> {
    check(
        unsafe {
            hip_powf_f8e4m3(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u8>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<u8>(),
                layout.elem_count(),
                value,
            )
        },
        "powf_f8e4m3",
    )
}

pub(crate) fn elu_f32(src: &Buffer, layout: &LayoutArg, dst: &Buffer, alpha: f32) -> Result<()> {
    check(
        unsafe {
            hip_elu_f32(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<f32>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<f32>(),
                layout.elem_count(),
                alpha,
            )
        },
        "elu_f32",
    )
}

pub(crate) fn elu_bf16(src: &Buffer, layout: &LayoutArg, dst: &Buffer, alpha: f32) -> Result<()> {
    check(
        unsafe {
            hip_elu_bf16(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u16>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<u16>(),
                layout.elem_count(),
                alpha,
            )
        },
        "elu_bf16",
    )
}

pub(crate) fn elu_f8e4m3(src: &Buffer, layout: &LayoutArg, dst: &Buffer, alpha: f32) -> Result<()> {
    check(
        unsafe {
            hip_elu_f8e4m3(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u8>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<u8>(),
                layout.elem_count(),
                alpha,
            )
        },
        "elu_f8e4m3",
    )
}

pub(crate) fn reduce_f32(
    op: i32,
    src: &Buffer,
    layout: &LayoutArg,
    reduce_mask: u64,
    reduce_count: usize,
    dst: &Buffer,
    elem_count: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_reduce_f32(
                ordinal_to_c_int(src.device_ordinal())?,
                op,
                src.device_ptr().cast::<f32>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                reduce_mask,
                reduce_count,
                dst.device_ptr(),
                elem_count,
            )
        },
        "reduce_f32",
    )
}

pub(crate) fn reduce_bf16(
    op: i32,
    src: &Buffer,
    layout: &LayoutArg,
    reduce_mask: u64,
    reduce_count: usize,
    dst: &Buffer,
    elem_count: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_reduce_bf16(
                ordinal_to_c_int(src.device_ordinal())?,
                op,
                src.device_ptr().cast::<u16>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                reduce_mask,
                reduce_count,
                dst.device_ptr(),
                elem_count,
            )
        },
        "reduce_bf16",
    )
}

pub(crate) fn reduce_f8e4m3(
    op: i32,
    src: &Buffer,
    layout: &LayoutArg,
    reduce_mask: u64,
    reduce_count: usize,
    dst: &Buffer,
    elem_count: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_reduce_f8e4m3(
                ordinal_to_c_int(src.device_ordinal())?,
                op,
                src.device_ptr().cast::<u8>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                reduce_mask,
                reduce_count,
                dst.device_ptr(),
                elem_count,
            )
        },
        "reduce_f8e4m3",
    )
}

pub(crate) fn random_uniform_f32(
    dst: &Buffer,
    elem_count: usize,
    seed: u64,
    lo: f32,
    up: f32,
) -> Result<()> {
    check(
        unsafe {
            hip_random_uniform_f32(
                ordinal_to_c_int(dst.device_ordinal())?,
                dst.device_ptr().cast::<f32>(),
                elem_count,
                seed,
                lo,
                up,
            )
        },
        "random_uniform_f32",
    )
}

pub(crate) fn random_uniform_bf16(
    dst: &Buffer,
    elem_count: usize,
    seed: u64,
    lo: f32,
    up: f32,
) -> Result<()> {
    check(
        unsafe {
            hip_random_uniform_bf16(
                ordinal_to_c_int(dst.device_ordinal())?,
                dst.device_ptr().cast::<u16>(),
                elem_count,
                seed,
                lo,
                up,
            )
        },
        "random_uniform_bf16",
    )
}

pub(crate) fn random_uniform_f8e4m3(
    dst: &Buffer,
    elem_count: usize,
    seed: u64,
    lo: f32,
    up: f32,
) -> Result<()> {
    check(
        unsafe {
            hip_random_uniform_f8e4m3(
                ordinal_to_c_int(dst.device_ordinal())?,
                dst.device_ptr().cast::<u8>(),
                elem_count,
                seed,
                lo,
                up,
            )
        },
        "random_uniform_f8e4m3",
    )
}

pub(crate) fn random_normal_f32(
    dst: &Buffer,
    elem_count: usize,
    seed: u64,
    mean: f32,
    std: f32,
) -> Result<()> {
    check(
        unsafe {
            hip_random_normal_f32(
                ordinal_to_c_int(dst.device_ordinal())?,
                dst.device_ptr().cast::<f32>(),
                elem_count,
                seed,
                mean,
                std,
            )
        },
        "random_normal_f32",
    )
}

pub(crate) fn random_normal_bf16(
    dst: &Buffer,
    elem_count: usize,
    seed: u64,
    mean: f32,
    std: f32,
) -> Result<()> {
    check(
        unsafe {
            hip_random_normal_bf16(
                ordinal_to_c_int(dst.device_ordinal())?,
                dst.device_ptr().cast::<u16>(),
                elem_count,
                seed,
                mean,
                std,
            )
        },
        "random_normal_bf16",
    )
}

pub(crate) fn random_normal_f8e4m3(
    dst: &Buffer,
    elem_count: usize,
    seed: u64,
    mean: f32,
    std: f32,
) -> Result<()> {
    check(
        unsafe {
            hip_random_normal_f8e4m3(
                ordinal_to_c_int(dst.device_ordinal())?,
                dst.device_ptr().cast::<u8>(),
                elem_count,
                seed,
                mean,
                std,
            )
        },
        "random_normal_f8e4m3",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_f32(
    lhs: &Buffer,
    rhs: &Buffer,
    dst: &Buffer,
    bmnk: (usize, usize, usize, usize),
    lhs_start_offset: usize,
    rhs_start_offset: usize,
    lhs_batch_stride: usize,
    rhs_batch_stride: usize,
    lhs_row_stride: usize,
    lhs_col_stride: usize,
    rhs_row_stride: usize,
    rhs_col_stride: usize,
) -> Result<()> {
    let (b, m, n, k) = bmnk;
    check(
        unsafe {
            hip_matmul_f32(
                ordinal_to_c_int(lhs.device_ordinal())?,
                lhs.device_ptr().cast::<f32>(),
                rhs.device_ptr().cast::<f32>(),
                dst.device_ptr().cast::<f32>(),
                b,
                m,
                n,
                k,
                lhs_start_offset,
                rhs_start_offset,
                lhs_batch_stride,
                rhs_batch_stride,
                lhs_row_stride,
                lhs_col_stride,
                rhs_row_stride,
                rhs_col_stride,
            )
        },
        "matmul_f32",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_bf16(
    lhs: &Buffer,
    rhs: &Buffer,
    dst: &Buffer,
    bmnk: (usize, usize, usize, usize),
    lhs_start_offset: usize,
    rhs_start_offset: usize,
    lhs_batch_stride: usize,
    rhs_batch_stride: usize,
    lhs_row_stride: usize,
    lhs_col_stride: usize,
    rhs_row_stride: usize,
    rhs_col_stride: usize,
) -> Result<()> {
    let (b, m, n, k) = bmnk;
    check(
        unsafe {
            hip_matmul_bf16(
                ordinal_to_c_int(lhs.device_ordinal())?,
                lhs.device_ptr().cast::<u16>(),
                rhs.device_ptr().cast::<u16>(),
                dst.device_ptr().cast::<u16>(),
                b,
                m,
                n,
                k,
                lhs_start_offset,
                rhs_start_offset,
                lhs_batch_stride,
                rhs_batch_stride,
                lhs_row_stride,
                lhs_col_stride,
                rhs_row_stride,
                rhs_col_stride,
            )
        },
        "matmul_bf16",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_f8e4m3(
    lhs: &Buffer,
    rhs: &Buffer,
    dst: &Buffer,
    bmnk: (usize, usize, usize, usize),
    lhs_start_offset: usize,
    rhs_start_offset: usize,
    lhs_batch_stride: usize,
    rhs_batch_stride: usize,
    lhs_row_stride: usize,
    lhs_col_stride: usize,
    rhs_row_stride: usize,
    rhs_col_stride: usize,
) -> Result<()> {
    let (b, m, n, k) = bmnk;
    check(
        unsafe {
            hip_matmul_f8e4m3(
                ordinal_to_c_int(lhs.device_ordinal())?,
                lhs.device_ptr().cast::<u8>(),
                rhs.device_ptr().cast::<u8>(),
                dst.device_ptr().cast::<u8>(),
                b,
                m,
                n,
                k,
                lhs_start_offset,
                rhs_start_offset,
                lhs_batch_stride,
                rhs_batch_stride,
                lhs_row_stride,
                lhs_col_stride,
                rhs_row_stride,
                rhs_col_stride,
            )
        },
        "matmul_f8e4m3",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qmatmul_t_q5_0_f32(
    weights: &Buffer,
    rhs: &Buffer,
    rhs_layout: &LayoutArg,
    dst: &Buffer,
    batch_size: usize,
    nrows: usize,
    ncols: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_qmatmul_t_q5_0_f32(
                ordinal_to_c_int(weights.device_ordinal())?,
                weights.device_ptr().cast::<u8>(),
                rhs.device_ptr().cast::<f32>(),
                rhs_layout.dims().as_ptr(),
                rhs_layout.stride().as_ptr(),
                rhs_layout.dims().len(),
                rhs_layout.start_offset(),
                dst.device_ptr().cast::<f32>(),
                batch_size,
                nrows,
                ncols,
            )
        },
        "qmatmul_t_q5_0_f32",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qmatmul_t_q8_0_f32(
    weights: &Buffer,
    rhs: &Buffer,
    rhs_layout: &LayoutArg,
    dst: &Buffer,
    batch_size: usize,
    nrows: usize,
    ncols: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_qmatmul_t_q8_0_f32(
                ordinal_to_c_int(weights.device_ordinal())?,
                weights.device_ptr().cast::<u8>(),
                rhs.device_ptr().cast::<f32>(),
                rhs_layout.dims().as_ptr(),
                rhs_layout.stride().as_ptr(),
                rhs_layout.dims().len(),
                rhs_layout.start_offset(),
                dst.device_ptr().cast::<f32>(),
                batch_size,
                nrows,
                ncols,
            )
        },
        "qmatmul_t_q8_0_f32",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qmatmul_t_q4k_f32(
    weights: &Buffer,
    rhs: &Buffer,
    rhs_layout: &LayoutArg,
    dst: &Buffer,
    batch_size: usize,
    nrows: usize,
    ncols: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_qmatmul_t_q4k_f32(
                ordinal_to_c_int(weights.device_ordinal())?,
                weights.device_ptr().cast::<u8>(),
                rhs.device_ptr().cast::<f32>(),
                rhs_layout.dims().as_ptr(),
                rhs_layout.stride().as_ptr(),
                rhs_layout.dims().len(),
                rhs_layout.start_offset(),
                dst.device_ptr().cast::<f32>(),
                batch_size,
                nrows,
                ncols,
            )
        },
        "qmatmul_t_q4k_f32",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qmatmul_t_q6k_f32(
    weights: &Buffer,
    rhs: &Buffer,
    rhs_layout: &LayoutArg,
    dst: &Buffer,
    batch_size: usize,
    nrows: usize,
    ncols: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_qmatmul_t_q6k_f32(
                ordinal_to_c_int(weights.device_ordinal())?,
                weights.device_ptr().cast::<u8>(),
                rhs.device_ptr().cast::<f32>(),
                rhs_layout.dims().as_ptr(),
                rhs_layout.stride().as_ptr(),
                rhs_layout.dims().len(),
                rhs_layout.start_offset(),
                dst.device_ptr().cast::<f32>(),
                batch_size,
                nrows,
                ncols,
            )
        },
        "qmatmul_t_q6k_f32",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_gemm_gguf_q8_0_f32(
    input: &Buffer,
    weights: &Buffer,
    sorted_token_ids: &Buffer,
    expert_ids: &Buffer,
    topk_weights: Option<&Buffer>,
    dst: &Buffer,
    num_experts: usize,
    topk: usize,
    size_m: usize,
    size_n: usize,
    size_k: usize,
) -> Result<()> {
    let topk_weights = topk_weights
        .map(|buffer| buffer.device_ptr().cast::<f32>() as *const f32)
        .unwrap_or(ptr::null());
    check(
        unsafe {
            hip_moe_gemm_gguf_q8_0_f32(
                ordinal_to_c_int(weights.device_ordinal())?,
                input.device_ptr().cast::<f32>(),
                weights.device_ptr().cast::<u8>(),
                sorted_token_ids.device_ptr().cast::<u32>(),
                expert_ids.device_ptr().cast::<u32>(),
                topk_weights,
                dst.device_ptr().cast::<f32>(),
                num_experts,
                topk,
                size_m,
                size_n,
                size_k,
            )
        },
        "moe_gemm_gguf_q8_0_f32",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_gemm_gguf_q4k_f32(
    input: &Buffer,
    weights: &Buffer,
    sorted_token_ids: &Buffer,
    expert_ids: &Buffer,
    topk_weights: Option<&Buffer>,
    dst: &Buffer,
    num_experts: usize,
    topk: usize,
    size_m: usize,
    size_n: usize,
    size_k: usize,
) -> Result<()> {
    let topk_weights = topk_weights
        .map(|buffer| buffer.device_ptr().cast::<f32>() as *const f32)
        .unwrap_or(ptr::null());
    check(
        unsafe {
            hip_moe_gemm_gguf_q4k_f32(
                ordinal_to_c_int(weights.device_ordinal())?,
                input.device_ptr().cast::<f32>(),
                weights.device_ptr().cast::<u8>(),
                sorted_token_ids.device_ptr().cast::<u32>(),
                expert_ids.device_ptr().cast::<u32>(),
                topk_weights,
                dst.device_ptr().cast::<f32>(),
                num_experts,
                topk,
                size_m,
                size_n,
                size_k,
            )
        },
        "moe_gemm_gguf_q4k_f32",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_gemm_gguf_q6k_f32(
    input: &Buffer,
    weights: &Buffer,
    sorted_token_ids: &Buffer,
    expert_ids: &Buffer,
    topk_weights: Option<&Buffer>,
    dst: &Buffer,
    num_experts: usize,
    topk: usize,
    size_m: usize,
    size_n: usize,
    size_k: usize,
) -> Result<()> {
    let topk_weights = topk_weights
        .map(|buffer| buffer.device_ptr().cast::<f32>() as *const f32)
        .unwrap_or(ptr::null());
    check(
        unsafe {
            hip_moe_gemm_gguf_q6k_f32(
                ordinal_to_c_int(weights.device_ordinal())?,
                input.device_ptr().cast::<f32>(),
                weights.device_ptr().cast::<u8>(),
                sorted_token_ids.device_ptr().cast::<u32>(),
                expert_ids.device_ptr().cast::<u32>(),
                topk_weights,
                dst.device_ptr().cast::<f32>(),
                num_experts,
                topk,
                size_m,
                size_n,
                size_k,
            )
        },
        "moe_gemm_gguf_q6k_f32",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn index_select_u32_f32(
    src: &Buffer,
    src_layout: &LayoutArg,
    ids: &Buffer,
    ids_start_offset: usize,
    ids_stride: usize,
    dim: usize,
    n_ids: usize,
    dst: &Buffer,
    elem_count: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_index_select_u32_f32(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<f32>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                ids.device_ptr().cast::<u32>(),
                ids_start_offset,
                ids_stride,
                dim,
                n_ids,
                dst.device_ptr().cast::<f32>(),
                elem_count,
            )
        },
        "index_select_u32_f32",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn index_select_i64_f32(
    src: &Buffer,
    src_layout: &LayoutArg,
    ids: &Buffer,
    ids_start_offset: usize,
    ids_stride: usize,
    dim: usize,
    n_ids: usize,
    dst: &Buffer,
    elem_count: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_index_select_i64_f32(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<f32>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                ids.device_ptr().cast::<i64>(),
                ids_start_offset,
                ids_stride,
                dim,
                n_ids,
                dst.device_ptr().cast::<f32>(),
                elem_count,
            )
        },
        "index_select_i64_f32",
    )
}

pub(crate) fn index_select_u32_bf16(
    src: &Buffer,
    src_layout: &LayoutArg,
    ids: &Buffer,
    ids_start_offset: usize,
    ids_stride: usize,
    dim: usize,
    n_ids: usize,
    dst: &Buffer,
    elem_count: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_index_select_u32_bf16(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u16>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                ids.device_ptr().cast::<u32>(),
                ids_start_offset,
                ids_stride,
                dim,
                n_ids,
                dst.device_ptr().cast::<u16>(),
                elem_count,
            )
        },
        "index_select_u32_bf16",
    )
}

pub(crate) fn index_select_i64_bf16(
    src: &Buffer,
    src_layout: &LayoutArg,
    ids: &Buffer,
    ids_start_offset: usize,
    ids_stride: usize,
    dim: usize,
    n_ids: usize,
    dst: &Buffer,
    elem_count: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_index_select_i64_bf16(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u16>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                ids.device_ptr().cast::<i64>(),
                ids_start_offset,
                ids_stride,
                dim,
                n_ids,
                dst.device_ptr().cast::<u16>(),
                elem_count,
            )
        },
        "index_select_i64_bf16",
    )
}

pub(crate) fn index_select_u32_f8e4m3(
    src: &Buffer,
    src_layout: &LayoutArg,
    ids: &Buffer,
    ids_start_offset: usize,
    ids_stride: usize,
    dim: usize,
    n_ids: usize,
    dst: &Buffer,
    elem_count: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_index_select_u32_f8e4m3(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u8>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                ids.device_ptr().cast::<u32>(),
                ids_start_offset,
                ids_stride,
                dim,
                n_ids,
                dst.device_ptr().cast::<u8>(),
                elem_count,
            )
        },
        "index_select_u32_f8e4m3",
    )
}

pub(crate) fn index_select_i64_f8e4m3(
    src: &Buffer,
    src_layout: &LayoutArg,
    ids: &Buffer,
    ids_start_offset: usize,
    ids_stride: usize,
    dim: usize,
    n_ids: usize,
    dst: &Buffer,
    elem_count: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_index_select_i64_f8e4m3(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u8>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                ids.device_ptr().cast::<i64>(),
                ids_start_offset,
                ids_stride,
                dim,
                n_ids,
                dst.device_ptr().cast::<u8>(),
                elem_count,
            )
        },
        "index_select_i64_f8e4m3",
    )
}

pub(crate) fn gather_u32_f32(
    src: &Buffer,
    src_layout: &LayoutArg,
    ids: &Buffer,
    ids_layout: &LayoutArg,
    dim: usize,
    dst: &Buffer,
) -> Result<()> {
    check(
        unsafe {
            hip_gather_u32_f32(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<f32>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                ids.device_ptr().cast::<u32>(),
                ids_layout.dims().as_ptr(),
                ids_layout.stride().as_ptr(),
                ids_layout.dims().len(),
                ids_layout.start_offset(),
                dim,
                dst.device_ptr().cast::<f32>(),
                ids_layout.elem_count(),
            )
        },
        "gather_u32_f32",
    )
}

pub(crate) fn gather_u32_bf16(
    src: &Buffer,
    src_layout: &LayoutArg,
    ids: &Buffer,
    ids_layout: &LayoutArg,
    dim: usize,
    dst: &Buffer,
) -> Result<()> {
    check(
        unsafe {
            hip_gather_u32_bf16(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u16>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                ids.device_ptr().cast::<u32>(),
                ids_layout.dims().as_ptr(),
                ids_layout.stride().as_ptr(),
                ids_layout.dims().len(),
                ids_layout.start_offset(),
                dim,
                dst.device_ptr().cast::<u16>(),
                ids_layout.elem_count(),
            )
        },
        "gather_u32_bf16",
    )
}

pub(crate) fn gather_u32_f8e4m3(
    src: &Buffer,
    src_layout: &LayoutArg,
    ids: &Buffer,
    ids_layout: &LayoutArg,
    dim: usize,
    dst: &Buffer,
) -> Result<()> {
    check(
        unsafe {
            hip_gather_u32_f8e4m3(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u8>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                ids.device_ptr().cast::<u32>(),
                ids_layout.dims().as_ptr(),
                ids_layout.stride().as_ptr(),
                ids_layout.dims().len(),
                ids_layout.start_offset(),
                dim,
                dst.device_ptr().cast::<u8>(),
                ids_layout.elem_count(),
            )
        },
        "gather_u32_f8e4m3",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter_u32_f32(
    add: bool,
    dst: &Buffer,
    dst_layout: &LayoutArg,
    ids: &Buffer,
    ids_layout: &LayoutArg,
    src: &Buffer,
    src_layout: &LayoutArg,
    dim: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_scatter_u32_f32(
                ordinal_to_c_int(dst.device_ordinal())?,
                if add { 1 } else { 0 },
                dst.device_ptr().cast::<f32>(),
                dst_layout.dims().as_ptr(),
                dst_layout.stride().as_ptr(),
                dst_layout.dims().len(),
                dst_layout.start_offset(),
                ids.device_ptr().cast::<u32>(),
                ids_layout.dims().as_ptr(),
                ids_layout.stride().as_ptr(),
                ids_layout.dims().len(),
                ids_layout.start_offset(),
                src.device_ptr().cast::<f32>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                dim,
                src_layout.elem_count(),
                dst_layout.elem_count(),
            )
        },
        "scatter_u32_f32",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter_u32_bf16(
    add: bool,
    dst: &Buffer,
    dst_layout: &LayoutArg,
    ids: &Buffer,
    ids_layout: &LayoutArg,
    src: &Buffer,
    src_layout: &LayoutArg,
    dim: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_scatter_u32_bf16(
                ordinal_to_c_int(dst.device_ordinal())?,
                if add { 1 } else { 0 },
                dst.device_ptr().cast::<u16>(),
                dst_layout.dims().as_ptr(),
                dst_layout.stride().as_ptr(),
                dst_layout.dims().len(),
                dst_layout.start_offset(),
                ids.device_ptr().cast::<u32>(),
                ids_layout.dims().as_ptr(),
                ids_layout.stride().as_ptr(),
                ids_layout.dims().len(),
                ids_layout.start_offset(),
                src.device_ptr().cast::<u16>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                dim,
                src_layout.elem_count(),
                dst_layout.elem_count(),
            )
        },
        "scatter_u32_bf16",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter_u32_f8e4m3(
    add: bool,
    dst: &Buffer,
    dst_layout: &LayoutArg,
    ids: &Buffer,
    ids_layout: &LayoutArg,
    src: &Buffer,
    src_layout: &LayoutArg,
    dim: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_scatter_u32_f8e4m3(
                ordinal_to_c_int(dst.device_ordinal())?,
                if add { 1 } else { 0 },
                dst.device_ptr().cast::<u8>(),
                dst_layout.dims().as_ptr(),
                dst_layout.stride().as_ptr(),
                dst_layout.dims().len(),
                dst_layout.start_offset(),
                ids.device_ptr().cast::<u32>(),
                ids_layout.dims().as_ptr(),
                ids_layout.stride().as_ptr(),
                ids_layout.dims().len(),
                ids_layout.start_offset(),
                src.device_ptr().cast::<u8>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                dim,
                src_layout.elem_count(),
                dst_layout.elem_count(),
            )
        },
        "scatter_u32_f8e4m3",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn index_add_u32_f32(
    input: &Buffer,
    input_layout: &LayoutArg,
    ids: &Buffer,
    ids_start_offset: usize,
    ids_stride: usize,
    ids_len: usize,
    src: &Buffer,
    src_layout: &LayoutArg,
    dim: usize,
    dst: &Buffer,
) -> Result<()> {
    check(
        unsafe {
            hip_index_add_u32_f32(
                ordinal_to_c_int(input.device_ordinal())?,
                input.device_ptr().cast::<f32>(),
                input_layout.dims().as_ptr(),
                input_layout.stride().as_ptr(),
                input_layout.dims().len(),
                input_layout.start_offset(),
                ids.device_ptr().cast::<u32>(),
                ids_start_offset,
                ids_stride,
                ids_len,
                src.device_ptr().cast::<f32>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                dim,
                dst.device_ptr().cast::<f32>(),
                input_layout.elem_count(),
            )
        },
        "index_add_u32_f32",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn index_add_u32_bf16(
    input: &Buffer,
    input_layout: &LayoutArg,
    ids: &Buffer,
    ids_start_offset: usize,
    ids_stride: usize,
    ids_len: usize,
    src: &Buffer,
    src_layout: &LayoutArg,
    dim: usize,
    dst: &Buffer,
) -> Result<()> {
    check(
        unsafe {
            hip_index_add_u32_bf16(
                ordinal_to_c_int(input.device_ordinal())?,
                input.device_ptr().cast::<u16>(),
                input_layout.dims().as_ptr(),
                input_layout.stride().as_ptr(),
                input_layout.dims().len(),
                input_layout.start_offset(),
                ids.device_ptr().cast::<u32>(),
                ids_start_offset,
                ids_stride,
                ids_len,
                src.device_ptr().cast::<u16>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                dim,
                dst.device_ptr().cast::<u16>(),
                input_layout.elem_count(),
            )
        },
        "index_add_u32_bf16",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn index_add_u32_f8e4m3(
    input: &Buffer,
    input_layout: &LayoutArg,
    ids: &Buffer,
    ids_start_offset: usize,
    ids_stride: usize,
    ids_len: usize,
    src: &Buffer,
    src_layout: &LayoutArg,
    dim: usize,
    dst: &Buffer,
) -> Result<()> {
    check(
        unsafe {
            hip_index_add_u32_f8e4m3(
                ordinal_to_c_int(input.device_ordinal())?,
                input.device_ptr().cast::<u8>(),
                input_layout.dims().as_ptr(),
                input_layout.stride().as_ptr(),
                input_layout.dims().len(),
                input_layout.start_offset(),
                ids.device_ptr().cast::<u32>(),
                ids_start_offset,
                ids_stride,
                ids_len,
                src.device_ptr().cast::<u8>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                dim,
                dst.device_ptr().cast::<u8>(),
                input_layout.elem_count(),
            )
        },
        "index_add_u32_f8e4m3",
    )
}

pub(crate) fn where_u8_f32(
    cond: &Buffer,
    cond_layout: &LayoutArg,
    on_true: &Buffer,
    true_layout: &LayoutArg,
    on_false: &Buffer,
    false_layout: &LayoutArg,
    dst: &Buffer,
) -> Result<()> {
    check(
        unsafe {
            hip_where_u8_f32(
                ordinal_to_c_int(cond.device_ordinal())?,
                cond.device_ptr().cast::<u8>(),
                cond_layout.dims().as_ptr(),
                cond_layout.stride().as_ptr(),
                cond_layout.dims().len(),
                cond_layout.start_offset(),
                on_true.device_ptr().cast::<f32>(),
                true_layout.dims().as_ptr(),
                true_layout.stride().as_ptr(),
                true_layout.dims().len(),
                true_layout.start_offset(),
                on_false.device_ptr().cast::<f32>(),
                false_layout.dims().as_ptr(),
                false_layout.stride().as_ptr(),
                false_layout.dims().len(),
                false_layout.start_offset(),
                dst.device_ptr().cast::<f32>(),
                cond_layout.elem_count(),
            )
        },
        "where_u8_f32",
    )
}

pub(crate) fn where_u8_bf16(
    cond: &Buffer,
    cond_layout: &LayoutArg,
    on_true: &Buffer,
    true_layout: &LayoutArg,
    on_false: &Buffer,
    false_layout: &LayoutArg,
    dst: &Buffer,
) -> Result<()> {
    check(
        unsafe {
            hip_where_u8_bf16(
                ordinal_to_c_int(cond.device_ordinal())?,
                cond.device_ptr().cast::<u8>(),
                cond_layout.dims().as_ptr(),
                cond_layout.stride().as_ptr(),
                cond_layout.dims().len(),
                cond_layout.start_offset(),
                on_true.device_ptr().cast::<u16>(),
                true_layout.dims().as_ptr(),
                true_layout.stride().as_ptr(),
                true_layout.dims().len(),
                true_layout.start_offset(),
                on_false.device_ptr().cast::<u16>(),
                false_layout.dims().as_ptr(),
                false_layout.stride().as_ptr(),
                false_layout.dims().len(),
                false_layout.start_offset(),
                dst.device_ptr().cast::<u16>(),
                cond_layout.elem_count(),
            )
        },
        "where_u8_bf16",
    )
}

pub(crate) fn where_u8_f8e4m3(
    cond: &Buffer,
    cond_layout: &LayoutArg,
    on_true: &Buffer,
    true_layout: &LayoutArg,
    on_false: &Buffer,
    false_layout: &LayoutArg,
    dst: &Buffer,
) -> Result<()> {
    check(
        unsafe {
            hip_where_u8_f8e4m3(
                ordinal_to_c_int(cond.device_ordinal())?,
                cond.device_ptr().cast::<u8>(),
                cond_layout.dims().as_ptr(),
                cond_layout.stride().as_ptr(),
                cond_layout.dims().len(),
                cond_layout.start_offset(),
                on_true.device_ptr().cast::<u8>(),
                true_layout.dims().as_ptr(),
                true_layout.stride().as_ptr(),
                true_layout.dims().len(),
                true_layout.start_offset(),
                on_false.device_ptr().cast::<u8>(),
                false_layout.dims().as_ptr(),
                false_layout.stride().as_ptr(),
                false_layout.dims().len(),
                false_layout.start_offset(),
                dst.device_ptr().cast::<u8>(),
                cond_layout.elem_count(),
            )
        },
        "where_u8_f8e4m3",
    )
}

pub(crate) fn softmax_last_dim_f32(
    src: &Buffer,
    start_offset: usize,
    dst: &Buffer,
    rows: usize,
    cols: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_softmax_last_dim_f32(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<f32>(),
                start_offset,
                dst.device_ptr().cast::<f32>(),
                rows,
                cols,
            )
        },
        "softmax_last_dim_f32",
    )
}

pub(crate) fn softmax_last_dim_bf16(
    src: &Buffer,
    start_offset: usize,
    dst: &Buffer,
    rows: usize,
    cols: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_softmax_last_dim_bf16(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u16>(),
                start_offset,
                dst.device_ptr().cast::<u16>(),
                rows,
                cols,
            )
        },
        "softmax_last_dim_bf16",
    )
}

pub(crate) fn softmax_last_dim_f8e4m3(
    src: &Buffer,
    start_offset: usize,
    dst: &Buffer,
    rows: usize,
    cols: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_softmax_last_dim_f8e4m3(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u8>(),
                start_offset,
                dst.device_ptr().cast::<u8>(),
                rows,
                cols,
            )
        },
        "softmax_last_dim_f8e4m3",
    )
}

pub(crate) fn repeat_penalty_f32(
    src: &Buffer,
    src_start_offset: usize,
    src_stride: usize,
    token_ids: &Buffer,
    token_ids_start_offset: usize,
    token_ids_stride: usize,
    dst: &Buffer,
    elem_count: usize,
    token_ids_count: usize,
    penalty: f32,
) -> Result<()> {
    check(
        unsafe {
            hip_repeat_penalty_f32(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<f32>(),
                src_start_offset,
                src_stride,
                token_ids.device_ptr().cast::<u32>(),
                token_ids_start_offset,
                token_ids_stride,
                dst.device_ptr().cast::<f32>(),
                elem_count,
                token_ids_count,
                penalty,
            )
        },
        "repeat_penalty_f32",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn pool2d_f32(
    op: i32,
    src: &Buffer,
    layout: &LayoutArg,
    dst: &Buffer,
    kernel: (usize, usize),
    stride: (usize, usize),
    out_h: usize,
    out_w: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_pool2d_f32(
                ordinal_to_c_int(src.device_ordinal())?,
                op,
                src.device_ptr().cast::<f32>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<f32>(),
                kernel.0,
                kernel.1,
                stride.0,
                stride.1,
                out_h,
                out_w,
                out_h * out_w * layout.dims()[0] * layout.dims()[1],
            )
        },
        "pool2d_f32",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn pool2d_bf16(
    op: i32,
    src: &Buffer,
    layout: &LayoutArg,
    dst: &Buffer,
    kernel: (usize, usize),
    stride: (usize, usize),
    out_h: usize,
    out_w: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_pool2d_bf16(
                ordinal_to_c_int(src.device_ordinal())?,
                op,
                src.device_ptr().cast::<u16>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<u16>(),
                kernel.0,
                kernel.1,
                stride.0,
                stride.1,
                out_h,
                out_w,
                out_h * out_w * layout.dims()[0] * layout.dims()[1],
            )
        },
        "pool2d_bf16",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn pool2d_f8e4m3(
    op: i32,
    src: &Buffer,
    layout: &LayoutArg,
    dst: &Buffer,
    kernel: (usize, usize),
    stride: (usize, usize),
    out_h: usize,
    out_w: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_pool2d_f8e4m3(
                ordinal_to_c_int(src.device_ordinal())?,
                op,
                src.device_ptr().cast::<u8>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<u8>(),
                kernel.0,
                kernel.1,
                stride.0,
                stride.1,
                out_h,
                out_w,
                out_h * out_w * layout.dims()[0] * layout.dims()[1],
            )
        },
        "pool2d_f8e4m3",
    )
}

pub(crate) fn upsample_nearest1d_f32(
    src: &Buffer,
    layout: &LayoutArg,
    dst: &Buffer,
    out_size: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_upsample_nearest1d_f32(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<f32>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<f32>(),
                out_size,
                layout.dims()[0] * layout.dims()[1] * out_size,
            )
        },
        "upsample_nearest1d_f32",
    )
}

pub(crate) fn upsample_nearest1d_bf16(
    src: &Buffer,
    layout: &LayoutArg,
    dst: &Buffer,
    out_size: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_upsample_nearest1d_bf16(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u16>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<u16>(),
                out_size,
                layout.dims()[0] * layout.dims()[1] * out_size,
            )
        },
        "upsample_nearest1d_bf16",
    )
}

pub(crate) fn upsample_nearest1d_f8e4m3(
    src: &Buffer,
    layout: &LayoutArg,
    dst: &Buffer,
    out_size: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_upsample_nearest1d_f8e4m3(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u8>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<u8>(),
                out_size,
                layout.dims()[0] * layout.dims()[1] * out_size,
            )
        },
        "upsample_nearest1d_f8e4m3",
    )
}

pub(crate) fn upsample_nearest2d_f32(
    src: &Buffer,
    layout: &LayoutArg,
    dst: &Buffer,
    out_h: usize,
    out_w: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_upsample_nearest2d_f32(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<f32>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<f32>(),
                out_h,
                out_w,
                layout.dims()[0] * layout.dims()[1] * out_h * out_w,
            )
        },
        "upsample_nearest2d_f32",
    )
}

pub(crate) fn upsample_nearest2d_bf16(
    src: &Buffer,
    layout: &LayoutArg,
    dst: &Buffer,
    out_h: usize,
    out_w: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_upsample_nearest2d_bf16(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u16>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<u16>(),
                out_h,
                out_w,
                layout.dims()[0] * layout.dims()[1] * out_h * out_w,
            )
        },
        "upsample_nearest2d_bf16",
    )
}

pub(crate) fn upsample_nearest2d_f8e4m3(
    src: &Buffer,
    layout: &LayoutArg,
    dst: &Buffer,
    out_h: usize,
    out_w: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_upsample_nearest2d_f8e4m3(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u8>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<u8>(),
                out_h,
                out_w,
                layout.dims()[0] * layout.dims()[1] * out_h * out_w,
            )
        },
        "upsample_nearest2d_f8e4m3",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn upsample_bilinear2d_f32(
    src: &Buffer,
    layout: &LayoutArg,
    dst: &Buffer,
    out_h: usize,
    out_w: usize,
    scale_h: f64,
    scale_w: f64,
    align_corners: bool,
) -> Result<()> {
    check(
        unsafe {
            hip_upsample_bilinear2d_f32(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<f32>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<f32>(),
                out_h,
                out_w,
                scale_h,
                scale_w,
                if align_corners { 1 } else { 0 },
                layout.dims()[0] * layout.dims()[1] * out_h * out_w,
            )
        },
        "upsample_bilinear2d_f32",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn upsample_bilinear2d_bf16(
    src: &Buffer,
    layout: &LayoutArg,
    dst: &Buffer,
    out_h: usize,
    out_w: usize,
    scale_h: f64,
    scale_w: f64,
    align_corners: bool,
) -> Result<()> {
    check(
        unsafe {
            hip_upsample_bilinear2d_bf16(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u16>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<u16>(),
                out_h,
                out_w,
                scale_h,
                scale_w,
                if align_corners { 1 } else { 0 },
                layout.dims()[0] * layout.dims()[1] * out_h * out_w,
            )
        },
        "upsample_bilinear2d_bf16",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn upsample_bilinear2d_f8e4m3(
    src: &Buffer,
    layout: &LayoutArg,
    dst: &Buffer,
    out_h: usize,
    out_w: usize,
    scale_h: f64,
    scale_w: f64,
    align_corners: bool,
) -> Result<()> {
    check(
        unsafe {
            hip_upsample_bilinear2d_f8e4m3(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u8>(),
                layout.dims().as_ptr(),
                layout.stride().as_ptr(),
                layout.dims().len(),
                layout.start_offset(),
                dst.device_ptr().cast::<u8>(),
                out_h,
                out_w,
                scale_h,
                scale_w,
                if align_corners { 1 } else { 0 },
                layout.dims()[0] * layout.dims()[1] * out_h * out_w,
            )
        },
        "upsample_bilinear2d_f8e4m3",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn conv1d_f32(
    src: &Buffer,
    src_layout: &LayoutArg,
    kernel: &Buffer,
    kernel_layout: &LayoutArg,
    dst: &Buffer,
    padding: usize,
    stride: usize,
    dilation: usize,
    l_out: usize,
    elem_count: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_conv1d_f32(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<f32>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                kernel.device_ptr().cast::<f32>(),
                kernel_layout.dims().as_ptr(),
                kernel_layout.stride().as_ptr(),
                kernel_layout.dims().len(),
                kernel_layout.start_offset(),
                dst.device_ptr().cast::<f32>(),
                padding,
                stride,
                dilation,
                l_out,
                elem_count,
            )
        },
        "conv1d_f32",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_transpose1d_f32(
    src: &Buffer,
    src_layout: &LayoutArg,
    kernel: &Buffer,
    kernel_layout: &LayoutArg,
    dst: &Buffer,
    padding: usize,
    stride: usize,
    dilation: usize,
    l_out: usize,
    elem_count: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_conv_transpose1d_f32(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<f32>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                kernel.device_ptr().cast::<f32>(),
                kernel_layout.dims().as_ptr(),
                kernel_layout.stride().as_ptr(),
                kernel_layout.dims().len(),
                kernel_layout.start_offset(),
                dst.device_ptr().cast::<f32>(),
                padding,
                stride,
                dilation,
                l_out,
                elem_count,
            )
        },
        "conv_transpose1d_f32",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_f32(
    src: &Buffer,
    src_layout: &LayoutArg,
    kernel: &Buffer,
    kernel_layout: &LayoutArg,
    dst: &Buffer,
    padding: usize,
    stride: usize,
    dilation: usize,
    out_h: usize,
    out_w: usize,
    elem_count: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_conv2d_f32(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<f32>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                kernel.device_ptr().cast::<f32>(),
                kernel_layout.dims().as_ptr(),
                kernel_layout.stride().as_ptr(),
                kernel_layout.dims().len(),
                kernel_layout.start_offset(),
                dst.device_ptr().cast::<f32>(),
                padding,
                stride,
                dilation,
                out_h,
                out_w,
                elem_count,
            )
        },
        "conv2d_f32",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_transpose2d_f32(
    src: &Buffer,
    src_layout: &LayoutArg,
    kernel: &Buffer,
    kernel_layout: &LayoutArg,
    dst: &Buffer,
    padding: usize,
    stride: usize,
    dilation: usize,
    out_h: usize,
    out_w: usize,
    elem_count: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_conv_transpose2d_f32(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<f32>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                kernel.device_ptr().cast::<f32>(),
                kernel_layout.dims().as_ptr(),
                kernel_layout.stride().as_ptr(),
                kernel_layout.dims().len(),
                kernel_layout.start_offset(),
                dst.device_ptr().cast::<f32>(),
                padding,
                stride,
                dilation,
                out_h,
                out_w,
                elem_count,
            )
        },
        "conv_transpose2d_f32",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn conv1d_bf16(
    src: &Buffer,
    src_layout: &LayoutArg,
    kernel: &Buffer,
    kernel_layout: &LayoutArg,
    dst: &Buffer,
    padding: usize,
    stride: usize,
    dilation: usize,
    l_out: usize,
    elem_count: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_conv1d_bf16(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u16>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                kernel.device_ptr().cast::<u16>(),
                kernel_layout.dims().as_ptr(),
                kernel_layout.stride().as_ptr(),
                kernel_layout.dims().len(),
                kernel_layout.start_offset(),
                dst.device_ptr().cast::<u16>(),
                padding,
                stride,
                dilation,
                l_out,
                elem_count,
            )
        },
        "conv1d_bf16",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_transpose1d_bf16(
    src: &Buffer,
    src_layout: &LayoutArg,
    kernel: &Buffer,
    kernel_layout: &LayoutArg,
    dst: &Buffer,
    padding: usize,
    stride: usize,
    dilation: usize,
    l_out: usize,
    elem_count: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_conv_transpose1d_bf16(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u16>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                kernel.device_ptr().cast::<u16>(),
                kernel_layout.dims().as_ptr(),
                kernel_layout.stride().as_ptr(),
                kernel_layout.dims().len(),
                kernel_layout.start_offset(),
                dst.device_ptr().cast::<u16>(),
                padding,
                stride,
                dilation,
                l_out,
                elem_count,
            )
        },
        "conv_transpose1d_bf16",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_bf16(
    src: &Buffer,
    src_layout: &LayoutArg,
    kernel: &Buffer,
    kernel_layout: &LayoutArg,
    dst: &Buffer,
    padding: usize,
    stride: usize,
    dilation: usize,
    out_h: usize,
    out_w: usize,
    elem_count: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_conv2d_bf16(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u16>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                kernel.device_ptr().cast::<u16>(),
                kernel_layout.dims().as_ptr(),
                kernel_layout.stride().as_ptr(),
                kernel_layout.dims().len(),
                kernel_layout.start_offset(),
                dst.device_ptr().cast::<u16>(),
                padding,
                stride,
                dilation,
                out_h,
                out_w,
                elem_count,
            )
        },
        "conv2d_bf16",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_transpose2d_bf16(
    src: &Buffer,
    src_layout: &LayoutArg,
    kernel: &Buffer,
    kernel_layout: &LayoutArg,
    dst: &Buffer,
    padding: usize,
    stride: usize,
    dilation: usize,
    out_h: usize,
    out_w: usize,
    elem_count: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_conv_transpose2d_bf16(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u16>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                kernel.device_ptr().cast::<u16>(),
                kernel_layout.dims().as_ptr(),
                kernel_layout.stride().as_ptr(),
                kernel_layout.dims().len(),
                kernel_layout.start_offset(),
                dst.device_ptr().cast::<u16>(),
                padding,
                stride,
                dilation,
                out_h,
                out_w,
                elem_count,
            )
        },
        "conv_transpose2d_bf16",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn conv1d_f8e4m3(
    src: &Buffer,
    src_layout: &LayoutArg,
    kernel: &Buffer,
    kernel_layout: &LayoutArg,
    dst: &Buffer,
    padding: usize,
    stride: usize,
    dilation: usize,
    l_out: usize,
    elem_count: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_conv1d_f8e4m3(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u8>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                kernel.device_ptr().cast::<u8>(),
                kernel_layout.dims().as_ptr(),
                kernel_layout.stride().as_ptr(),
                kernel_layout.dims().len(),
                kernel_layout.start_offset(),
                dst.device_ptr().cast::<u8>(),
                padding,
                stride,
                dilation,
                l_out,
                elem_count,
            )
        },
        "conv1d_f8e4m3",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_transpose1d_f8e4m3(
    src: &Buffer,
    src_layout: &LayoutArg,
    kernel: &Buffer,
    kernel_layout: &LayoutArg,
    dst: &Buffer,
    padding: usize,
    stride: usize,
    dilation: usize,
    l_out: usize,
    elem_count: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_conv_transpose1d_f8e4m3(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u8>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                kernel.device_ptr().cast::<u8>(),
                kernel_layout.dims().as_ptr(),
                kernel_layout.stride().as_ptr(),
                kernel_layout.dims().len(),
                kernel_layout.start_offset(),
                dst.device_ptr().cast::<u8>(),
                padding,
                stride,
                dilation,
                l_out,
                elem_count,
            )
        },
        "conv_transpose1d_f8e4m3",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_f8e4m3(
    src: &Buffer,
    src_layout: &LayoutArg,
    kernel: &Buffer,
    kernel_layout: &LayoutArg,
    dst: &Buffer,
    padding: usize,
    stride: usize,
    dilation: usize,
    out_h: usize,
    out_w: usize,
    elem_count: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_conv2d_f8e4m3(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u8>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                kernel.device_ptr().cast::<u8>(),
                kernel_layout.dims().as_ptr(),
                kernel_layout.stride().as_ptr(),
                kernel_layout.dims().len(),
                kernel_layout.start_offset(),
                dst.device_ptr().cast::<u8>(),
                padding,
                stride,
                dilation,
                out_h,
                out_w,
                elem_count,
            )
        },
        "conv2d_f8e4m3",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_transpose2d_f8e4m3(
    src: &Buffer,
    src_layout: &LayoutArg,
    kernel: &Buffer,
    kernel_layout: &LayoutArg,
    dst: &Buffer,
    padding: usize,
    stride: usize,
    dilation: usize,
    out_h: usize,
    out_w: usize,
    elem_count: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_conv_transpose2d_f8e4m3(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u8>(),
                src_layout.dims().as_ptr(),
                src_layout.stride().as_ptr(),
                src_layout.dims().len(),
                src_layout.start_offset(),
                kernel.device_ptr().cast::<u8>(),
                kernel_layout.dims().as_ptr(),
                kernel_layout.stride().as_ptr(),
                kernel_layout.dims().len(),
                kernel_layout.start_offset(),
                dst.device_ptr().cast::<u8>(),
                padding,
                stride,
                dilation,
                out_h,
                out_w,
                elem_count,
            )
        },
        "conv_transpose2d_f8e4m3",
    )
}

pub(crate) fn arg_sort_f32(
    src: &Buffer,
    layout: &LayoutArg,
    dst: &Buffer,
    asc: bool,
    last_dim: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_arg_sort_f32(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<f32>(),
                layout.start_offset(),
                dst.device_ptr().cast::<u32>(),
                layout.elem_count(),
                last_dim,
                if asc { 1 } else { 0 },
            )
        },
        "arg_sort_f32",
    )
}

pub(crate) fn arg_sort_bf16(
    src: &Buffer,
    layout: &LayoutArg,
    dst: &Buffer,
    asc: bool,
    last_dim: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_arg_sort_bf16(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u16>(),
                layout.start_offset(),
                dst.device_ptr().cast::<u32>(),
                layout.elem_count(),
                last_dim,
                if asc { 1 } else { 0 },
            )
        },
        "arg_sort_bf16",
    )
}

pub(crate) fn arg_sort_f8e4m3(
    src: &Buffer,
    layout: &LayoutArg,
    dst: &Buffer,
    asc: bool,
    last_dim: usize,
) -> Result<()> {
    check(
        unsafe {
            hip_arg_sort_f8e4m3(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u8>(),
                layout.start_offset(),
                dst.device_ptr().cast::<u32>(),
                layout.elem_count(),
                last_dim,
                if asc { 1 } else { 0 },
            )
        },
        "arg_sort_f8e4m3",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rms_norm_f32(
    src: &Buffer,
    src_start_offset: usize,
    alpha: &Buffer,
    alpha_start_offset: usize,
    dst: &Buffer,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<()> {
    check(
        unsafe {
            hip_rms_norm_f32(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<f32>(),
                src_start_offset,
                alpha.device_ptr().cast::<f32>(),
                alpha_start_offset,
                dst.device_ptr().cast::<f32>(),
                rows,
                cols,
                eps,
            )
        },
        "rms_norm_f32",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rms_norm_bf16(
    src: &Buffer,
    src_start_offset: usize,
    alpha: &Buffer,
    alpha_start_offset: usize,
    dst: &Buffer,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<()> {
    check(
        unsafe {
            hip_rms_norm_bf16(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u16>(),
                src_start_offset,
                alpha.device_ptr().cast::<u16>(),
                alpha_start_offset,
                dst.device_ptr().cast::<u16>(),
                rows,
                cols,
                eps,
            )
        },
        "rms_norm_bf16",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rms_norm_f8e4m3(
    src: &Buffer,
    src_start_offset: usize,
    alpha: &Buffer,
    alpha_start_offset: usize,
    dst: &Buffer,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<()> {
    check(
        unsafe {
            hip_rms_norm_f8e4m3(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u8>(),
                src_start_offset,
                alpha.device_ptr().cast::<u8>(),
                alpha_start_offset,
                dst.device_ptr().cast::<u8>(),
                rows,
                cols,
                eps,
            )
        },
        "rms_norm_f8e4m3",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_norm_f32(
    src: &Buffer,
    src_start_offset: usize,
    alpha: &Buffer,
    alpha_start_offset: usize,
    beta: &Buffer,
    beta_start_offset: usize,
    dst: &Buffer,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<()> {
    check(
        unsafe {
            hip_layer_norm_f32(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<f32>(),
                src_start_offset,
                alpha.device_ptr().cast::<f32>(),
                alpha_start_offset,
                beta.device_ptr().cast::<f32>(),
                beta_start_offset,
                dst.device_ptr().cast::<f32>(),
                rows,
                cols,
                eps,
            )
        },
        "layer_norm_f32",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_norm_bf16(
    src: &Buffer,
    src_start_offset: usize,
    alpha: &Buffer,
    alpha_start_offset: usize,
    beta: &Buffer,
    beta_start_offset: usize,
    dst: &Buffer,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<()> {
    check(
        unsafe {
            hip_layer_norm_bf16(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u16>(),
                src_start_offset,
                alpha.device_ptr().cast::<u16>(),
                alpha_start_offset,
                beta.device_ptr().cast::<u16>(),
                beta_start_offset,
                dst.device_ptr().cast::<u16>(),
                rows,
                cols,
                eps,
            )
        },
        "layer_norm_bf16",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_norm_f8e4m3(
    src: &Buffer,
    src_start_offset: usize,
    alpha: &Buffer,
    alpha_start_offset: usize,
    beta: &Buffer,
    beta_start_offset: usize,
    dst: &Buffer,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<()> {
    check(
        unsafe {
            hip_layer_norm_f8e4m3(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u8>(),
                src_start_offset,
                alpha.device_ptr().cast::<u8>(),
                alpha_start_offset,
                beta.device_ptr().cast::<u8>(),
                beta_start_offset,
                dst.device_ptr().cast::<u8>(),
                rows,
                cols,
                eps,
            )
        },
        "layer_norm_f8e4m3",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rope_f32(
    src: &Buffer,
    src_start_offset: usize,
    cos: &Buffer,
    cos_start_offset: usize,
    sin: &Buffer,
    sin_start_offset: usize,
    dst: &Buffer,
    b: usize,
    h: usize,
    t: usize,
    d: usize,
    interleaved: bool,
    unbatched_rope: bool,
    thd: bool,
) -> Result<()> {
    check(
        unsafe {
            hip_rope_f32(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<f32>(),
                src_start_offset,
                cos.device_ptr().cast::<f32>(),
                cos_start_offset,
                sin.device_ptr().cast::<f32>(),
                sin_start_offset,
                dst.device_ptr().cast::<f32>(),
                b,
                h,
                t,
                d,
                if interleaved { 1 } else { 0 },
                if unbatched_rope { 1 } else { 0 },
                if thd { 1 } else { 0 },
            )
        },
        "rope_f32",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rope_bf16(
    src: &Buffer,
    src_start_offset: usize,
    cos: &Buffer,
    cos_start_offset: usize,
    sin: &Buffer,
    sin_start_offset: usize,
    dst: &Buffer,
    b: usize,
    h: usize,
    t: usize,
    d: usize,
    interleaved: bool,
    unbatched_rope: bool,
    thd: bool,
) -> Result<()> {
    check(
        unsafe {
            hip_rope_bf16(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u16>(),
                src_start_offset,
                cos.device_ptr().cast::<u16>(),
                cos_start_offset,
                sin.device_ptr().cast::<u16>(),
                sin_start_offset,
                dst.device_ptr().cast::<u16>(),
                b,
                h,
                t,
                d,
                if interleaved { 1 } else { 0 },
                if unbatched_rope { 1 } else { 0 },
                if thd { 1 } else { 0 },
            )
        },
        "rope_bf16",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rope_f8e4m3(
    src: &Buffer,
    src_start_offset: usize,
    cos: &Buffer,
    cos_start_offset: usize,
    sin: &Buffer,
    sin_start_offset: usize,
    dst: &Buffer,
    b: usize,
    h: usize,
    t: usize,
    d: usize,
    interleaved: bool,
    unbatched_rope: bool,
    thd: bool,
) -> Result<()> {
    check(
        unsafe {
            hip_rope_f8e4m3(
                ordinal_to_c_int(src.device_ordinal())?,
                src.device_ptr().cast::<u8>(),
                src_start_offset,
                cos.device_ptr().cast::<u8>(),
                cos_start_offset,
                sin.device_ptr().cast::<u8>(),
                sin_start_offset,
                dst.device_ptr().cast::<u8>(),
                b,
                h,
                t,
                d,
                if interleaved { 1 } else { 0 },
                if unbatched_rope { 1 } else { 0 },
                if thd { 1 } else { 0 },
            )
        },
        "rope_f8e4m3",
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
