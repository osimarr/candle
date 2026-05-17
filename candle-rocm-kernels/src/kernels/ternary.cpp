#include "common.h"

namespace {

__global__ void where_u8_f32_kernel(
    const uint8_t* cond,
    const size_t* cond_dims,
    const size_t* cond_strides,
    size_t cond_rank,
    size_t cond_start_offset,
    const float* on_true,
    const size_t* true_dims,
    const size_t* true_strides,
    size_t true_rank,
    size_t true_start_offset,
    const float* on_false,
    const size_t* false_dims,
    const size_t* false_strides,
    size_t false_rank,
    size_t false_start_offset,
    float* dst,
    size_t elem_count) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }
    const size_t cond_index = storage_index(
        logical_index,
        cond_dims,
        cond_strides,
        cond_rank,
        cond_start_offset);
    const size_t true_index = storage_index(
        logical_index,
        true_dims,
        true_strides,
        true_rank,
        true_start_offset);
    const size_t false_index = storage_index(
        logical_index,
        false_dims,
        false_strides,
        false_rank,
        false_start_offset);
    dst[logical_index] = cond[cond_index] != 0 ? on_true[true_index] : on_false[false_index];
}

} // namespace

extern "C" int hip_where_u8_f32(
    int ordinal,
    const uint8_t* cond,
    const size_t* cond_dims,
    const size_t* cond_strides,
    size_t cond_rank,
    size_t cond_start_offset,
    const float* on_true,
    const size_t* true_dims,
    const size_t* true_strides,
    size_t true_rank,
    size_t true_start_offset,
    const float* on_false,
    const size_t* false_dims,
    const size_t* false_strides,
    size_t false_rank,
    size_t false_start_offset,
    float* dst,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    DeviceLayout cond_layout;
    rc = cond_layout.init(cond_dims, cond_strides, cond_rank);
    if (rc != 0) {
        return rc;
    }
    DeviceLayout true_layout;
    rc = true_layout.init(true_dims, true_strides, true_rank);
    if (rc != 0) {
        return rc;
    }
    DeviceLayout false_layout;
    rc = false_layout.init(false_dims, false_strides, false_rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        "where_u8_f32",
        elem_count,
        where_u8_f32_kernel,
        cond,
        cond_layout.dims,
        cond_layout.strides,
        cond_layout.rank,
        cond_start_offset,
        on_true,
        true_layout.dims,
        true_layout.strides,
        true_layout.rank,
        true_start_offset,
        on_false,
        false_layout.dims,
        false_layout.strides,
        false_layout.rank,
        false_start_offset,
        dst,
        elem_count);
}
