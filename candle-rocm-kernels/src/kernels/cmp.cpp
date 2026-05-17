#include "common.h"

namespace {

__device__ uint8_t cmp_value(int op, float lhs, float rhs) {
    switch (op) {
    case 1:
        return lhs == rhs;
    case 2:
        return lhs >= rhs;
    case 3:
        return lhs > rhs;
    case 4:
        return lhs <= rhs;
    case 5:
        return lhs < rhs;
    case 6:
        return lhs != rhs;
    default:
        return 0;
    }
}

__global__ void cmp_f32_kernel(
    int op,
    const float* lhs,
    const size_t* lhs_dims,
    const size_t* lhs_strides,
    size_t lhs_rank,
    size_t lhs_start_offset,
    const float* rhs,
    const size_t* rhs_dims,
    const size_t* rhs_strides,
    size_t rhs_rank,
    size_t rhs_start_offset,
    uint8_t* dst,
    size_t elem_count) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }
    const size_t lhs_index = storage_index(
        logical_index,
        lhs_dims,
        lhs_strides,
        lhs_rank,
        lhs_start_offset);
    const size_t rhs_index = storage_index(
        logical_index,
        rhs_dims,
        rhs_strides,
        rhs_rank,
        rhs_start_offset);
    dst[logical_index] = cmp_value(op, lhs[lhs_index], rhs[rhs_index]);
}

} // namespace

extern "C" int hip_cmp_f32(
    int ordinal,
    int op,
    const float* lhs,
    const size_t* lhs_dims,
    const size_t* lhs_strides,
    size_t lhs_rank,
    size_t lhs_start_offset,
    const float* rhs,
    const size_t* rhs_dims,
    const size_t* rhs_strides,
    size_t rhs_rank,
    size_t rhs_start_offset,
    uint8_t* dst,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    DeviceLayout lhs_layout;
    rc = lhs_layout.init(lhs_dims, lhs_strides, lhs_rank);
    if (rc != 0) {
        return rc;
    }
    DeviceLayout rhs_layout;
    rc = rhs_layout.init(rhs_dims, rhs_strides, rhs_rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        "cmp_f32",
        elem_count,
        cmp_f32_kernel,
        op,
        lhs,
        lhs_layout.dims,
        lhs_layout.strides,
        lhs_layout.rank,
        lhs_start_offset,
        rhs,
        rhs_layout.dims,
        rhs_layout.strides,
        rhs_layout.rank,
        rhs_start_offset,
        dst,
        elem_count);
}
