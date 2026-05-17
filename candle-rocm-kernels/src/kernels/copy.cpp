#include "common.h"

namespace {

__global__ void copy_strided_src_kernel(
    const uint8_t* src,
    const size_t* src_dims,
    const size_t* src_strides,
    size_t src_rank,
    size_t src_start_offset,
    uint8_t* dst,
    size_t dst_offset,
    size_t elem_size,
    size_t elem_count) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }
    const size_t src_index = storage_index(
        logical_index,
        src_dims,
        src_strides,
        src_rank,
        src_start_offset);
    const size_t dst_index = dst_offset + logical_index;
    for (size_t byte = 0; byte < elem_size; ++byte) {
        dst[dst_index * elem_size + byte] = src[src_index * elem_size + byte];
    }
}

__global__ void copy2d_kernel(
    const uint8_t* src,
    uint8_t* dst,
    size_t d1,
    size_t d2,
    size_t src_stride1,
    size_t dst_stride1,
    size_t src_offset,
    size_t dst_offset,
    size_t elem_size) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t elem_count = d1 * d2;
    if (index >= elem_count) {
        return;
    }
    const size_t i = index / d2;
    const size_t j = index - i * d2;
    const size_t src_index = src_offset + i * src_stride1 + j;
    const size_t dst_index = dst_offset + i * dst_stride1 + j;
    for (size_t byte = 0; byte < elem_size; ++byte) {
        dst[dst_index * elem_size + byte] = src[src_index * elem_size + byte];
    }
}

} // namespace

extern "C" int hip_copy_strided_src(
    int ordinal,
    const uint8_t* src,
    const size_t* src_dims,
    const size_t* src_strides,
    size_t src_rank,
    size_t src_start_offset,
    uint8_t* dst,
    size_t dst_offset,
    size_t elem_size,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    DeviceLayout src_layout;
    rc = src_layout.init(src_dims, src_strides, src_rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        "copy_strided_src",
        elem_count,
        copy_strided_src_kernel,
        src,
        src_layout.dims,
        src_layout.strides,
        src_layout.rank,
        src_start_offset,
        dst,
        dst_offset,
        elem_size,
        elem_count);
}

extern "C" int hip_copy2d(
    int ordinal,
    const uint8_t* src,
    uint8_t* dst,
    size_t d1,
    size_t d2,
    size_t src_stride1,
    size_t dst_stride1,
    size_t src_offset,
    size_t dst_offset,
    size_t elem_size) {
    const size_t elem_count = d1 * d2;
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    return launch_1d(
        "copy2d",
        elem_count,
        copy2d_kernel,
        src,
        dst,
        d1,
        d2,
        src_stride1,
        dst_stride1,
        src_offset,
        dst_offset,
        elem_size);
}
