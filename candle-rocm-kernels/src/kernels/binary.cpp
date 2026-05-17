#include "common.h"

#include <cmath>

namespace {

__device__ float binary_value(int op, float lhs, float rhs) {
    switch (op) {
    case 1:
        return lhs + rhs;
    case 2:
        return lhs / rhs;
    case 3:
        return fmaxf(lhs, rhs);
    case 4:
        return fminf(lhs, rhs);
    case 5:
        return lhs * rhs;
    case 6:
        return lhs - rhs;
    default:
        return lhs;
    }
}

__global__ void binary_f32_kernel(
    int op,
    const float* lhs,
    DeviceLayout lhs_layout,
    size_t lhs_start_offset,
    const float* rhs,
    DeviceLayout rhs_layout,
    size_t rhs_start_offset,
    float* dst,
    size_t elem_count) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }
    const size_t lhs_index = storage_index(
        logical_index,
        lhs_layout,
        lhs_start_offset);
    const size_t rhs_index = storage_index(
        logical_index,
        rhs_layout,
        rhs_start_offset);
    dst[logical_index] = binary_value(op, lhs[lhs_index], rhs[rhs_index]);
}

__global__ void binary_bf16_kernel(
    int op,
    const uint16_t* lhs,
    DeviceLayout lhs_layout,
    size_t lhs_start_offset,
    const uint16_t* rhs,
    DeviceLayout rhs_layout,
    size_t rhs_start_offset,
    uint16_t* dst,
    size_t elem_count) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }
    const size_t lhs_index = storage_index(
        logical_index,
        lhs_layout,
        lhs_start_offset);
    const size_t rhs_index = storage_index(
        logical_index,
        rhs_layout,
        rhs_start_offset);
    const float value =
        binary_value(op, bf16_bits_to_f32(lhs[lhs_index]), bf16_bits_to_f32(rhs[rhs_index]));
    dst[logical_index] = f32_to_bf16_bits(value);
}

__global__ void binary_f8e4m3_kernel(
    int op,
    const uint8_t* lhs,
    DeviceLayout lhs_layout,
    size_t lhs_start_offset,
    const uint8_t* rhs,
    DeviceLayout rhs_layout,
    size_t rhs_start_offset,
    uint8_t* dst,
    size_t elem_count) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }
    const size_t lhs_index = storage_index(
        logical_index,
        lhs_layout,
        lhs_start_offset);
    const size_t rhs_index = storage_index(
        logical_index,
        rhs_layout,
        rhs_start_offset);
    const float value = binary_value(
        op,
        f8e4m3_bits_to_f32(lhs[lhs_index]),
        f8e4m3_bits_to_f32(rhs[rhs_index]));
    dst[logical_index] = f32_to_f8e4m3_bits(value);
}

} // namespace

extern "C" int hip_binary_f32(
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
    float* dst,
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
        "binary_f32",
        elem_count,
        binary_f32_kernel,
        op,
        lhs,
        lhs_layout,
        lhs_start_offset,
        rhs,
        rhs_layout,
        rhs_start_offset,
        dst,
        elem_count);
}

extern "C" int hip_binary_bf16(
    int ordinal,
    int op,
    const uint16_t* lhs,
    const size_t* lhs_dims,
    const size_t* lhs_strides,
    size_t lhs_rank,
    size_t lhs_start_offset,
    const uint16_t* rhs,
    const size_t* rhs_dims,
    const size_t* rhs_strides,
    size_t rhs_rank,
    size_t rhs_start_offset,
    uint16_t* dst,
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
        "binary_bf16",
        elem_count,
        binary_bf16_kernel,
        op,
        lhs,
        lhs_layout,
        lhs_start_offset,
        rhs,
        rhs_layout,
        rhs_start_offset,
        dst,
        elem_count);
}

extern "C" int hip_binary_f8e4m3(
    int ordinal,
    int op,
    const uint8_t* lhs,
    const size_t* lhs_dims,
    const size_t* lhs_strides,
    size_t lhs_rank,
    size_t lhs_start_offset,
    const uint8_t* rhs,
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
        "binary_f8e4m3",
        elem_count,
        binary_f8e4m3_kernel,
        op,
        lhs,
        lhs_layout,
        lhs_start_offset,
        rhs,
        rhs_layout,
        rhs_start_offset,
        dst,
        elem_count);
}
