#include "common.h"

namespace {

__global__ void cast_f32_to_bf16_kernel(
    const float* src,
    DeviceLayout layout,
    size_t start_offset,
    uint16_t* dst,
    size_t elem_count) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }
    const size_t src_index = storage_index(logical_index, layout, start_offset);
    dst[logical_index] = f32_to_bf16_bits(src[src_index]);
}

__global__ void cast_bf16_to_f32_kernel(
    const uint16_t* src,
    DeviceLayout layout,
    size_t start_offset,
    float* dst,
    size_t elem_count) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }
    const size_t src_index = storage_index(logical_index, layout, start_offset);
    dst[logical_index] = bf16_bits_to_f32(src[src_index]);
}

__global__ void cast_f32_to_f16_kernel(
    const float* src,
    DeviceLayout layout,
    size_t start_offset,
    uint16_t* dst,
    size_t elem_count) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }
    const size_t src_index = storage_index(logical_index, layout, start_offset);
    dst[logical_index] = f32_to_f16_bits(src[src_index]);
}

__global__ void cast_f16_to_f32_kernel(
    const uint16_t* src,
    DeviceLayout layout,
    size_t start_offset,
    float* dst,
    size_t elem_count) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }
    const size_t src_index = storage_index(logical_index, layout, start_offset);
    dst[logical_index] = f16_bits_to_f32(src[src_index]);
}

__global__ void cast_f32_to_f8e4m3_kernel(
    const float* src,
    DeviceLayout layout,
    size_t start_offset,
    uint8_t* dst,
    size_t elem_count) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }
    const size_t src_index = storage_index(logical_index, layout, start_offset);
    dst[logical_index] = f32_to_f8e4m3_bits(src[src_index]);
}

__global__ void cast_f8e4m3_to_f32_kernel(
    const uint8_t* src,
    DeviceLayout layout,
    size_t start_offset,
    float* dst,
    size_t elem_count) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }
    const size_t src_index = storage_index(logical_index, layout, start_offset);
    dst[logical_index] = f8e4m3_bits_to_f32(src[src_index]);
}

} // namespace

extern "C" int hip_cast_f32_to_bf16(
    int ordinal,
    const float* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    uint16_t* dst,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    DeviceLayout layout;
    rc = layout.init(dims, strides, rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        "cast_f32_to_bf16",
        elem_count,
        cast_f32_to_bf16_kernel,
        src,
        layout,
        start_offset,
        dst,
        elem_count);
}

extern "C" int hip_cast_bf16_to_f32(
    int ordinal,
    const uint16_t* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    float* dst,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    DeviceLayout layout;
    rc = layout.init(dims, strides, rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        "cast_bf16_to_f32",
        elem_count,
        cast_bf16_to_f32_kernel,
        src,
        layout,
        start_offset,
        dst,
        elem_count);
}

extern "C" int hip_cast_f32_to_f16(
    int ordinal,
    const float* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    uint16_t* dst,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    DeviceLayout layout;
    rc = layout.init(dims, strides, rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        "cast_f32_to_f16",
        elem_count,
        cast_f32_to_f16_kernel,
        src,
        layout,
        start_offset,
        dst,
        elem_count);
}

extern "C" int hip_cast_f16_to_f32(
    int ordinal,
    const uint16_t* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    float* dst,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    DeviceLayout layout;
    rc = layout.init(dims, strides, rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        "cast_f16_to_f32",
        elem_count,
        cast_f16_to_f32_kernel,
        src,
        layout,
        start_offset,
        dst,
        elem_count);
}

extern "C" int hip_cast_f32_to_f8e4m3(
    int ordinal,
    const float* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    uint8_t* dst,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    DeviceLayout layout;
    rc = layout.init(dims, strides, rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        "cast_f32_to_f8e4m3",
        elem_count,
        cast_f32_to_f8e4m3_kernel,
        src,
        layout,
        start_offset,
        dst,
        elem_count);
}

extern "C" int hip_cast_f8e4m3_to_f32(
    int ordinal,
    const uint8_t* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    float* dst,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    DeviceLayout layout;
    rc = layout.init(dims, strides, rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        "cast_f8e4m3_to_f32",
        elem_count,
        cast_f8e4m3_to_f32_kernel,
        src,
        layout,
        start_offset,
        dst,
        elem_count);
}
