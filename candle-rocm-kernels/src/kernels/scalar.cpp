#include "common.h"

#include <cmath>

namespace {

template <typename T>
__global__ void affine_kernel(
    const T* src,
    DeviceLayout layout,
    size_t start_offset,
    T* dst,
    size_t elem_count,
    float mul,
    float add) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }
    const size_t src_index = storage_index(logical_index, layout, start_offset);
    dst[logical_index] = from_f32<T>(to_f32(src[src_index]) * mul + add);
}

template <typename T>
__global__ void powf_kernel(
    const T* src,
    DeviceLayout layout,
    size_t start_offset,
    T* dst,
    size_t elem_count,
    float value) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }
    const size_t src_index = storage_index(logical_index, layout, start_offset);
    dst[logical_index] = from_f32<T>(powf(to_f32(src[src_index]), value));
}

template <typename T>
__global__ void elu_kernel(
    const T* src,
    DeviceLayout layout,
    size_t start_offset,
    T* dst,
    size_t elem_count,
    float alpha) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }
    const size_t src_index = storage_index(logical_index, layout, start_offset);
    const float value = to_f32(src[src_index]);
    dst[logical_index] = from_f32<T>(value > 0.0f ? value : alpha * (expf(value) - 1.0f));
}

} // namespace

extern "C" int hip_affine_f32(
    int ordinal,
    const float* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    float* dst,
    size_t elem_count,
    float mul,
    float add) {
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
        "affine_f32",
        elem_count,
        affine_kernel<float>,
        src,
        layout,
        start_offset,
        dst,
        elem_count,
        mul,
        add);
}

extern "C" int hip_powf_f32(
    int ordinal,
    const float* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    float* dst,
    size_t elem_count,
    float value) {
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
        "powf_f32",
        elem_count,
        powf_kernel<float>,
        src,
        layout,
        start_offset,
        dst,
        elem_count,
        value);
}

extern "C" int hip_elu_f32(
    int ordinal,
    const float* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    float* dst,
    size_t elem_count,
    float alpha) {
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
        "elu_f32",
        elem_count,
        elu_kernel<float>,
        src,
        layout,
        start_offset,
        dst,
        elem_count,
        alpha);
}

extern "C" int hip_affine_bf16(
    int ordinal,
    const uint16_t* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    uint16_t* dst,
    size_t elem_count,
    float mul,
    float add) {
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
        "affine_bf16",
        elem_count,
        affine_kernel<uint16_t>,
        src,
        layout,
        start_offset,
        dst,
        elem_count,
        mul,
        add);
}

extern "C" int hip_powf_bf16(
    int ordinal,
    const uint16_t* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    uint16_t* dst,
    size_t elem_count,
    float value) {
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
        "powf_bf16",
        elem_count,
        powf_kernel<uint16_t>,
        src,
        layout,
        start_offset,
        dst,
        elem_count,
        value);
}

extern "C" int hip_elu_bf16(
    int ordinal,
    const uint16_t* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    uint16_t* dst,
    size_t elem_count,
    float alpha) {
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
        "elu_bf16",
        elem_count,
        elu_kernel<uint16_t>,
        src,
        layout,
        start_offset,
        dst,
        elem_count,
        alpha);
}
