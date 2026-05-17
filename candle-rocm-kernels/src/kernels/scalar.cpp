#include "common.h"

#include <cmath>

namespace {

__global__ void affine_f32_kernel(
    const float* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    float* dst,
    size_t elem_count,
    float mul,
    float add) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }
    const size_t src_index =
        storage_index(logical_index, dims, strides, rank, start_offset);
    dst[logical_index] = src[src_index] * mul + add;
}

__global__ void powf_f32_kernel(
    const float* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    float* dst,
    size_t elem_count,
    float value) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }
    const size_t src_index =
        storage_index(logical_index, dims, strides, rank, start_offset);
    dst[logical_index] = powf(src[src_index], value);
}

__global__ void elu_f32_kernel(
    const float* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    float* dst,
    size_t elem_count,
    float alpha) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }
    const size_t src_index =
        storage_index(logical_index, dims, strides, rank, start_offset);
    const float value = src[src_index];
    dst[logical_index] = value > 0.0f ? value : alpha * (expf(value) - 1.0f);
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
        affine_f32_kernel,
        src,
        layout.dims,
        layout.strides,
        layout.rank,
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
        powf_f32_kernel,
        src,
        layout.dims,
        layout.strides,
        layout.rank,
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
        elu_f32_kernel,
        src,
        layout.dims,
        layout.strides,
        layout.rank,
        start_offset,
        dst,
        elem_count,
        alpha);
}
