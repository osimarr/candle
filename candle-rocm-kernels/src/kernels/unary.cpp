#include "common.h"

#include <cmath>

namespace {

__device__ float unary_value(int op, float value) {
    switch (op) {
    case 1:
        return fabsf(value);
    case 2:
        return ceilf(value);
    case 3:
        return cosf(value);
    case 4:
        return expf(value);
    case 5:
        return floorf(value);
    case 6:
        return logf(value);
    case 7:
        return -value;
    case 8:
        return 1.0f / value;
    case 9:
        return fmaxf(value, 0.0f);
    case 10:
        return roundf(value);
    case 11:
        return sinf(value);
    case 12:
        return value * value;
    case 13:
        return sqrtf(value);
    case 14:
        return tanhf(value);
    case 15:
        return value / (1.0f + expf(-value));
    case 16:
        return 0.5f * value *
               (1.0f + tanhf(0.7978845608028654f * value *
                              (1.0f + 0.044715f * value * value)));
    case 17:
        return erff(value);
    case 18:
        return 0.5f * value * (1.0f + erff(value * 0.7071067811865475f));
    case 19:
        return (0.0f < value) - (value < 0.0f);
    case 20:
        return 1.0f / (1.0f + expf(-value));
    default:
        return value;
    }
}

__global__ void unary_f32_kernel(
    int op,
    const float* src,
    DeviceLayout layout,
    size_t start_offset,
    float* dst,
    size_t elem_count) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }
    const size_t src_index = storage_index(logical_index, layout, start_offset);
    dst[logical_index] = unary_value(op, src[src_index]);
}

} // namespace

extern "C" int hip_unary_f32(
    int ordinal,
    int op,
    const float* src,
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
    DeviceLayout src_layout;
    rc = src_layout.init(dims, strides, rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        "unary_f32",
        elem_count,
        unary_f32_kernel,
        op,
        src,
        src_layout,
        start_offset,
        dst,
        elem_count);
}
