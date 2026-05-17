#include "common.h"

#include <cstdint>

namespace {

__device__ inline bool comes_before_asc(float lhs, size_t lhs_index, float rhs, size_t rhs_index) {
    const bool lhs_nan = isnan(lhs);
    const bool rhs_nan = isnan(rhs);
    if (lhs_nan || rhs_nan) {
        if (lhs_nan != rhs_nan) {
            return !lhs_nan;
        }
        return lhs_index < rhs_index;
    }
    return lhs < rhs || (lhs == rhs && lhs_index < rhs_index);
}

__device__ inline bool comes_before_desc(float lhs, size_t lhs_index, float rhs, size_t rhs_index) {
    const bool lhs_nan = isnan(lhs);
    const bool rhs_nan = isnan(rhs);
    if (lhs_nan || rhs_nan) {
        if (lhs_nan != rhs_nan) {
            return !lhs_nan;
        }
        return lhs_index < rhs_index;
    }
    return lhs > rhs || (lhs == rhs && lhs_index < rhs_index);
}

__global__ void arg_sort_f32_kernel(
    const float* src,
    size_t start_offset,
    uint32_t* dst,
    size_t elem_count,
    size_t last_dim,
    int asc) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elem_count) {
        return;
    }

    const size_t row_start = index - index % last_dim;
    const size_t col = index - row_start;
    const float value = src[start_offset + index];

    size_t rank = 0;
    for (size_t other_col = 0; other_col < last_dim; ++other_col) {
        if (other_col == col) {
            continue;
        }
        const float other = src[start_offset + row_start + other_col];
        const bool before = asc != 0
                                ? comes_before_asc(other, other_col, value, col)
                                : comes_before_desc(other, other_col, value, col);
        if (before) {
            ++rank;
        }
    }
    dst[row_start + rank] = static_cast<uint32_t>(col);
}

__global__ void arg_sort_bf16_kernel(
    const uint16_t* src,
    size_t start_offset,
    uint32_t* dst,
    size_t elem_count,
    size_t last_dim,
    int asc) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elem_count) {
        return;
    }

    const size_t row_start = index - index % last_dim;
    const size_t col = index - row_start;
    const float value = bf16_bits_to_f32(src[start_offset + index]);

    size_t rank = 0;
    for (size_t other_col = 0; other_col < last_dim; ++other_col) {
        if (other_col == col) {
            continue;
        }
        const float other = bf16_bits_to_f32(src[start_offset + row_start + other_col]);
        const bool before = asc != 0
                                ? comes_before_asc(other, other_col, value, col)
                                : comes_before_desc(other, other_col, value, col);
        if (before) {
            ++rank;
        }
    }
    dst[row_start + rank] = static_cast<uint32_t>(col);
}

__global__ void arg_sort_f8e4m3_kernel(
    const uint8_t* src,
    size_t start_offset,
    uint32_t* dst,
    size_t elem_count,
    size_t last_dim,
    int asc) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elem_count) {
        return;
    }

    const size_t row_start = index - index % last_dim;
    const size_t col = index - row_start;
    const float value = f8e4m3_bits_to_f32(src[start_offset + index]);

    size_t rank = 0;
    for (size_t other_col = 0; other_col < last_dim; ++other_col) {
        if (other_col == col) {
            continue;
        }
        const float other = f8e4m3_bits_to_f32(src[start_offset + row_start + other_col]);
        const bool before = asc != 0
                                ? comes_before_asc(other, other_col, value, col)
                                : comes_before_desc(other, other_col, value, col);
        if (before) {
            ++rank;
        }
    }
    dst[row_start + rank] = static_cast<uint32_t>(col);
}

} // namespace

extern "C" int hip_arg_sort_f32(
    int ordinal,
    const float* src,
    size_t start_offset,
    uint32_t* dst,
    size_t elem_count,
    size_t last_dim,
    int asc) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    if (last_dim == 0 || elem_count % last_dim != 0) {
        return set_error("arg_sort_f32 shape", hipErrorInvalidValue);
    }
    return launch_1d_async(
        "arg_sort_f32",
        elem_count,
        arg_sort_f32_kernel,
        src,
        start_offset,
        dst,
        elem_count,
        last_dim,
        asc);
}

extern "C" int hip_arg_sort_bf16(
    int ordinal,
    const uint16_t* src,
    size_t start_offset,
    uint32_t* dst,
    size_t elem_count,
    size_t last_dim,
    int asc) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    if (last_dim == 0 || elem_count % last_dim != 0) {
        return set_error("arg_sort_bf16 shape", hipErrorInvalidValue);
    }
    return launch_1d_async(
        "arg_sort_bf16",
        elem_count,
        arg_sort_bf16_kernel,
        src,
        start_offset,
        dst,
        elem_count,
        last_dim,
        asc);
}

extern "C" int hip_arg_sort_f8e4m3(
    int ordinal,
    const uint8_t* src,
    size_t start_offset,
    uint32_t* dst,
    size_t elem_count,
    size_t last_dim,
    int asc) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    if (last_dim == 0 || elem_count % last_dim != 0) {
        return set_error("arg_sort_f8e4m3 shape", hipErrorInvalidValue);
    }
    return launch_1d_async(
        "arg_sort_f8e4m3",
        elem_count,
        arg_sort_f8e4m3_kernel,
        src,
        start_offset,
        dst,
        elem_count,
        last_dim,
        asc);
}
