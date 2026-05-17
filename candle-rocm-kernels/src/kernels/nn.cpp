#include "common.h"

#include <cmath>

namespace {

__global__ void softmax_last_dim_f32_kernel(
    const float* src,
    size_t start_offset,
    float* dst,
    size_t rows,
    size_t cols) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const float* src_row = src + start_offset + row * cols;
    float* dst_row = dst + row * cols;
    float max_value = src_row[0];
    for (size_t col = 1; col < cols; ++col) {
        max_value = fmaxf(max_value, src_row[col]);
    }
    float sum = 0.0f;
    for (size_t col = 0; col < cols; ++col) {
        const float value = expf(src_row[col] - max_value);
        dst_row[col] = value;
        sum += value;
    }
    for (size_t col = 0; col < cols; ++col) {
        dst_row[col] /= sum;
    }
}

__global__ void softmax_last_dim_bf16_kernel(
    const uint16_t* src,
    size_t start_offset,
    uint16_t* dst,
    size_t rows,
    size_t cols) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const uint16_t* src_row = src + start_offset + row * cols;
    uint16_t* dst_row = dst + row * cols;
    float max_value = bf16_bits_to_f32(src_row[0]);
    for (size_t col = 1; col < cols; ++col) {
        max_value = fmaxf(max_value, bf16_bits_to_f32(src_row[col]));
    }
    float sum = 0.0f;
    for (size_t col = 0; col < cols; ++col) {
        const float value = expf(bf16_bits_to_f32(src_row[col]) - max_value);
        sum += value;
    }
    for (size_t col = 0; col < cols; ++col) {
        const float value = expf(bf16_bits_to_f32(src_row[col]) - max_value) / sum;
        dst_row[col] = f32_to_bf16_bits(value);
    }
}

__global__ void rms_norm_f32_kernel(
    const float* src,
    size_t src_start_offset,
    const float* alpha,
    size_t alpha_start_offset,
    float* dst,
    size_t rows,
    size_t cols,
    float eps) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const float* src_row = src + src_start_offset + row * cols;
    const float* alpha_row = alpha + alpha_start_offset;
    float sum2 = 0.0f;
    for (size_t col = 0; col < cols; ++col) {
        const float value = src_row[col];
        volatile float squared = value * value;
        sum2 += squared;
    }
    const float mean = sum2 * (1.0f / static_cast<float>(cols));
    const float denom = sqrtf(mean + eps);
    float* dst_row = dst + row * cols;
    for (size_t col = 0; col < cols; ++col) {
        dst_row[col] = (src_row[col] / denom) * alpha_row[col];
    }
}

__global__ void rms_norm_bf16_kernel(
    const uint16_t* src,
    size_t src_start_offset,
    const uint16_t* alpha,
    size_t alpha_start_offset,
    uint16_t* dst,
    size_t rows,
    size_t cols,
    float eps) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const uint16_t* src_row = src + src_start_offset + row * cols;
    const uint16_t* alpha_row = alpha + alpha_start_offset;
    float sum2 = 0.0f;
    for (size_t col = 0; col < cols; ++col) {
        const float value = bf16_bits_to_f32(src_row[col]);
        volatile float squared = value * value;
        sum2 += squared;
    }
    const float mean = sum2 * (1.0f / static_cast<float>(cols));
    const float denom = sqrtf(mean + eps);
    uint16_t* dst_row = dst + row * cols;
    for (size_t col = 0; col < cols; ++col) {
        const float value =
            (bf16_bits_to_f32(src_row[col]) / denom) * bf16_bits_to_f32(alpha_row[col]);
        dst_row[col] = f32_to_bf16_bits(value);
    }
}

__global__ void layer_norm_f32_kernel(
    const float* src,
    size_t src_start_offset,
    const float* alpha,
    size_t alpha_start_offset,
    const float* beta,
    size_t beta_start_offset,
    float* dst,
    size_t rows,
    size_t cols,
    float eps) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const float* src_row = src + src_start_offset + row * cols;
    const float* alpha_row = alpha + alpha_start_offset;
    const float* beta_row = beta + beta_start_offset;
    float sum = 0.0f;
    float sum2 = 0.0f;
    for (size_t col = 0; col < cols; ++col) {
        const float value = src_row[col];
        sum += value;
        volatile float squared = value * value;
        sum2 += squared;
    }
    const float mean = sum / static_cast<float>(cols);
    volatile float mean_squared = mean * mean;
    const float variance = sum2 / static_cast<float>(cols) - mean_squared;
    const float inv_std = 1.0f / sqrtf(variance + eps);
    float* dst_row = dst + row * cols;
    for (size_t col = 0; col < cols; ++col) {
        dst_row[col] = (src_row[col] - mean) * inv_std * alpha_row[col] + beta_row[col];
    }
}

__global__ void layer_norm_bf16_kernel(
    const uint16_t* src,
    size_t src_start_offset,
    const uint16_t* alpha,
    size_t alpha_start_offset,
    const uint16_t* beta,
    size_t beta_start_offset,
    uint16_t* dst,
    size_t rows,
    size_t cols,
    float eps) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const uint16_t* src_row = src + src_start_offset + row * cols;
    const uint16_t* alpha_row = alpha + alpha_start_offset;
    const uint16_t* beta_row = beta + beta_start_offset;
    float sum = 0.0f;
    float sum2 = 0.0f;
    for (size_t col = 0; col < cols; ++col) {
        const float value = bf16_bits_to_f32(src_row[col]);
        sum += value;
        volatile float squared = value * value;
        sum2 += squared;
    }
    const float mean = sum / static_cast<float>(cols);
    volatile float mean_squared = mean * mean;
    const float variance = sum2 / static_cast<float>(cols) - mean_squared;
    const float inv_std = 1.0f / sqrtf(variance + eps);
    uint16_t* dst_row = dst + row * cols;
    for (size_t col = 0; col < cols; ++col) {
        const float value = (bf16_bits_to_f32(src_row[col]) - mean) * inv_std *
                                bf16_bits_to_f32(alpha_row[col]) +
                            bf16_bits_to_f32(beta_row[col]);
        dst_row[col] = f32_to_bf16_bits(value);
    }
}

__global__ void rope_f32_kernel(
    const float* src,
    size_t src_start_offset,
    const float* cos,
    size_t cos_start_offset,
    const float* sin,
    size_t sin_start_offset,
    float* dst,
    size_t b,
    size_t h,
    size_t t,
    size_t d,
    bool interleaved,
    bool unbatched_rope,
    bool thd) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t half_d = d / 2;
    const size_t pair_count = b * h * t * half_d;
    if (index >= pair_count) {
        return;
    }

    const size_t d_pair = index % half_d;
    const size_t h_i = thd ? (index / half_d) % h : (index / (half_d * t)) % h;
    const size_t t_i = thd ? (index / (half_d * h)) % t : (index / half_d) % t;
    const size_t b_i = index / (half_d * t * h);
    const size_t row_base =
        thd ? ((b_i * t + t_i) * h + h_i) * d : ((b_i * h + h_i) * t + t_i) * d;
    const size_t rope_offset =
        (unbatched_rope ? b_i * t * half_d : 0) + t_i * half_d + d_pair;
    const float c = cos[cos_start_offset + rope_offset];
    const float s = sin[sin_start_offset + rope_offset];
    if (interleaved) {
        const size_t i0 = src_start_offset + row_base + 2 * d_pair;
        const size_t i1 = i0 + 1;
        const float x0 = src[i0];
        const float x1 = src[i1];
        dst[row_base + 2 * d_pair] = x0 * c - x1 * s;
        dst[row_base + 2 * d_pair + 1] = x0 * s + x1 * c;
    } else {
        const size_t i0 = src_start_offset + row_base + d_pair;
        const size_t i1 = src_start_offset + row_base + d_pair + half_d;
        const float x0 = src[i0];
        const float x1 = src[i1];
        dst[row_base + d_pair] = x0 * c - x1 * s;
        dst[row_base + d_pair + half_d] = x0 * s + x1 * c;
    }
}

__global__ void rope_bf16_kernel(
    const uint16_t* src,
    size_t src_start_offset,
    const uint16_t* cos,
    size_t cos_start_offset,
    const uint16_t* sin,
    size_t sin_start_offset,
    uint16_t* dst,
    size_t b,
    size_t h,
    size_t t,
    size_t d,
    bool interleaved,
    bool unbatched_rope,
    bool thd) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t half_d = d / 2;
    const size_t pair_count = b * h * t * half_d;
    if (index >= pair_count) {
        return;
    }

    const size_t d_pair = index % half_d;
    const size_t h_i = thd ? (index / half_d) % h : (index / (half_d * t)) % h;
    const size_t t_i = thd ? (index / (half_d * h)) % t : (index / half_d) % t;
    const size_t b_i = index / (half_d * t * h);
    const size_t row_base =
        thd ? ((b_i * t + t_i) * h + h_i) * d : ((b_i * h + h_i) * t + t_i) * d;
    const size_t rope_offset =
        (unbatched_rope ? b_i * t * half_d : 0) + t_i * half_d + d_pair;
    const float c = bf16_bits_to_f32(cos[cos_start_offset + rope_offset]);
    const float s = bf16_bits_to_f32(sin[sin_start_offset + rope_offset]);
    if (interleaved) {
        const size_t i0 = src_start_offset + row_base + 2 * d_pair;
        const size_t i1 = i0 + 1;
        const float x0 = bf16_bits_to_f32(src[i0]);
        const float x1 = bf16_bits_to_f32(src[i1]);
        dst[row_base + 2 * d_pair] = f32_to_bf16_bits(x0 * c - x1 * s);
        dst[row_base + 2 * d_pair + 1] = f32_to_bf16_bits(x0 * s + x1 * c);
    } else {
        const size_t i0 = src_start_offset + row_base + d_pair;
        const size_t i1 = src_start_offset + row_base + d_pair + half_d;
        const float x0 = bf16_bits_to_f32(src[i0]);
        const float x1 = bf16_bits_to_f32(src[i1]);
        dst[row_base + d_pair] = f32_to_bf16_bits(x0 * c - x1 * s);
        dst[row_base + d_pair + half_d] = f32_to_bf16_bits(x0 * s + x1 * c);
    }
}

} // namespace

extern "C" int hip_softmax_last_dim_f32(
    int ordinal,
    const float* src,
    size_t start_offset,
    float* dst,
    size_t rows,
    size_t cols) {
    int rc = select_device(ordinal);
    if (rc != 0 || rows == 0 || cols == 0) {
        return rc;
    }
    return launch_1d_async(
        "softmax_last_dim_f32",
        rows,
        softmax_last_dim_f32_kernel,
        src,
        start_offset,
        dst,
        rows,
        cols);
}

extern "C" int hip_softmax_last_dim_bf16(
    int ordinal,
    const uint16_t* src,
    size_t start_offset,
    uint16_t* dst,
    size_t rows,
    size_t cols) {
    int rc = select_device(ordinal);
    if (rc != 0 || rows == 0 || cols == 0) {
        return rc;
    }
    return launch_1d_async(
        "softmax_last_dim_bf16",
        rows,
        softmax_last_dim_bf16_kernel,
        src,
        start_offset,
        dst,
        rows,
        cols);
}

extern "C" int hip_rms_norm_f32(
    int ordinal,
    const float* src,
    size_t src_start_offset,
    const float* alpha,
    size_t alpha_start_offset,
    float* dst,
    size_t rows,
    size_t cols,
    float eps) {
    int rc = select_device(ordinal);
    if (rc != 0 || rows == 0 || cols == 0) {
        return rc;
    }
    return launch_1d_async(
        "rms_norm_f32",
        rows,
        rms_norm_f32_kernel,
        src,
        src_start_offset,
        alpha,
        alpha_start_offset,
        dst,
        rows,
        cols,
        eps);
}

extern "C" int hip_rms_norm_bf16(
    int ordinal,
    const uint16_t* src,
    size_t src_start_offset,
    const uint16_t* alpha,
    size_t alpha_start_offset,
    uint16_t* dst,
    size_t rows,
    size_t cols,
    float eps) {
    int rc = select_device(ordinal);
    if (rc != 0 || rows == 0 || cols == 0) {
        return rc;
    }
    return launch_1d_async(
        "rms_norm_bf16",
        rows,
        rms_norm_bf16_kernel,
        src,
        src_start_offset,
        alpha,
        alpha_start_offset,
        dst,
        rows,
        cols,
        eps);
}

extern "C" int hip_layer_norm_f32(
    int ordinal,
    const float* src,
    size_t src_start_offset,
    const float* alpha,
    size_t alpha_start_offset,
    const float* beta,
    size_t beta_start_offset,
    float* dst,
    size_t rows,
    size_t cols,
    float eps) {
    int rc = select_device(ordinal);
    if (rc != 0 || rows == 0 || cols == 0) {
        return rc;
    }
    return launch_1d_async(
        "layer_norm_f32",
        rows,
        layer_norm_f32_kernel,
        src,
        src_start_offset,
        alpha,
        alpha_start_offset,
        beta,
        beta_start_offset,
        dst,
        rows,
        cols,
        eps);
}

extern "C" int hip_layer_norm_bf16(
    int ordinal,
    const uint16_t* src,
    size_t src_start_offset,
    const uint16_t* alpha,
    size_t alpha_start_offset,
    const uint16_t* beta,
    size_t beta_start_offset,
    uint16_t* dst,
    size_t rows,
    size_t cols,
    float eps) {
    int rc = select_device(ordinal);
    if (rc != 0 || rows == 0 || cols == 0) {
        return rc;
    }
    return launch_1d_async(
        "layer_norm_bf16",
        rows,
        layer_norm_bf16_kernel,
        src,
        src_start_offset,
        alpha,
        alpha_start_offset,
        beta,
        beta_start_offset,
        dst,
        rows,
        cols,
        eps);
}

extern "C" int hip_rope_f32(
    int ordinal,
    const float* src,
    size_t src_start_offset,
    const float* cos,
    size_t cos_start_offset,
    const float* sin,
    size_t sin_start_offset,
    float* dst,
    size_t b,
    size_t h,
    size_t t,
    size_t d,
    int interleaved,
    int unbatched_rope,
    int thd) {
    int rc = select_device(ordinal);
    const size_t pair_count = b * h * t * (d / 2);
    if (rc != 0 || pair_count == 0) {
        return rc;
    }
    return launch_1d_async(
        "rope_f32",
        pair_count,
        rope_f32_kernel,
        src,
        src_start_offset,
        cos,
        cos_start_offset,
        sin,
        sin_start_offset,
        dst,
        b,
        h,
        t,
        d,
        interleaved != 0,
        unbatched_rope != 0,
        thd != 0);
}

extern "C" int hip_rope_bf16(
    int ordinal,
    const uint16_t* src,
    size_t src_start_offset,
    const uint16_t* cos,
    size_t cos_start_offset,
    const uint16_t* sin,
    size_t sin_start_offset,
    uint16_t* dst,
    size_t b,
    size_t h,
    size_t t,
    size_t d,
    int interleaved,
    int unbatched_rope,
    int thd) {
    int rc = select_device(ordinal);
    const size_t pair_count = b * h * t * (d / 2);
    if (rc != 0 || pair_count == 0) {
        return rc;
    }
    return launch_1d_async(
        "rope_bf16",
        pair_count,
        rope_bf16_kernel,
        src,
        src_start_offset,
        cos,
        cos_start_offset,
        sin,
        sin_start_offset,
        dst,
        b,
        h,
        t,
        d,
        interleaved != 0,
        unbatched_rope != 0,
        thd != 0);
}
