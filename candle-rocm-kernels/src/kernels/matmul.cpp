#include "common.h"

namespace {

__global__ void matmul_f32_kernel(
    const float* lhs,
    const float* rhs,
    float* dst,
    size_t b,
    size_t m,
    size_t n,
    size_t k,
    size_t lhs_start_offset,
    size_t rhs_start_offset,
    size_t lhs_batch_stride,
    size_t rhs_batch_stride,
    size_t lhs_row_stride,
    size_t lhs_col_stride,
    size_t rhs_row_stride,
    size_t rhs_col_stride,
    size_t elem_count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elem_count) {
        return;
    }
    const size_t col = index % n;
    const size_t row = (index / n) % m;
    const size_t batch = index / (m * n);

    const size_t lhs_base = lhs_start_offset + batch * lhs_batch_stride + row * lhs_row_stride;
    const size_t rhs_base = rhs_start_offset + batch * rhs_batch_stride + col * rhs_col_stride;
    float acc = 0.0f;
    for (size_t inner = 0; inner < k; ++inner) {
        acc += lhs[lhs_base + inner * lhs_col_stride] *
               rhs[rhs_base + inner * rhs_row_stride];
    }
    dst[index] = acc;
}

__global__ void matmul_bf16_kernel(
    const uint16_t* lhs,
    const uint16_t* rhs,
    uint16_t* dst,
    size_t b,
    size_t m,
    size_t n,
    size_t k,
    size_t lhs_start_offset,
    size_t rhs_start_offset,
    size_t lhs_batch_stride,
    size_t rhs_batch_stride,
    size_t lhs_row_stride,
    size_t lhs_col_stride,
    size_t rhs_row_stride,
    size_t rhs_col_stride,
    size_t elem_count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elem_count) {
        return;
    }
    const size_t col = index % n;
    const size_t row = (index / n) % m;
    const size_t batch = index / (m * n);

    const size_t lhs_base = lhs_start_offset + batch * lhs_batch_stride + row * lhs_row_stride;
    const size_t rhs_base = rhs_start_offset + batch * rhs_batch_stride + col * rhs_col_stride;
    float acc = 0.0f;
    for (size_t inner = 0; inner < k; ++inner) {
        acc += bf16_bits_to_f32(lhs[lhs_base + inner * lhs_col_stride]) *
               bf16_bits_to_f32(rhs[rhs_base + inner * rhs_row_stride]);
    }
    dst[index] = f32_to_bf16_bits(acc);
}

__global__ void matmul_f8e4m3_kernel(
    const uint8_t* lhs,
    const uint8_t* rhs,
    uint8_t* dst,
    size_t b,
    size_t m,
    size_t n,
    size_t k,
    size_t lhs_start_offset,
    size_t rhs_start_offset,
    size_t lhs_batch_stride,
    size_t rhs_batch_stride,
    size_t lhs_row_stride,
    size_t lhs_col_stride,
    size_t rhs_row_stride,
    size_t rhs_col_stride,
    size_t elem_count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elem_count) {
        return;
    }
    const size_t col = index % n;
    const size_t row = (index / n) % m;
    const size_t batch = index / (m * n);

    const size_t lhs_base = lhs_start_offset + batch * lhs_batch_stride + row * lhs_row_stride;
    const size_t rhs_base = rhs_start_offset + batch * rhs_batch_stride + col * rhs_col_stride;
    float acc = 0.0f;
    for (size_t inner = 0; inner < k; ++inner) {
        acc += f8e4m3_bits_to_f32(lhs[lhs_base + inner * lhs_col_stride]) *
               f8e4m3_bits_to_f32(rhs[rhs_base + inner * rhs_row_stride]);
    }
    dst[index] = f32_to_f8e4m3_bits(acc);
}

} // namespace

extern "C" int hip_matmul_f32(
    int ordinal,
    const float* lhs,
    const float* rhs,
    float* dst,
    size_t b,
    size_t m,
    size_t n,
    size_t k,
    size_t lhs_start_offset,
    size_t rhs_start_offset,
    size_t lhs_batch_stride,
    size_t rhs_batch_stride,
    size_t lhs_row_stride,
    size_t lhs_col_stride,
    size_t rhs_row_stride,
    size_t rhs_col_stride) {
    const size_t elem_count = b * m * n;
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    return launch_1d_async(
        "matmul_f32",
        elem_count,
        matmul_f32_kernel,
        lhs,
        rhs,
        dst,
        b,
        m,
        n,
        k,
        lhs_start_offset,
        rhs_start_offset,
        lhs_batch_stride,
        rhs_batch_stride,
        lhs_row_stride,
        lhs_col_stride,
        rhs_row_stride,
        rhs_col_stride,
        elem_count);
}

extern "C" int hip_matmul_bf16(
    int ordinal,
    const uint16_t* lhs,
    const uint16_t* rhs,
    uint16_t* dst,
    size_t b,
    size_t m,
    size_t n,
    size_t k,
    size_t lhs_start_offset,
    size_t rhs_start_offset,
    size_t lhs_batch_stride,
    size_t rhs_batch_stride,
    size_t lhs_row_stride,
    size_t lhs_col_stride,
    size_t rhs_row_stride,
    size_t rhs_col_stride) {
    const size_t elem_count = b * m * n;
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    return launch_1d_async(
        "matmul_bf16",
        elem_count,
        matmul_bf16_kernel,
        lhs,
        rhs,
        dst,
        b,
        m,
        n,
        k,
        lhs_start_offset,
        rhs_start_offset,
        lhs_batch_stride,
        rhs_batch_stride,
        lhs_row_stride,
        lhs_col_stride,
        rhs_row_stride,
        rhs_col_stride,
        elem_count);
}

extern "C" int hip_matmul_f8e4m3(
    int ordinal,
    const uint8_t* lhs,
    const uint8_t* rhs,
    uint8_t* dst,
    size_t b,
    size_t m,
    size_t n,
    size_t k,
    size_t lhs_start_offset,
    size_t rhs_start_offset,
    size_t lhs_batch_stride,
    size_t rhs_batch_stride,
    size_t lhs_row_stride,
    size_t lhs_col_stride,
    size_t rhs_row_stride,
    size_t rhs_col_stride) {
    const size_t elem_count = b * m * n;
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    return launch_1d_async(
        "matmul_f8e4m3",
        elem_count,
        matmul_f8e4m3_kernel,
        lhs,
        rhs,
        dst,
        b,
        m,
        n,
        k,
        lhs_start_offset,
        rhs_start_offset,
        lhs_batch_stride,
        rhs_batch_stride,
        lhs_row_stride,
        lhs_col_stride,
        rhs_row_stride,
        rhs_col_stride,
        elem_count);
}
