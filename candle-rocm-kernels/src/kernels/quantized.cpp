#include "common.h"

namespace {

constexpr size_t QK5_0 = 32;
constexpr size_t QK8_0 = 32;
constexpr size_t QK_K = 256;

struct BlockQ8_0 {
    uint16_t d;
    int8_t qs[QK8_0];
};

struct BlockQ5_0 {
    uint16_t d;
    uint8_t qh[4];
    uint8_t qs[QK5_0 / 2];
};

struct BlockQ4K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2];
};

struct BlockQ6K {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    uint16_t d;
};

static_assert(sizeof(BlockQ8_0) == 34);
static_assert(sizeof(BlockQ5_0) == 22);
static_assert(sizeof(BlockQ4K) == 144);
static_assert(sizeof(BlockQ6K) == 210);

constexpr unsigned int QMATMUL_THREADS_PER_ROW = 64;
constexpr unsigned int QMATMUL_ROWS_PER_BLOCK = 4;
constexpr unsigned int MOE_GGUF_THREADS_PER_ROW = 64;
constexpr unsigned int MOE_GGUF_ROWS_PER_BLOCK = 4;

__device__ inline uint32_t read_u32_le(const uint8_t* src) {
    return static_cast<uint32_t>(src[0]) |
           (static_cast<uint32_t>(src[1]) << 8) |
           (static_cast<uint32_t>(src[2]) << 16) |
           (static_cast<uint32_t>(src[3]) << 24);
}

__device__ inline void get_scale_min_k4(size_t j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

inline bool is_contiguous(const DeviceLayout& layout) {
    size_t expected_stride = 1;
    for (size_t rev = 0; rev < layout.rank; ++rev) {
        const size_t axis = layout.rank - 1 - rev;
        if (layout.dims[axis] == 0) {
            return true;
        }
        if (layout.strides[axis] != expected_stride) {
            return false;
        }
        expected_stride *= layout.dims[axis];
    }
    return true;
}

__device__ inline float dequant_q8_0(const BlockQ8_0* blocks, size_t inner) {
    const BlockQ8_0& block = blocks[inner / QK8_0];
    return f16_bits_to_f32(block.d) * static_cast<float>(block.qs[inner % QK8_0]);
}

__device__ inline float dequant_q5_0(const BlockQ5_0* blocks, size_t inner) {
    const BlockQ5_0& block = blocks[inner / QK5_0];
    const size_t offset = inner % QK5_0;
    const size_t j = offset % (QK5_0 / 2);
    const uint32_t qh = read_u32_le(block.qh);
    int q;
    if (offset < QK5_0 / 2) {
        const uint8_t xh = static_cast<uint8_t>(((qh >> j) << 4) & 0x10);
        q = static_cast<int>((block.qs[j] & 0x0F) | xh) - 16;
    } else {
        const uint8_t xh = static_cast<uint8_t>((qh >> (j + 12)) & 0x10);
        q = static_cast<int>((block.qs[j] >> 4) | xh) - 16;
    }
    return f16_bits_to_f32(block.d) * static_cast<float>(q);
}

__device__ inline float dequant_q4k(const BlockQ4K* blocks, size_t inner) {
    const BlockQ4K& block = blocks[inner / QK_K];
    const size_t offset = inner % QK_K;
    const size_t group64 = offset / 64;
    const size_t offset64 = offset % 64;
    const size_t scale_index = group64 * 2 + offset64 / 32;
    const uint8_t packed = block.qs[group64 * 32 + offset64 % 32];
    const uint8_t q = offset64 < 32 ? (packed & 0x0F) : (packed >> 4);

    uint8_t scale;
    uint8_t min;
    get_scale_min_k4(scale_index, block.scales, scale, min);
    return f16_bits_to_f32(block.d) * static_cast<float>(scale) * static_cast<float>(q) -
           f16_bits_to_f32(block.dmin) * static_cast<float>(min);
}

__device__ inline float dequant_q6k(const BlockQ6K* blocks, size_t inner) {
    const BlockQ6K& block = blocks[inner / QK_K];
    const size_t offset = inner % QK_K;
    const size_t chunk = offset / 128;
    const size_t offset128 = offset % 128;
    const size_t l = offset128 % 32;
    const uint8_t* ql = block.ql + 64 * chunk;
    const uint8_t* qh = block.qh + 32 * chunk;
    const int8_t* scales = block.scales + 8 * chunk;

    int q;
    size_t scale_index;
    if (offset128 < 32) {
        q = static_cast<int>((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) - 32;
        scale_index = l / 16;
    } else if (offset128 < 64) {
        q = static_cast<int>((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
        scale_index = l / 16 + 2;
    } else if (offset128 < 96) {
        q = static_cast<int>((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        scale_index = l / 16 + 4;
    } else {
        q = static_cast<int>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        scale_index = l / 16 + 6;
    }

    return f16_bits_to_f32(block.d) * static_cast<float>(scales[scale_index]) *
           static_cast<float>(q);
}

template <typename Block, size_t BLOCK_SIZE, bool RHS_CONTIGUOUS, typename Dequant>
__global__ void qmatmul_t_f32_kernel(
    const uint8_t* weights,
    const float* rhs,
    DeviceLayout rhs_layout,
    size_t rhs_start_offset,
    float* dst,
    size_t batch_size,
    size_t nrows,
    size_t ncols,
    size_t elem_count,
    Dequant dequant) {
    const size_t index = blockIdx.x * QMATMUL_ROWS_PER_BLOCK + threadIdx.y;
    const bool active = index < elem_count;
    float acc = 0.0f;

    if (active) {
        const size_t row = index % nrows;
        const size_t batch = index / nrows;
        const size_t blocks_per_row = ncols / BLOCK_SIZE;
        const Block* row_blocks = reinterpret_cast<const Block*>(weights) + row * blocks_per_row;
        const size_t rhs_base = batch * ncols;

        for (size_t inner = threadIdx.x; inner < ncols; inner += QMATMUL_THREADS_PER_ROW) {
            size_t rhs_index;
            if constexpr (RHS_CONTIGUOUS) {
                rhs_index = rhs_start_offset + rhs_base + inner;
            } else {
                rhs_index = storage_index(rhs_base + inner, rhs_layout, rhs_start_offset);
            }
            acc += dequant(row_blocks, inner) * rhs[rhs_index];
        }
    }

    __shared__ float partial[QMATMUL_ROWS_PER_BLOCK][QMATMUL_THREADS_PER_ROW];
    partial[threadIdx.y][threadIdx.x] = acc;
    __syncthreads();

    for (unsigned int stride = QMATMUL_THREADS_PER_ROW / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.y][threadIdx.x] += partial[threadIdx.y][threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (active && threadIdx.x == 0) {
        dst[index] = partial[threadIdx.y][0];
    }
}

template <typename Block, size_t BLOCK_SIZE, typename Dequant>
__global__ void moe_gemm_gguf_f32_kernel(
    const float* input,
    const uint8_t* weights,
    const uint32_t* sorted_token_ids,
    const uint32_t* expert_ids,
    const float* topk_weights,
    float* dst,
    size_t num_experts,
    size_t topk,
    size_t size_m,
    size_t size_n,
    size_t size_k,
    size_t elem_count,
    Dequant dequant) {
    const size_t output_index = blockIdx.x * MOE_GGUF_ROWS_PER_BLOCK + threadIdx.y;
    const bool active = output_index < elem_count;
    float acc = 0.0f;
    uint32_t token_id = 0;
    size_t out_col = 0;

    if (active) {
        const size_t m_idx = output_index / size_n;
        out_col = output_index % size_n;
        token_id = sorted_token_ids[m_idx];
        const uint32_t expert = expert_ids[m_idx];

        if (expert < num_experts) {
            const size_t blocks_per_row = size_k / BLOCK_SIZE;
            const size_t expert_offset =
                static_cast<size_t>(expert) * size_n * blocks_per_row;
            const Block* row_blocks =
                reinterpret_cast<const Block*>(weights) + expert_offset + out_col * blocks_per_row;
            const size_t input_row = topk_weights == nullptr ? token_id / topk : token_id;
            const float* input_row_ptr = input + input_row * size_k;

            for (size_t inner = threadIdx.x; inner < size_k; inner += MOE_GGUF_THREADS_PER_ROW) {
                acc += dequant(row_blocks, inner) * input_row_ptr[inner];
            }
        }
    }

    __shared__ float partial[MOE_GGUF_ROWS_PER_BLOCK][MOE_GGUF_THREADS_PER_ROW];
    partial[threadIdx.y][threadIdx.x] = acc;
    __syncthreads();

    for (unsigned int stride = MOE_GGUF_THREADS_PER_ROW / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.y][threadIdx.x] += partial[threadIdx.y][threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (active && threadIdx.x == 0) {
        const float scale = topk_weights == nullptr ? 1.0f : topk_weights[token_id];
        dst[static_cast<size_t>(token_id) * size_n + out_col] =
            partial[threadIdx.y][0] * scale;
    }
}

struct DequantQ5_0 {
    __device__ float operator()(const BlockQ5_0* blocks, size_t inner) const {
        return dequant_q5_0(blocks, inner);
    }
};

struct DequantQ8_0 {
    __device__ float operator()(const BlockQ8_0* blocks, size_t inner) const {
        return dequant_q8_0(blocks, inner);
    }
};

struct DequantQ4K {
    __device__ float operator()(const BlockQ4K* blocks, size_t inner) const {
        return dequant_q4k(blocks, inner);
    }
};

struct DequantQ6K {
    __device__ float operator()(const BlockQ6K* blocks, size_t inner) const {
        return dequant_q6k(blocks, inner);
    }
};

template <typename Block, size_t BLOCK_SIZE, typename Dequant>
int launch_qmatmul_t_f32(
    const char* op,
    int ordinal,
    const uint8_t* weights,
    const float* rhs,
    const size_t* rhs_dims,
    const size_t* rhs_strides,
    size_t rhs_rank,
    size_t rhs_start_offset,
    float* dst,
    size_t batch_size,
    size_t nrows,
    size_t ncols,
    Dequant dequant) {
    const size_t elem_count = batch_size * nrows;
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    DeviceLayout rhs_layout;
    rc = rhs_layout.init(rhs_dims, rhs_strides, rhs_rank);
    if (rc != 0) {
        return rc;
    }
    const dim3 grid(static_cast<unsigned int>(
        (elem_count + QMATMUL_ROWS_PER_BLOCK - 1) / QMATMUL_ROWS_PER_BLOCK));
    const dim3 block(QMATMUL_THREADS_PER_ROW, QMATMUL_ROWS_PER_BLOCK);
    if (is_contiguous(rhs_layout)) {
        hipLaunchKernelGGL(
            (qmatmul_t_f32_kernel<Block, BLOCK_SIZE, true, Dequant>),
            grid,
            block,
            0,
            0,
            weights,
            rhs,
            rhs_layout,
            rhs_start_offset,
            dst,
            batch_size,
            nrows,
            ncols,
            elem_count,
            dequant);
    } else {
        hipLaunchKernelGGL(
            (qmatmul_t_f32_kernel<Block, BLOCK_SIZE, false, Dequant>),
            grid,
            block,
            0,
            0,
            weights,
            rhs,
            rhs_layout,
            rhs_start_offset,
            dst,
            batch_size,
            nrows,
            ncols,
            elem_count,
            dequant);
    }
    return check_last_launch_error(op);
}

template <typename Block, size_t BLOCK_SIZE, typename Dequant>
int launch_moe_gemm_gguf_f32(
    const char* op,
    int ordinal,
    const float* input,
    const uint8_t* weights,
    const uint32_t* sorted_token_ids,
    const uint32_t* expert_ids,
    const float* topk_weights,
    float* dst,
    size_t num_experts,
    size_t topk,
    size_t size_m,
    size_t size_n,
    size_t size_k,
    Dequant dequant) {
    const size_t elem_count = size_m * size_n;
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    if (topk == 0 || size_k % BLOCK_SIZE != 0) {
        return set_error(op, hipErrorInvalidValue);
    }
    const dim3 grid(static_cast<unsigned int>(
        (elem_count + MOE_GGUF_ROWS_PER_BLOCK - 1) / MOE_GGUF_ROWS_PER_BLOCK));
    const dim3 block(MOE_GGUF_THREADS_PER_ROW, MOE_GGUF_ROWS_PER_BLOCK);
    hipLaunchKernelGGL(
        (moe_gemm_gguf_f32_kernel<Block, BLOCK_SIZE, Dequant>),
        grid,
        block,
        0,
        0,
        input,
        weights,
        sorted_token_ids,
        expert_ids,
        topk_weights,
        dst,
        num_experts,
        topk,
        size_m,
        size_n,
        size_k,
        elem_count,
        dequant);
    return check_last_launch_error(op);
}

} // namespace

extern "C" int hip_qmatmul_t_q8_0_f32(
    int ordinal,
    const uint8_t* weights,
    const float* rhs,
    const size_t* rhs_dims,
    const size_t* rhs_strides,
    size_t rhs_rank,
    size_t rhs_start_offset,
    float* dst,
    size_t batch_size,
    size_t nrows,
    size_t ncols) {
    return launch_qmatmul_t_f32<BlockQ8_0, QK8_0>(
        "qmatmul_t_q8_0_f32",
        ordinal,
        weights,
        rhs,
        rhs_dims,
        rhs_strides,
        rhs_rank,
        rhs_start_offset,
        dst,
        batch_size,
        nrows,
        ncols,
        DequantQ8_0{});
}

extern "C" int hip_qmatmul_t_q5_0_f32(
    int ordinal,
    const uint8_t* weights,
    const float* rhs,
    const size_t* rhs_dims,
    const size_t* rhs_strides,
    size_t rhs_rank,
    size_t rhs_start_offset,
    float* dst,
    size_t batch_size,
    size_t nrows,
    size_t ncols) {
    return launch_qmatmul_t_f32<BlockQ5_0, QK5_0>(
        "qmatmul_t_q5_0_f32",
        ordinal,
        weights,
        rhs,
        rhs_dims,
        rhs_strides,
        rhs_rank,
        rhs_start_offset,
        dst,
        batch_size,
        nrows,
        ncols,
        DequantQ5_0{});
}

extern "C" int hip_qmatmul_t_q4k_f32(
    int ordinal,
    const uint8_t* weights,
    const float* rhs,
    const size_t* rhs_dims,
    const size_t* rhs_strides,
    size_t rhs_rank,
    size_t rhs_start_offset,
    float* dst,
    size_t batch_size,
    size_t nrows,
    size_t ncols) {
    return launch_qmatmul_t_f32<BlockQ4K, QK_K>(
        "qmatmul_t_q4k_f32",
        ordinal,
        weights,
        rhs,
        rhs_dims,
        rhs_strides,
        rhs_rank,
        rhs_start_offset,
        dst,
        batch_size,
        nrows,
        ncols,
        DequantQ4K{});
}

extern "C" int hip_qmatmul_t_q6k_f32(
    int ordinal,
    const uint8_t* weights,
    const float* rhs,
    const size_t* rhs_dims,
    const size_t* rhs_strides,
    size_t rhs_rank,
    size_t rhs_start_offset,
    float* dst,
    size_t batch_size,
    size_t nrows,
    size_t ncols) {
    return launch_qmatmul_t_f32<BlockQ6K, QK_K>(
        "qmatmul_t_q6k_f32",
        ordinal,
        weights,
        rhs,
        rhs_dims,
        rhs_strides,
        rhs_rank,
        rhs_start_offset,
        dst,
        batch_size,
        nrows,
        ncols,
        DequantQ6K{});
}

extern "C" int hip_moe_gemm_gguf_q8_0_f32(
    int ordinal,
    const float* input,
    const uint8_t* weights,
    const uint32_t* sorted_token_ids,
    const uint32_t* expert_ids,
    const float* topk_weights,
    float* dst,
    size_t num_experts,
    size_t topk,
    size_t size_m,
    size_t size_n,
    size_t size_k) {
    return launch_moe_gemm_gguf_f32<BlockQ8_0, QK8_0>(
        "moe_gemm_gguf_q8_0_f32",
        ordinal,
        input,
        weights,
        sorted_token_ids,
        expert_ids,
        topk_weights,
        dst,
        num_experts,
        topk,
        size_m,
        size_n,
        size_k,
        DequantQ8_0{});
}

extern "C" int hip_moe_gemm_gguf_q4k_f32(
    int ordinal,
    const float* input,
    const uint8_t* weights,
    const uint32_t* sorted_token_ids,
    const uint32_t* expert_ids,
    const float* topk_weights,
    float* dst,
    size_t num_experts,
    size_t topk,
    size_t size_m,
    size_t size_n,
    size_t size_k) {
    return launch_moe_gemm_gguf_f32<BlockQ4K, QK_K>(
        "moe_gemm_gguf_q4k_f32",
        ordinal,
        input,
        weights,
        sorted_token_ids,
        expert_ids,
        topk_weights,
        dst,
        num_experts,
        topk,
        size_m,
        size_n,
        size_k,
        DequantQ4K{});
}

extern "C" int hip_moe_gemm_gguf_q6k_f32(
    int ordinal,
    const float* input,
    const uint8_t* weights,
    const uint32_t* sorted_token_ids,
    const uint32_t* expert_ids,
    const float* topk_weights,
    float* dst,
    size_t num_experts,
    size_t topk,
    size_t size_m,
    size_t size_n,
    size_t size_k) {
    return launch_moe_gemm_gguf_f32<BlockQ6K, QK_K>(
        "moe_gemm_gguf_q6k_f32",
        ordinal,
        input,
        weights,
        sorted_token_ids,
        expert_ids,
        topk_weights,
        dst,
        num_experts,
        topk,
        size_m,
        size_n,
        size_k,
        DequantQ6K{});
}
