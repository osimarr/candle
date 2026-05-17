#include "common.h"

#include <cstdint>
#include <limits>

namespace {

__device__ size_t base_index_for_output(
    size_t logical_index,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    uint64_t reduce_mask) {
    size_t storage = start_offset;
    for (size_t rev = 0; rev < rank; ++rev) {
        const size_t axis = rank - 1 - rev;
        const bool reduced = ((reduce_mask >> axis) & 1U) != 0;
        const size_t out_dim = reduced ? 1 : dims[axis];
        const size_t index = out_dim == 0 ? 0 : logical_index % out_dim;
        if (out_dim != 0) {
            logical_index /= out_dim;
        }
        storage += index * strides[axis];
    }
    return storage;
}

__device__ size_t reduce_offset(
    size_t reduce_index,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    uint64_t reduce_mask) {
    size_t offset = 0;
    for (size_t rev = 0; rev < rank; ++rev) {
        const size_t axis = rank - 1 - rev;
        if (((reduce_mask >> axis) & 1U) == 0) {
            continue;
        }
        const size_t dim = dims[axis];
        const size_t index = dim == 0 ? 0 : reduce_index % dim;
        if (dim != 0) {
            reduce_index /= dim;
        }
        offset += index * strides[axis];
    }
    return offset;
}

__global__ void reduce_f32_kernel(
    int op,
    const float* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    uint64_t reduce_mask,
    size_t reduce_count,
    float* dst,
    uint32_t* dst_u32,
    size_t elem_count) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }

    const size_t base =
        base_index_for_output(logical_index, dims, strides, rank, start_offset, reduce_mask);
    if (op == 1) {
        float acc = 0.0f;
        for (size_t r = 0; r < reduce_count; ++r) {
            acc += src[base + reduce_offset(r, dims, strides, rank, reduce_mask)];
        }
        dst[logical_index] = acc;
        return;
    }

    float best = src[base];
    uint32_t best_index = 0;
    for (size_t r = 1; r < reduce_count; ++r) {
        const float value = src[base + reduce_offset(r, dims, strides, rank, reduce_mask)];
        const bool update_min = (op == 2 || op == 4) && value < best;
        const bool update_max = (op == 3 || op == 5) && value > best;
        if (update_min || update_max) {
            best = value;
            best_index = static_cast<uint32_t>(r);
        }
    }
    if (op == 4 || op == 5) {
        dst_u32[logical_index] = best_index;
    } else {
        dst[logical_index] = best;
    }
}

} // namespace

extern "C" int hip_reduce_f32(
    int ordinal,
    int op,
    const float* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    uint64_t reduce_mask,
    size_t reduce_count,
    void* dst,
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
        "reduce_f32",
        elem_count,
        reduce_f32_kernel,
        op,
        src,
        layout.dims,
        layout.strides,
        layout.rank,
        start_offset,
        reduce_mask,
        reduce_count,
        static_cast<float*>(dst),
        static_cast<uint32_t*>(dst),
        elem_count);
}
