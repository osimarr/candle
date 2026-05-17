#include "common.h"

#include <cstdint>
#include <limits>

namespace {

__device__ size_t base_index_for_output(
    size_t logical_index,
    const DeviceLayout& layout,
    size_t start_offset,
    uint64_t reduce_mask) {
    size_t storage = start_offset;
    for (size_t rev = 0; rev < layout.rank; ++rev) {
        const size_t axis = layout.rank - 1 - rev;
        const bool reduced = ((reduce_mask >> axis) & 1U) != 0;
        const size_t out_dim = reduced ? 1 : layout.dims[axis];
        const size_t index = out_dim == 0 ? 0 : logical_index % out_dim;
        if (out_dim != 0) {
            logical_index /= out_dim;
        }
        storage += index * layout.strides[axis];
    }
    return storage;
}

__device__ size_t reduce_offset(
    size_t reduce_index,
    const DeviceLayout& layout,
    uint64_t reduce_mask) {
    size_t offset = 0;
    for (size_t rev = 0; rev < layout.rank; ++rev) {
        const size_t axis = layout.rank - 1 - rev;
        if (((reduce_mask >> axis) & 1U) == 0) {
            continue;
        }
        const size_t dim = layout.dims[axis];
        const size_t index = dim == 0 ? 0 : reduce_index % dim;
        if (dim != 0) {
            reduce_index /= dim;
        }
        offset += index * layout.strides[axis];
    }
    return offset;
}

template <typename T>
__global__ void reduce_kernel(
    int op,
    const T* src,
    DeviceLayout layout,
    size_t start_offset,
    uint64_t reduce_mask,
    size_t reduce_count,
    T* dst,
    uint32_t* dst_u32,
    size_t elem_count) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }

    const size_t base =
        base_index_for_output(logical_index, layout, start_offset, reduce_mask);
    if (op == 1) {
        float acc = 0.0f;
        for (size_t r = 0; r < reduce_count; ++r) {
            acc += to_f32(src[base + reduce_offset(r, layout, reduce_mask)]);
        }
        dst[logical_index] = from_f32<T>(acc);
        return;
    }

    float best = to_f32(src[base]);
    uint32_t best_index = 0;
    for (size_t r = 1; r < reduce_count; ++r) {
        const float value = to_f32(src[base + reduce_offset(r, layout, reduce_mask)]);
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
        dst[logical_index] = from_f32<T>(best);
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
        reduce_kernel<float>,
        op,
        src,
        layout,
        start_offset,
        reduce_mask,
        reduce_count,
        static_cast<float*>(dst),
        static_cast<uint32_t*>(dst),
        elem_count);
}

extern "C" int hip_reduce_bf16(
    int ordinal,
    int op,
    const uint16_t* src,
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
        "reduce_bf16",
        elem_count,
        reduce_kernel<uint16_t>,
        op,
        src,
        layout,
        start_offset,
        reduce_mask,
        reduce_count,
        static_cast<uint16_t*>(dst),
        static_cast<uint32_t*>(dst),
        elem_count);
}

extern "C" int hip_reduce_f8e4m3(
    int ordinal,
    int op,
    const uint8_t* src,
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
        "reduce_f8e4m3",
        elem_count,
        reduce_kernel<F8E4M3Storage>,
        op,
        as_f8e4m3(src),
        layout,
        start_offset,
        reduce_mask,
        reduce_count,
        as_f8e4m3(static_cast<uint8_t*>(dst)),
        static_cast<uint32_t*>(dst),
        elem_count);
}
