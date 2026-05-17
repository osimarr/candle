#include "common.h"

#include <cstdint>
#include <limits>

namespace {

template <typename I>
__device__ bool is_missing_index(I value);

template <>
__device__ bool is_missing_index<uint32_t>(uint32_t value) {
    return value == std::numeric_limits<uint32_t>::max();
}

template <>
__device__ bool is_missing_index<int64_t>(int64_t value) {
    return value == std::numeric_limits<int64_t>::max();
}

template <typename T, typename I>
__global__ void index_select_kernel(
    const T* src,
    DeviceLayout src_layout,
    size_t src_start_offset,
    const I* ids,
    size_t ids_start_offset,
    size_t ids_stride,
    size_t dim,
    size_t n_ids,
    T* dst,
    size_t elem_count) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }

    size_t tmp = logical_index;
    size_t src_index = src_start_offset;
    size_t selected_id_pos = 0;
    for (size_t rev = 0; rev < src_layout.rank; ++rev) {
        const size_t axis = src_layout.rank - 1 - rev;
        const size_t out_dim = axis == dim ? n_ids : src_layout.dims[axis];
        const size_t coord = tmp % out_dim;
        tmp /= out_dim;
        if (axis == dim) {
            selected_id_pos = coord;
        } else {
            src_index += coord * src_layout.strides[axis];
        }
    }

    const I raw_index = ids[ids_start_offset + selected_id_pos * ids_stride];
    if (is_missing_index<I>(raw_index)) {
        dst[logical_index] = T{};
        return;
    }
    const size_t selected = static_cast<size_t>(raw_index);
    dst[logical_index] = src[src_index + selected * src_layout.strides[dim]];
}

template <typename T, typename I>
__global__ void gather_kernel(
    const T* src,
    DeviceLayout src_layout,
    size_t src_start_offset,
    const I* ids,
    DeviceLayout ids_layout,
    size_t ids_start_offset,
    size_t dim,
    T* dst,
    size_t elem_count) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }

    const size_t ids_index = storage_index(logical_index, ids_layout, ids_start_offset);
    const I raw_index = ids[ids_index];
    if (is_missing_index<I>(raw_index)) {
        dst[logical_index] = T{};
        return;
    }

    size_t tmp = logical_index;
    size_t src_index = src_start_offset;
    for (size_t rev = 0; rev < src_layout.rank; ++rev) {
        const size_t axis = src_layout.rank - 1 - rev;
        const size_t coord = tmp % ids_layout.dims[axis];
        tmp /= ids_layout.dims[axis];
        src_index +=
            (axis == dim ? static_cast<size_t>(raw_index) : coord) * src_layout.strides[axis];
    }
    dst[logical_index] = src[src_index];
}

template <typename T, typename I>
__global__ void scatter_kernel(
    int add,
    T* dst,
    DeviceLayout dst_layout,
    size_t dst_start_offset,
    const I* ids,
    DeviceLayout ids_layout,
    size_t ids_start_offset,
    const T* src,
    DeviceLayout src_layout,
    size_t src_start_offset,
    size_t dim,
    size_t src_elem_count,
    size_t dst_elem_count) {
    const size_t dst_logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst_logical_index >= dst_elem_count) {
        return;
    }
    const size_t dst_storage_index = storage_index(
        dst_logical_index,
        dst_layout,
        dst_start_offset);
    float value = to_f32(dst[dst_storage_index]);

    for (size_t src_logical_index = 0; src_logical_index < src_elem_count;
         ++src_logical_index) {
        const size_t ids_index = storage_index(
            src_logical_index,
            ids_layout,
            ids_start_offset);
        const I raw_index = ids[ids_index];
        if (is_missing_index<I>(raw_index)) {
            continue;
        }
        size_t src_tmp = src_logical_index;
        size_t dst_tmp = dst_logical_index;
        bool matches = true;
        for (size_t rev = 0; rev < dst_layout.rank; ++rev) {
            const size_t axis = dst_layout.rank - 1 - rev;
            const size_t src_coord = src_tmp % src_layout.dims[axis];
            src_tmp /= src_layout.dims[axis];
            const size_t dst_coord = dst_tmp % dst_layout.dims[axis];
            dst_tmp /= dst_layout.dims[axis];
            if (axis == dim) {
                if (static_cast<size_t>(raw_index) != dst_coord) {
                    matches = false;
                    break;
                }
            } else if (src_coord != dst_coord) {
                matches = false;
                break;
            }
        }
        if (matches) {
            const float src_value = to_f32(src[storage_index(
                src_logical_index,
                src_layout,
                src_start_offset)]);
            if (add) {
                value += src_value;
            } else {
                value = src_value;
            }
        }
    }
    dst[dst_storage_index] = from_f32<T>(value);
}

template <typename T, typename I>
__global__ void index_add_kernel(
    const T* input,
    DeviceLayout input_layout,
    size_t input_start_offset,
    const I* ids,
    size_t ids_start_offset,
    size_t ids_stride,
    size_t ids_len,
    const T* src,
    DeviceLayout src_layout,
    size_t src_start_offset,
    size_t dim,
    T* dst,
    size_t elem_count) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }
    float value = to_f32(input[storage_index(
        logical_index,
        input_layout,
        input_start_offset)]);

    size_t dst_dim_coord = logical_index;
    for (size_t rev = 0; rev < input_layout.rank; ++rev) {
        const size_t axis = input_layout.rank - 1 - rev;
        const size_t coord = dst_dim_coord % input_layout.dims[axis];
        dst_dim_coord /= input_layout.dims[axis];
        if (axis == dim) {
            dst_dim_coord = coord;
            break;
        }
    }

    for (size_t id_pos = 0; id_pos < ids_len; ++id_pos) {
        const I raw_index = ids[ids_start_offset + id_pos * ids_stride];
        if (is_missing_index<I>(raw_index) || static_cast<size_t>(raw_index) != dst_dim_coord) {
            continue;
        }
        size_t tmp = logical_index;
        size_t src_index = src_start_offset;
        for (size_t rev = 0; rev < input_layout.rank; ++rev) {
            const size_t axis = input_layout.rank - 1 - rev;
            const size_t coord = tmp % input_layout.dims[axis];
            tmp /= input_layout.dims[axis];
            src_index += (axis == dim ? id_pos : coord) * src_layout.strides[axis];
        }
        value += to_f32(src[src_index]);
    }
    dst[logical_index] = from_f32<T>(value);
}

template <typename T, typename I>
int index_select(
    const char* name,
    int ordinal,
    const T* src,
    const size_t* src_dims,
    const size_t* src_strides,
    size_t rank,
    size_t src_start_offset,
    const I* ids,
    size_t ids_start_offset,
    size_t ids_stride,
    size_t dim,
    size_t n_ids,
    T* dst,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    DeviceLayout src_layout;
    rc = src_layout.init(src_dims, src_strides, rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        name,
        elem_count,
        index_select_kernel<T, I>,
        src,
        src_layout,
        src_start_offset,
        ids,
        ids_start_offset,
        ids_stride,
        dim,
        n_ids,
        dst,
        elem_count);
}

template <typename T, typename I>
int gather(
    const char* name,
    int ordinal,
    const T* src,
    const size_t* src_dims,
    const size_t* src_strides,
    size_t src_rank,
    size_t src_start_offset,
    const I* ids,
    const size_t* ids_dims,
    const size_t* ids_strides,
    size_t ids_rank,
    size_t ids_start_offset,
    size_t dim,
    T* dst,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    DeviceLayout src_layout;
    rc = src_layout.init(src_dims, src_strides, src_rank);
    if (rc != 0) {
        return rc;
    }
    DeviceLayout ids_layout;
    rc = ids_layout.init(ids_dims, ids_strides, ids_rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        name,
        elem_count,
        gather_kernel<T, I>,
        src,
        src_layout,
        src_start_offset,
        ids,
        ids_layout,
        ids_start_offset,
        dim,
        dst,
        elem_count);
}

template <typename T, typename I>
int scatter(
    const char* name,
    int ordinal,
    int add,
    T* dst,
    const size_t* dst_dims,
    const size_t* dst_strides,
    size_t dst_rank,
    size_t dst_start_offset,
    const I* ids,
    const size_t* ids_dims,
    const size_t* ids_strides,
    size_t ids_rank,
    size_t ids_start_offset,
    const T* src,
    const size_t* src_dims,
    const size_t* src_strides,
    size_t src_rank,
    size_t src_start_offset,
    size_t dim,
    size_t src_elem_count,
    size_t dst_elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || dst_elem_count == 0) {
        return rc;
    }
    DeviceLayout dst_layout;
    rc = dst_layout.init(dst_dims, dst_strides, dst_rank);
    if (rc != 0) {
        return rc;
    }
    DeviceLayout ids_layout;
    rc = ids_layout.init(ids_dims, ids_strides, ids_rank);
    if (rc != 0) {
        return rc;
    }
    DeviceLayout src_layout;
    rc = src_layout.init(src_dims, src_strides, src_rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        name,
        dst_elem_count,
        scatter_kernel<T, I>,
        add,
        dst,
        dst_layout,
        dst_start_offset,
        ids,
        ids_layout,
        ids_start_offset,
        src,
        src_layout,
        src_start_offset,
        dim,
        src_elem_count,
        dst_elem_count);
}

template <typename T, typename I>
int index_add(
    const char* name,
    int ordinal,
    const T* input,
    const size_t* input_dims,
    const size_t* input_strides,
    size_t input_rank,
    size_t input_start_offset,
    const I* ids,
    size_t ids_start_offset,
    size_t ids_stride,
    size_t ids_len,
    const T* src,
    const size_t* src_dims,
    const size_t* src_strides,
    size_t src_rank,
    size_t src_start_offset,
    size_t dim,
    T* dst,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    DeviceLayout input_layout;
    rc = input_layout.init(input_dims, input_strides, input_rank);
    if (rc != 0) {
        return rc;
    }
    DeviceLayout src_layout;
    rc = src_layout.init(src_dims, src_strides, src_rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        name,
        elem_count,
        index_add_kernel<T, I>,
        input,
        input_layout,
        input_start_offset,
        ids,
        ids_start_offset,
        ids_stride,
        ids_len,
        src,
        src_layout,
        src_start_offset,
        dim,
        dst,
        elem_count);
}

} // namespace

extern "C" int hip_index_select_u32_f32(
    int ordinal,
    const float* src,
    const size_t* src_dims,
    const size_t* src_strides,
    size_t rank,
    size_t src_start_offset,
    const uint32_t* ids,
    size_t ids_start_offset,
    size_t ids_stride,
    size_t dim,
    size_t n_ids,
    float* dst,
    size_t elem_count) {
    return index_select<float>(
        "index_select_u32_f32",
        ordinal,
        src,
        src_dims,
        src_strides,
        rank,
        src_start_offset,
        ids,
        ids_start_offset,
        ids_stride,
        dim,
        n_ids,
        dst,
        elem_count);
}

extern "C" int hip_index_select_i64_f32(
    int ordinal,
    const float* src,
    const size_t* src_dims,
    const size_t* src_strides,
    size_t rank,
    size_t src_start_offset,
    const int64_t* ids,
    size_t ids_start_offset,
    size_t ids_stride,
    size_t dim,
    size_t n_ids,
    float* dst,
    size_t elem_count) {
    return index_select<float>(
        "index_select_i64_f32",
        ordinal,
        src,
        src_dims,
        src_strides,
        rank,
        src_start_offset,
        ids,
        ids_start_offset,
        ids_stride,
        dim,
        n_ids,
        dst,
        elem_count);
}

extern "C" int hip_index_select_u32_bf16(
    int ordinal,
    const uint16_t* src,
    const size_t* src_dims,
    const size_t* src_strides,
    size_t rank,
    size_t src_start_offset,
    const uint32_t* ids,
    size_t ids_start_offset,
    size_t ids_stride,
    size_t dim,
    size_t n_ids,
    uint16_t* dst,
    size_t elem_count) {
    return index_select<uint16_t>(
        "index_select_u32_bf16",
        ordinal,
        src,
        src_dims,
        src_strides,
        rank,
        src_start_offset,
        ids,
        ids_start_offset,
        ids_stride,
        dim,
        n_ids,
        dst,
        elem_count);
}

extern "C" int hip_index_select_i64_bf16(
    int ordinal,
    const uint16_t* src,
    const size_t* src_dims,
    const size_t* src_strides,
    size_t rank,
    size_t src_start_offset,
    const int64_t* ids,
    size_t ids_start_offset,
    size_t ids_stride,
    size_t dim,
    size_t n_ids,
    uint16_t* dst,
    size_t elem_count) {
    return index_select<uint16_t>(
        "index_select_i64_bf16",
        ordinal,
        src,
        src_dims,
        src_strides,
        rank,
        src_start_offset,
        ids,
        ids_start_offset,
        ids_stride,
        dim,
        n_ids,
        dst,
        elem_count);
}

extern "C" int hip_gather_u32_f32(
    int ordinal,
    const float* src,
    const size_t* src_dims,
    const size_t* src_strides,
    size_t src_rank,
    size_t src_start_offset,
    const uint32_t* ids,
    const size_t* ids_dims,
    const size_t* ids_strides,
    size_t ids_rank,
    size_t ids_start_offset,
    size_t dim,
    float* dst,
    size_t elem_count) {
    return gather<float>(
        "gather_u32_f32",
        ordinal,
        src,
        src_dims,
        src_strides,
        src_rank,
        src_start_offset,
        ids,
        ids_dims,
        ids_strides,
        ids_rank,
        ids_start_offset,
        dim,
        dst,
        elem_count);
}

extern "C" int hip_scatter_u32_f32(
    int ordinal,
    int add,
    float* dst,
    const size_t* dst_dims,
    const size_t* dst_strides,
    size_t dst_rank,
    size_t dst_start_offset,
    const uint32_t* ids,
    const size_t* ids_dims,
    const size_t* ids_strides,
    size_t ids_rank,
    size_t ids_start_offset,
    const float* src,
    const size_t* src_dims,
    const size_t* src_strides,
    size_t src_rank,
    size_t src_start_offset,
    size_t dim,
    size_t src_elem_count,
    size_t dst_elem_count) {
    return scatter<float>(
        "scatter_u32_f32",
        ordinal,
        add,
        dst,
        dst_dims,
        dst_strides,
        dst_rank,
        dst_start_offset,
        ids,
        ids_dims,
        ids_strides,
        ids_rank,
        ids_start_offset,
        src,
        src_dims,
        src_strides,
        src_rank,
        src_start_offset,
        dim,
        src_elem_count,
        dst_elem_count);
}

extern "C" int hip_index_add_u32_f32(
    int ordinal,
    const float* input,
    const size_t* input_dims,
    const size_t* input_strides,
    size_t input_rank,
    size_t input_start_offset,
    const uint32_t* ids,
    size_t ids_start_offset,
    size_t ids_stride,
    size_t ids_len,
    const float* src,
    const size_t* src_dims,
    const size_t* src_strides,
    size_t src_rank,
    size_t src_start_offset,
    size_t dim,
    float* dst,
    size_t elem_count) {
    return index_add<float>(
        "index_add_u32_f32",
        ordinal,
        input,
        input_dims,
        input_strides,
        input_rank,
        input_start_offset,
        ids,
        ids_start_offset,
        ids_stride,
        ids_len,
        src,
        src_dims,
        src_strides,
        src_rank,
        src_start_offset,
        dim,
        dst,
        elem_count);
}

extern "C" int hip_gather_u32_bf16(
    int ordinal,
    const uint16_t* src,
    const size_t* src_dims,
    const size_t* src_strides,
    size_t src_rank,
    size_t src_start_offset,
    const uint32_t* ids,
    const size_t* ids_dims,
    const size_t* ids_strides,
    size_t ids_rank,
    size_t ids_start_offset,
    size_t dim,
    uint16_t* dst,
    size_t elem_count) {
    return gather<uint16_t>(
        "gather_u32_bf16",
        ordinal,
        src,
        src_dims,
        src_strides,
        src_rank,
        src_start_offset,
        ids,
        ids_dims,
        ids_strides,
        ids_rank,
        ids_start_offset,
        dim,
        dst,
        elem_count);
}

extern "C" int hip_scatter_u32_bf16(
    int ordinal,
    int add,
    uint16_t* dst,
    const size_t* dst_dims,
    const size_t* dst_strides,
    size_t dst_rank,
    size_t dst_start_offset,
    const uint32_t* ids,
    const size_t* ids_dims,
    const size_t* ids_strides,
    size_t ids_rank,
    size_t ids_start_offset,
    const uint16_t* src,
    const size_t* src_dims,
    const size_t* src_strides,
    size_t src_rank,
    size_t src_start_offset,
    size_t dim,
    size_t src_elem_count,
    size_t dst_elem_count) {
    return scatter<uint16_t>(
        "scatter_u32_bf16",
        ordinal,
        add,
        dst,
        dst_dims,
        dst_strides,
        dst_rank,
        dst_start_offset,
        ids,
        ids_dims,
        ids_strides,
        ids_rank,
        ids_start_offset,
        src,
        src_dims,
        src_strides,
        src_rank,
        src_start_offset,
        dim,
        src_elem_count,
        dst_elem_count);
}

extern "C" int hip_index_add_u32_bf16(
    int ordinal,
    const uint16_t* input,
    const size_t* input_dims,
    const size_t* input_strides,
    size_t input_rank,
    size_t input_start_offset,
    const uint32_t* ids,
    size_t ids_start_offset,
    size_t ids_stride,
    size_t ids_len,
    const uint16_t* src,
    const size_t* src_dims,
    const size_t* src_strides,
    size_t src_rank,
    size_t src_start_offset,
    size_t dim,
    uint16_t* dst,
    size_t elem_count) {
    return index_add<uint16_t>(
        "index_add_u32_bf16",
        ordinal,
        input,
        input_dims,
        input_strides,
        input_rank,
        input_start_offset,
        ids,
        ids_start_offset,
        ids_stride,
        ids_len,
        src,
        src_dims,
        src_strides,
        src_rank,
        src_start_offset,
        dim,
        dst,
        elem_count);
}
