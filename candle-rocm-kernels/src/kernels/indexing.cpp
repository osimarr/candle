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

template <typename I>
__global__ void index_select_f32_kernel(
    const float* src,
    const size_t* src_dims,
    const size_t* src_strides,
    size_t rank,
    size_t src_start_offset,
    const I* ids,
    size_t ids_start_offset,
    size_t ids_stride,
    size_t dim,
    size_t n_ids,
    float* dst,
    size_t elem_count) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }

    size_t tmp = logical_index;
    size_t src_index = src_start_offset;
    size_t selected_id_pos = 0;
    for (size_t rev = 0; rev < rank; ++rev) {
        const size_t axis = rank - 1 - rev;
        const size_t out_dim = axis == dim ? n_ids : src_dims[axis];
        const size_t coord = tmp % out_dim;
        tmp /= out_dim;
        if (axis == dim) {
            selected_id_pos = coord;
        } else {
            src_index += coord * src_strides[axis];
        }
    }

    const I raw_index = ids[ids_start_offset + selected_id_pos * ids_stride];
    if (is_missing_index<I>(raw_index)) {
        dst[logical_index] = 0.0f;
        return;
    }
    const size_t selected = static_cast<size_t>(raw_index);
    dst[logical_index] = src[src_index + selected * src_strides[dim]];
}

template <typename I>
__global__ void gather_f32_kernel(
    const float* src,
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
    float* dst,
    size_t elem_count) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }

    const size_t ids_index =
        storage_index(logical_index, ids_dims, ids_strides, ids_rank, ids_start_offset);
    const I raw_index = ids[ids_index];
    if (is_missing_index<I>(raw_index)) {
        dst[logical_index] = 0.0f;
        return;
    }

    size_t tmp = logical_index;
    size_t src_index = src_start_offset;
    for (size_t rev = 0; rev < src_rank; ++rev) {
        const size_t axis = src_rank - 1 - rev;
        const size_t coord = tmp % ids_dims[axis];
        tmp /= ids_dims[axis];
        src_index += (axis == dim ? static_cast<size_t>(raw_index) : coord) * src_strides[axis];
    }
    dst[logical_index] = src[src_index];
}

template <typename I>
__global__ void scatter_f32_kernel(
    int add,
    float* dst,
    const size_t* dst_dims,
    const size_t* dst_strides,
    size_t dst_rank,
    size_t dst_start_offset,
    const I* ids,
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
    const size_t dst_logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst_logical_index >= dst_elem_count) {
        return;
    }
    const size_t dst_storage_index = storage_index(
        dst_logical_index,
        dst_dims,
        dst_strides,
        dst_rank,
        dst_start_offset);
    float value = dst[dst_storage_index];

    for (size_t src_logical_index = 0; src_logical_index < src_elem_count;
         ++src_logical_index) {
        const size_t ids_index = storage_index(
            src_logical_index,
            ids_dims,
            ids_strides,
            ids_rank,
            ids_start_offset);
        const I raw_index = ids[ids_index];
        if (is_missing_index<I>(raw_index)) {
            continue;
        }
        size_t src_tmp = src_logical_index;
        size_t dst_tmp = dst_logical_index;
        bool matches = true;
        for (size_t rev = 0; rev < dst_rank; ++rev) {
            const size_t axis = dst_rank - 1 - rev;
            const size_t src_coord = src_tmp % src_dims[axis];
            src_tmp /= src_dims[axis];
            const size_t dst_coord = dst_tmp % dst_dims[axis];
            dst_tmp /= dst_dims[axis];
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
            const float src_value = src[storage_index(
                src_logical_index,
                src_dims,
                src_strides,
                src_rank,
                src_start_offset)];
            if (add) {
                value += src_value;
            } else {
                value = src_value;
            }
        }
    }
    dst[dst_storage_index] = value;
}

template <typename I>
__global__ void index_add_f32_kernel(
    const float* input,
    const size_t* input_dims,
    const size_t* input_strides,
    size_t input_rank,
    size_t input_start_offset,
    const I* ids,
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
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }
    float value = input[storage_index(
        logical_index,
        input_dims,
        input_strides,
        input_rank,
        input_start_offset)];

    size_t dst_dim_coord = logical_index;
    for (size_t rev = 0; rev < input_rank; ++rev) {
        const size_t axis = input_rank - 1 - rev;
        const size_t coord = dst_dim_coord % input_dims[axis];
        dst_dim_coord /= input_dims[axis];
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
        for (size_t rev = 0; rev < input_rank; ++rev) {
            const size_t axis = input_rank - 1 - rev;
            const size_t coord = tmp % input_dims[axis];
            tmp /= input_dims[axis];
            src_index += (axis == dim ? id_pos : coord) * src_strides[axis];
        }
        value += src[src_index];
    }
    dst[logical_index] = value;
}

template <typename I>
int index_select_f32(
    const char* name,
    int ordinal,
    const float* src,
    const size_t* src_dims,
    const size_t* src_strides,
    size_t rank,
    size_t src_start_offset,
    const I* ids,
    size_t ids_start_offset,
    size_t ids_stride,
    size_t dim,
    size_t n_ids,
    float* dst,
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
        index_select_f32_kernel<I>,
        src,
        src_layout.dims,
        src_layout.strides,
        src_layout.rank,
        src_start_offset,
        ids,
        ids_start_offset,
        ids_stride,
        dim,
        n_ids,
        dst,
        elem_count);
}

template <typename I>
int gather_f32(
    const char* name,
    int ordinal,
    const float* src,
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
    float* dst,
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
        gather_f32_kernel<I>,
        src,
        src_layout.dims,
        src_layout.strides,
        src_layout.rank,
        src_start_offset,
        ids,
        ids_layout.dims,
        ids_layout.strides,
        ids_layout.rank,
        ids_start_offset,
        dim,
        dst,
        elem_count);
}

template <typename I>
int scatter_f32(
    const char* name,
    int ordinal,
    int add,
    float* dst,
    const size_t* dst_dims,
    const size_t* dst_strides,
    size_t dst_rank,
    size_t dst_start_offset,
    const I* ids,
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
        scatter_f32_kernel<I>,
        add,
        dst,
        dst_layout.dims,
        dst_layout.strides,
        dst_layout.rank,
        dst_start_offset,
        ids,
        ids_layout.dims,
        ids_layout.strides,
        ids_layout.rank,
        ids_start_offset,
        src,
        src_layout.dims,
        src_layout.strides,
        src_layout.rank,
        src_start_offset,
        dim,
        src_elem_count,
        dst_elem_count);
}

template <typename I>
int index_add_f32(
    const char* name,
    int ordinal,
    const float* input,
    const size_t* input_dims,
    const size_t* input_strides,
    size_t input_rank,
    size_t input_start_offset,
    const I* ids,
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
        index_add_f32_kernel<I>,
        input,
        input_layout.dims,
        input_layout.strides,
        input_layout.rank,
        input_start_offset,
        ids,
        ids_start_offset,
        ids_stride,
        ids_len,
        src,
        src_layout.dims,
        src_layout.strides,
        src_layout.rank,
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
    return index_select_f32(
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
    return index_select_f32(
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
    return gather_f32(
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
    return scatter_f32(
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
    return index_add_f32(
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
