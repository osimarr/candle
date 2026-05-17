#include "common.h"

namespace {

template <typename T>
__global__ void const_set_kernel(
    T* dst,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    T value,
    size_t elem_count) {
    const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical_index >= elem_count) {
        return;
    }
    const size_t dst_index =
        storage_index(logical_index, dims, strides, rank, start_offset);
    dst[dst_index] = value;
}

template <typename T>
int const_set(
    const char* name,
    int ordinal,
    T* dst,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    T value,
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
        name,
        elem_count,
        const_set_kernel<T>,
        dst,
        layout.dims,
        layout.strides,
        layout.rank,
        start_offset,
        value,
        elem_count);
}

} // namespace

extern "C" int hip_const_set_f32(
    int ordinal,
    float* dst,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    float value,
    size_t elem_count) {
    return const_set("const_set_f32", ordinal, dst, dims, strides, rank, start_offset, value, elem_count);
}

extern "C" int hip_const_set_u8(
    int ordinal,
    uint8_t* dst,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    uint8_t value,
    size_t elem_count) {
    return const_set("const_set_u8", ordinal, dst, dims, strides, rank, start_offset, value, elem_count);
}

extern "C" int hip_const_set_u32(
    int ordinal,
    uint32_t* dst,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    uint32_t value,
    size_t elem_count) {
    return const_set("const_set_u32", ordinal, dst, dims, strides, rank, start_offset, value, elem_count);
}
