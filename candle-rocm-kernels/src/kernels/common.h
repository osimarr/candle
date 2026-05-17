#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

extern thread_local char HIP_LAST_ERROR[1024];

inline int set_error(const char* op, hipError_t error) {
    if (error == hipSuccess) {
        HIP_LAST_ERROR[0] = '\0';
        return 0;
    }
    std::snprintf(
        HIP_LAST_ERROR,
        sizeof(HIP_LAST_ERROR),
        "%s: %s",
        op,
        hipGetErrorString(error));
    return static_cast<int>(error);
}

inline int check_last_kernel_error(const char* op) {
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) {
        return set_error(op, error);
    }
    return set_error(op, hipDeviceSynchronize());
}

inline int check_last_launch_error(const char* op) {
    return set_error(op, hipGetLastError());
}

struct DeviceLayout {
    static constexpr size_t MAX_RANK = 32;
    size_t dims[MAX_RANK] = {};
    size_t strides[MAX_RANK] = {};
    size_t rank = 0;

    int init(const size_t* host_dims, const size_t* host_strides, size_t layout_rank) {
        if (layout_rank > MAX_RANK) {
            return set_error("layout rank", hipErrorInvalidValue);
        }
        rank = layout_rank;
        for (size_t i = 0; i < rank; ++i) {
            dims[i] = host_dims[i];
            strides[i] = host_strides[i];
        }
        return 0;
    }
};

__device__ inline size_t storage_index(
    size_t logical_index,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset) {
    size_t storage = start_offset;
    for (size_t rev = 0; rev < rank; ++rev) {
        const size_t axis = rank - 1 - rev;
        const size_t dim = dims[axis];
        if (dim == 0) {
            return start_offset;
        }
        const size_t index = logical_index % dim;
        logical_index /= dim;
        storage += index * strides[axis];
    }
    return storage;
}

__device__ inline size_t storage_index(
    size_t logical_index,
    const DeviceLayout& layout,
    size_t start_offset) {
    return storage_index(logical_index, layout.dims, layout.strides, layout.rank, start_offset);
}

__device__ inline uint16_t f32_to_bf16_bits(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t lsb = (bits >> 16) & 1U;
    const uint32_t rounding_bias = 0x7fffU + lsb;
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

__device__ inline float bf16_bits_to_f32(uint16_t value) {
    return __uint_as_float(static_cast<uint32_t>(value) << 16);
}

template <typename T>
__device__ inline float to_f32(T value) {
    return static_cast<float>(value);
}

template <>
__device__ inline float to_f32<uint16_t>(uint16_t value) {
    return bf16_bits_to_f32(value);
}

template <typename T>
__device__ inline T from_f32(float value) {
    return static_cast<T>(value);
}

template <>
__device__ inline uint16_t from_f32<uint16_t>(float value) {
    return f32_to_bf16_bits(value);
}

inline dim3 grid_for(size_t elem_count) {
    const unsigned int threads = 256;
    const unsigned int blocks =
        static_cast<unsigned int>((elem_count + threads - 1) / threads);
    return dim3(blocks == 0 ? 1 : blocks);
}

template <typename Kernel, typename... Args>
int launch_1d(const char* op, size_t elem_count, Kernel kernel, Args... args) {
    const dim3 block(256);
    hipLaunchKernelGGL(kernel, grid_for(elem_count), block, 0, 0, args...);
    return check_last_launch_error(op);
}

template <typename Kernel, typename... Args>
int launch_1d_async(const char* op, size_t elem_count, Kernel kernel, Args... args) {
    const dim3 block(256);
    hipLaunchKernelGGL(kernel, grid_for(elem_count), block, 0, 0, args...);
    return check_last_launch_error(op);
}

inline int select_device(int ordinal) {
    return set_error("hipSetDevice", hipSetDevice(ordinal));
}
