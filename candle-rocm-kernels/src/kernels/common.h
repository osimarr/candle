#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
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

inline void ignore_error(hipError_t error) {
    (void)error;
}

struct DeviceLayout {
    size_t* dims = nullptr;
    size_t* strides = nullptr;
    size_t rank = 0;

    int init(const size_t* host_dims, const size_t* host_strides, size_t layout_rank) {
        rank = layout_rank;
        if (rank == 0) {
            return 0;
        }
        hipError_t error = hipMalloc(&dims, rank * sizeof(size_t));
        if (error != hipSuccess) {
            return set_error("hipMalloc layout dims", error);
        }
        error = hipMalloc(&strides, rank * sizeof(size_t));
        if (error != hipSuccess) {
            return set_error("hipMalloc layout strides", error);
        }
        error = hipMemcpy(dims, host_dims, rank * sizeof(size_t), hipMemcpyHostToDevice);
        if (error != hipSuccess) {
            return set_error("hipMemcpy layout dims", error);
        }
        error = hipMemcpy(strides, host_strides, rank * sizeof(size_t), hipMemcpyHostToDevice);
        if (error != hipSuccess) {
            return set_error("hipMemcpy layout strides", error);
        }
        return 0;
    }

    ~DeviceLayout() {
        if (dims != nullptr) {
            ignore_error(hipFree(dims));
        }
        if (strides != nullptr) {
            ignore_error(hipFree(strides));
        }
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
    return check_last_kernel_error(op);
}

inline int select_device(int ordinal) {
    return set_error("hipSetDevice", hipSetDevice(ordinal));
}
