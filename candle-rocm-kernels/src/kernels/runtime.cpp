#include "common.h"

thread_local char HIP_LAST_ERROR[1024] = "no HIP error";

extern "C" const char* hip_last_error() {
    return HIP_LAST_ERROR;
}

extern "C" int hip_device_count(int* count) {
    if (count == nullptr) {
        std::snprintf(HIP_LAST_ERROR, sizeof(HIP_LAST_ERROR), "device_count: null count pointer");
        return 1;
    }
    return set_error("hipGetDeviceCount", hipGetDeviceCount(count));
}

extern "C" int hip_set_device(int ordinal) {
    return select_device(ordinal);
}

extern "C" int hip_malloc(int ordinal, size_t bytes, void** ptr) {
    if (ptr == nullptr) {
        std::snprintf(HIP_LAST_ERROR, sizeof(HIP_LAST_ERROR), "malloc: null output pointer");
        return 1;
    }
    *ptr = nullptr;
    int rc = select_device(ordinal);
    if (rc != 0 || bytes == 0) {
        return rc;
    }
    return set_error("hipMalloc", hipMalloc(ptr, bytes));
}

extern "C" int hip_free(int ordinal, void* ptr) {
    int rc = select_device(ordinal);
    if (rc != 0 || ptr == nullptr) {
        return rc;
    }
    return set_error("hipFree", hipFree(ptr));
}

extern "C" int hip_memset(int ordinal, void* ptr, int value, size_t bytes) {
    int rc = select_device(ordinal);
    if (rc != 0 || bytes == 0) {
        return rc;
    }
    return set_error("hipMemset", hipMemset(ptr, value, bytes));
}

extern "C" int hip_copy_h2d(int ordinal, void* dst, const uint8_t* src, size_t bytes) {
    int rc = select_device(ordinal);
    if (rc != 0 || bytes == 0) {
        return rc;
    }
    return set_error("hipMemcpyHostToDevice", hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
}

extern "C" int hip_copy_d2h(int ordinal, const void* src, uint8_t* dst, size_t bytes) {
    int rc = select_device(ordinal);
    if (rc != 0 || bytes == 0) {
        return rc;
    }
    return set_error("hipMemcpyDeviceToHost", hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost));
}

extern "C" int hip_copy_d2d(
    int dst_ordinal,
    void* dst,
    int src_ordinal,
    const void* src,
    size_t bytes) {
    if (bytes == 0) {
        return 0;
    }
    int rc = select_device(dst_ordinal);
    if (rc != 0) {
        return rc;
    }
    if (dst_ordinal == src_ordinal) {
        return set_error("hipMemcpyDeviceToDevice", hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice));
    }
    return set_error("hipMemcpyPeer", hipMemcpyPeer(dst, dst_ordinal, src, src_ordinal, bytes));
}

extern "C" int hip_synchronize(int ordinal) {
    int rc = select_device(ordinal);
    if (rc != 0) {
        return rc;
    }
    return set_error("hipDeviceSynchronize", hipDeviceSynchronize());
}
