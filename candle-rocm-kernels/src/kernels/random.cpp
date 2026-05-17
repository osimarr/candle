#include "common.h"

#include <cmath>
#include <cstdint>

namespace {

__device__ uint64_t splitmix64(uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

__device__ float uniform01(uint64_t seed, size_t index) {
    const uint64_t value = splitmix64(seed + index);
    const uint32_t mantissa = static_cast<uint32_t>(value >> 40);
    return static_cast<float>(mantissa) * (1.0f / 16777216.0f);
}

__global__ void random_uniform_f32_kernel(
    float* dst,
    size_t elem_count,
    uint64_t seed,
    float lo,
    float scale) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elem_count) {
        return;
    }
    dst[index] = lo + uniform01(seed, index) * scale;
}

__global__ void random_normal_f32_kernel(
    float* dst,
    size_t elem_count,
    uint64_t seed,
    float mean,
    float std) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elem_count) {
        return;
    }
    const float u1 = fmaxf(uniform01(seed, index * 2), 1.0e-7f);
    const float u2 = uniform01(seed, index * 2 + 1);
    const float radius = sqrtf(-2.0f * logf(u1));
    const float theta = 6.2831853071795864769f * u2;
    dst[index] = mean + std * radius * cosf(theta);
}

} // namespace

extern "C" int hip_random_uniform_f32(
    int ordinal,
    float* dst,
    size_t elem_count,
    uint64_t seed,
    float lo,
    float up) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    return launch_1d_async(
        "random_uniform_f32",
        elem_count,
        random_uniform_f32_kernel,
        dst,
        elem_count,
        seed,
        lo,
        up - lo);
}

extern "C" int hip_random_normal_f32(
    int ordinal,
    float* dst,
    size_t elem_count,
    uint64_t seed,
    float mean,
    float std) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    return launch_1d_async(
        "random_normal_f32",
        elem_count,
        random_normal_f32_kernel,
        dst,
        elem_count,
        seed,
        mean,
        std);
}
