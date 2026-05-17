#include "common.h"

namespace {

__global__ void conv1d_f32_kernel(
    const float* src,
    const size_t* src_strides,
    size_t src_start_offset,
    const float* kernel,
    const size_t* kernel_strides,
    size_t kernel_start_offset,
    float* dst,
    size_t b_size,
    size_t c_out,
    size_t c_in,
    size_t l_in,
    size_t k_size,
    size_t padding,
    size_t stride,
    size_t dilation,
    size_t l_out,
    size_t elem_count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elem_count) {
        return;
    }
    size_t tmp = index;
    const size_t out_x = tmp % l_out;
    tmp /= l_out;
    const size_t co = tmp % c_out;
    tmp /= c_out;
    const size_t b = tmp;

    float sum = 0.0f;
    for (size_t ci = 0; ci < c_in; ++ci) {
        for (size_t k = 0; k < k_size; ++k) {
            const size_t padded_x = out_x * stride + k * dilation;
            if (padded_x < padding) {
                continue;
            }
            const size_t in_x = padded_x - padding;
            if (in_x >= l_in) {
                continue;
            }
            const float v =
                src[src_start_offset + b * src_strides[0] + ci * src_strides[1] +
                    in_x * src_strides[2]];
            const float w = kernel[kernel_start_offset + co * kernel_strides[0] +
                                   ci * kernel_strides[1] + k * kernel_strides[2]];
            sum += v * w;
        }
    }
    dst[index] = sum;
}

__global__ void conv_transpose1d_f32_kernel(
    const float* src,
    const size_t* src_strides,
    size_t src_start_offset,
    const float* kernel,
    const size_t* kernel_strides,
    size_t kernel_start_offset,
    float* dst,
    size_t b_size,
    size_t c_out,
    size_t c_in,
    size_t l_in,
    size_t k_size,
    size_t padding,
    size_t stride,
    size_t dilation,
    size_t l_out,
    size_t elem_count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elem_count) {
        return;
    }
    size_t tmp = index;
    const size_t out_x = tmp % l_out;
    tmp /= l_out;
    const size_t co = tmp % c_out;
    tmp /= c_out;
    const size_t b = tmp;

    float sum = 0.0f;
    for (size_t ci = 0; ci < c_in; ++ci) {
        for (size_t k = 0; k < k_size; ++k) {
            const size_t numerator = out_x + padding;
            const size_t kernel_offset = k * dilation;
            if (numerator < kernel_offset) {
                continue;
            }
            const size_t shifted = numerator - kernel_offset;
            if (shifted % stride != 0) {
                continue;
            }
            const size_t in_x = shifted / stride;
            if (in_x >= l_in) {
                continue;
            }
            const float v =
                src[src_start_offset + b * src_strides[0] + ci * src_strides[1] +
                    in_x * src_strides[2]];
            const float w = kernel[kernel_start_offset + ci * kernel_strides[0] +
                                   co * kernel_strides[1] + k * kernel_strides[2]];
            sum += v * w;
        }
    }
    dst[index] = sum;
}

__global__ void conv2d_f32_kernel(
    const float* src,
    const size_t* src_strides,
    size_t src_start_offset,
    const float* kernel,
    const size_t* kernel_strides,
    size_t kernel_start_offset,
    float* dst,
    size_t b_size,
    size_t c_out,
    size_t c_in,
    size_t in_h,
    size_t in_w,
    size_t k_h,
    size_t k_w,
    size_t padding,
    size_t stride,
    size_t dilation,
    size_t out_h,
    size_t out_w,
    size_t elem_count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elem_count) {
        return;
    }
    size_t tmp = index;
    const size_t ow = tmp % out_w;
    tmp /= out_w;
    const size_t oh = tmp % out_h;
    tmp /= out_h;
    const size_t co = tmp % c_out;
    tmp /= c_out;
    const size_t b = tmp;

    float sum = 0.0f;
    for (size_t ci = 0; ci < c_in; ++ci) {
        for (size_t kh = 0; kh < k_h; ++kh) {
            const size_t padded_h = oh * stride + kh * dilation;
            if (padded_h < padding) {
                continue;
            }
            const size_t ih = padded_h - padding;
            if (ih >= in_h) {
                continue;
            }
            for (size_t kw = 0; kw < k_w; ++kw) {
                const size_t padded_w = ow * stride + kw * dilation;
                if (padded_w < padding) {
                    continue;
                }
                const size_t iw = padded_w - padding;
                if (iw >= in_w) {
                    continue;
                }
                const float v =
                    src[src_start_offset + b * src_strides[0] + ci * src_strides[1] +
                        ih * src_strides[2] + iw * src_strides[3]];
                const float w = kernel[kernel_start_offset + co * kernel_strides[0] +
                                       ci * kernel_strides[1] + kh * kernel_strides[2] +
                                       kw * kernel_strides[3]];
                sum += v * w;
            }
        }
    }
    dst[index] = sum;
}

__global__ void conv_transpose2d_f32_kernel(
    const float* src,
    const size_t* src_strides,
    size_t src_start_offset,
    const float* kernel,
    const size_t* kernel_strides,
    size_t kernel_start_offset,
    float* dst,
    size_t b_size,
    size_t c_out,
    size_t c_in,
    size_t in_h,
    size_t in_w,
    size_t k_h,
    size_t k_w,
    size_t padding,
    size_t stride,
    size_t dilation,
    size_t out_h,
    size_t out_w,
    size_t elem_count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elem_count) {
        return;
    }
    size_t tmp = index;
    const size_t ow = tmp % out_w;
    tmp /= out_w;
    const size_t oh = tmp % out_h;
    tmp /= out_h;
    const size_t co = tmp % c_out;
    tmp /= c_out;
    const size_t b = tmp;

    float sum = 0.0f;
    for (size_t ci = 0; ci < c_in; ++ci) {
        for (size_t kh = 0; kh < k_h; ++kh) {
            const size_t numerator_h = oh + padding;
            const size_t kernel_h = kh * dilation;
            if (numerator_h < kernel_h) {
                continue;
            }
            const size_t shifted_h = numerator_h - kernel_h;
            if (shifted_h % stride != 0) {
                continue;
            }
            const size_t ih = shifted_h / stride;
            if (ih >= in_h) {
                continue;
            }
            for (size_t kw = 0; kw < k_w; ++kw) {
                const size_t numerator_w = ow + padding;
                const size_t kernel_w = kw * dilation;
                if (numerator_w < kernel_w) {
                    continue;
                }
                const size_t shifted_w = numerator_w - kernel_w;
                if (shifted_w % stride != 0) {
                    continue;
                }
                const size_t iw = shifted_w / stride;
                if (iw >= in_w) {
                    continue;
                }
                const float v =
                    src[src_start_offset + b * src_strides[0] + ci * src_strides[1] +
                        ih * src_strides[2] + iw * src_strides[3]];
                const float w = kernel[kernel_start_offset + ci * kernel_strides[0] +
                                       co * kernel_strides[1] + kh * kernel_strides[2] +
                                       kw * kernel_strides[3]];
                sum += v * w;
            }
        }
    }
    dst[index] = sum;
}

int init_layout(DeviceLayout& layout, const size_t* dims, const size_t* strides, size_t rank) {
    return layout.init(dims, strides, rank);
}

} // namespace

extern "C" int hip_conv1d_f32(
    int ordinal,
    const float* src,
    const size_t* src_dims,
    const size_t* src_strides,
    size_t src_rank,
    size_t src_start_offset,
    const float* kernel,
    const size_t* kernel_dims,
    const size_t* kernel_strides,
    size_t kernel_rank,
    size_t kernel_start_offset,
    float* dst,
    size_t padding,
    size_t stride,
    size_t dilation,
    size_t l_out,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    if (src_rank != 3 || kernel_rank != 3) {
        return set_error("conv1d rank", hipErrorInvalidValue);
    }
    DeviceLayout src_layout;
    rc = init_layout(src_layout, src_dims, src_strides, src_rank);
    if (rc != 0) {
        return rc;
    }
    DeviceLayout kernel_layout;
    rc = init_layout(kernel_layout, kernel_dims, kernel_strides, kernel_rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        "conv1d_f32",
        elem_count,
        conv1d_f32_kernel,
        src,
        src_layout.strides,
        src_start_offset,
        kernel,
        kernel_layout.strides,
        kernel_start_offset,
        dst,
        src_dims[0],
        kernel_dims[0],
        src_dims[1],
        src_dims[2],
        kernel_dims[2],
        padding,
        stride,
        dilation,
        l_out,
        elem_count);
}

extern "C" int hip_conv_transpose1d_f32(
    int ordinal,
    const float* src,
    const size_t* src_dims,
    const size_t* src_strides,
    size_t src_rank,
    size_t src_start_offset,
    const float* kernel,
    const size_t* kernel_dims,
    const size_t* kernel_strides,
    size_t kernel_rank,
    size_t kernel_start_offset,
    float* dst,
    size_t padding,
    size_t stride,
    size_t dilation,
    size_t l_out,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    if (src_rank != 3 || kernel_rank != 3) {
        return set_error("conv_transpose1d rank", hipErrorInvalidValue);
    }
    DeviceLayout src_layout;
    rc = init_layout(src_layout, src_dims, src_strides, src_rank);
    if (rc != 0) {
        return rc;
    }
    DeviceLayout kernel_layout;
    rc = init_layout(kernel_layout, kernel_dims, kernel_strides, kernel_rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        "conv_transpose1d_f32",
        elem_count,
        conv_transpose1d_f32_kernel,
        src,
        src_layout.strides,
        src_start_offset,
        kernel,
        kernel_layout.strides,
        kernel_start_offset,
        dst,
        src_dims[0],
        kernel_dims[1],
        src_dims[1],
        src_dims[2],
        kernel_dims[2],
        padding,
        stride,
        dilation,
        l_out,
        elem_count);
}

extern "C" int hip_conv2d_f32(
    int ordinal,
    const float* src,
    const size_t* src_dims,
    const size_t* src_strides,
    size_t src_rank,
    size_t src_start_offset,
    const float* kernel,
    const size_t* kernel_dims,
    const size_t* kernel_strides,
    size_t kernel_rank,
    size_t kernel_start_offset,
    float* dst,
    size_t padding,
    size_t stride,
    size_t dilation,
    size_t out_h,
    size_t out_w,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    if (src_rank != 4 || kernel_rank != 4) {
        return set_error("conv2d rank", hipErrorInvalidValue);
    }
    DeviceLayout src_layout;
    rc = init_layout(src_layout, src_dims, src_strides, src_rank);
    if (rc != 0) {
        return rc;
    }
    DeviceLayout kernel_layout;
    rc = init_layout(kernel_layout, kernel_dims, kernel_strides, kernel_rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        "conv2d_f32",
        elem_count,
        conv2d_f32_kernel,
        src,
        src_layout.strides,
        src_start_offset,
        kernel,
        kernel_layout.strides,
        kernel_start_offset,
        dst,
        src_dims[0],
        kernel_dims[0],
        src_dims[1],
        src_dims[2],
        src_dims[3],
        kernel_dims[2],
        kernel_dims[3],
        padding,
        stride,
        dilation,
        out_h,
        out_w,
        elem_count);
}

extern "C" int hip_conv_transpose2d_f32(
    int ordinal,
    const float* src,
    const size_t* src_dims,
    const size_t* src_strides,
    size_t src_rank,
    size_t src_start_offset,
    const float* kernel,
    const size_t* kernel_dims,
    const size_t* kernel_strides,
    size_t kernel_rank,
    size_t kernel_start_offset,
    float* dst,
    size_t padding,
    size_t stride,
    size_t dilation,
    size_t out_h,
    size_t out_w,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    if (src_rank != 4 || kernel_rank != 4) {
        return set_error("conv_transpose2d rank", hipErrorInvalidValue);
    }
    DeviceLayout src_layout;
    rc = init_layout(src_layout, src_dims, src_strides, src_rank);
    if (rc != 0) {
        return rc;
    }
    DeviceLayout kernel_layout;
    rc = init_layout(kernel_layout, kernel_dims, kernel_strides, kernel_rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        "conv_transpose2d_f32",
        elem_count,
        conv_transpose2d_f32_kernel,
        src,
        src_layout.strides,
        src_start_offset,
        kernel,
        kernel_layout.strides,
        kernel_start_offset,
        dst,
        src_dims[0],
        kernel_dims[1],
        src_dims[1],
        src_dims[2],
        src_dims[3],
        kernel_dims[2],
        kernel_dims[3],
        padding,
        stride,
        dilation,
        out_h,
        out_w,
        elem_count);
}
