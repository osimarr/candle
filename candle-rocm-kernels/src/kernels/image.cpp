#include "common.h"

#include <cmath>

namespace {

template <typename T>
__global__ void pool2d_kernel(
    int op,
    const T* src,
    DeviceLayout layout,
    size_t start_offset,
    T* dst,
    size_t k_h,
    size_t k_w,
    size_t s_h,
    size_t s_w,
    size_t out_h,
    size_t out_w,
    size_t elem_count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elem_count) {
        return;
    }
    size_t tmp = index;
    const size_t out_w_idx = tmp % out_w;
    tmp /= out_w;
    const size_t out_h_idx = tmp % out_h;
    tmp /= out_h;
    const size_t c_idx = tmp % layout.dims[1];
    tmp /= layout.dims[1];
    const size_t b_idx = tmp;

    const size_t base = start_offset + b_idx * layout.strides[0] + c_idx * layout.strides[1];
    float value = to_f32(src[base + out_h_idx * s_h * layout.strides[2] +
                             out_w_idx * s_w * layout.strides[3]]);
    if (op == 1) {
        value = 0.0f;
    }
    for (size_t kh = 0; kh < k_h; ++kh) {
        const size_t src_h = out_h_idx * s_h + kh;
        for (size_t kw = 0; kw < k_w; ++kw) {
            const size_t src_w = out_w_idx * s_w + kw;
            const float current =
                to_f32(src[base + src_h * layout.strides[2] + src_w * layout.strides[3]]);
            if (op == 1) {
                value += current;
            } else {
                value = fmaxf(value, current);
            }
        }
    }
    if (op == 1) {
        value /= static_cast<float>(k_h * k_w);
    }
    dst[index] = from_f32<T>(value);
}

template <typename T>
__global__ void upsample_nearest1d_kernel(
    const T* src,
    DeviceLayout layout,
    size_t start_offset,
    T* dst,
    size_t out_size,
    size_t elem_count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elem_count) {
        return;
    }
    size_t tmp = index;
    const size_t out_idx = tmp % out_size;
    tmp /= out_size;
    const size_t c_idx = tmp % layout.dims[1];
    tmp /= layout.dims[1];
    const size_t b_idx = tmp;
    const double scale = static_cast<double>(layout.dims[2]) / static_cast<double>(out_size);
    const size_t src_idx =
        min(layout.dims[2] - 1, static_cast<size_t>(static_cast<double>(out_idx) * scale));
    dst[index] = src[start_offset + b_idx * layout.strides[0] + c_idx * layout.strides[1] +
                     src_idx * layout.strides[2]];
}

template <typename T>
__global__ void upsample_nearest2d_kernel(
    const T* src,
    DeviceLayout layout,
    size_t start_offset,
    T* dst,
    size_t out_h,
    size_t out_w,
    size_t elem_count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elem_count) {
        return;
    }
    size_t tmp = index;
    const size_t out_w_idx = tmp % out_w;
    tmp /= out_w;
    const size_t out_h_idx = tmp % out_h;
    tmp /= out_h;
    const size_t c_idx = tmp % layout.dims[1];
    tmp /= layout.dims[1];
    const size_t b_idx = tmp;
    const double scale_h = static_cast<double>(layout.dims[2]) / static_cast<double>(out_h);
    const double scale_w = static_cast<double>(layout.dims[3]) / static_cast<double>(out_w);
    const size_t src_h =
        min(layout.dims[2] - 1, static_cast<size_t>(static_cast<double>(out_h_idx) * scale_h));
    const size_t src_w =
        min(layout.dims[3] - 1, static_cast<size_t>(static_cast<double>(out_w_idx) * scale_w));
    dst[index] = src[start_offset + b_idx * layout.strides[0] + c_idx * layout.strides[1] +
                     src_h * layout.strides[2] + src_w * layout.strides[3]];
}

template <typename T>
__global__ void upsample_bilinear2d_kernel(
    const T* src,
    DeviceLayout layout,
    size_t start_offset,
    T* dst,
    size_t out_h,
    size_t out_w,
    double scale_h,
    double scale_w,
    bool align_corners,
    size_t elem_count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elem_count) {
        return;
    }
    size_t tmp = index;
    const size_t out_w_idx = tmp % out_w;
    tmp /= out_w;
    const size_t out_h_idx = tmp % out_h;
    tmp /= out_h;
    const size_t c_idx = tmp % layout.dims[1];
    tmp /= layout.dims[1];
    const size_t b_idx = tmp;

    const double raw_h = align_corners ? scale_h * static_cast<double>(out_h_idx)
                                      : scale_h * (static_cast<double>(out_h_idx) + 0.5) - 0.5;
    const double raw_w = align_corners ? scale_w * static_cast<double>(out_w_idx)
                                      : scale_w * (static_cast<double>(out_w_idx) + 0.5) - 0.5;
    const double src_h = fmax(raw_h, 0.0);
    const double src_w = fmax(raw_w, 0.0);
    const size_t h0 = static_cast<size_t>(floor(src_h));
    const size_t w0 = static_cast<size_t>(floor(src_w));
    const size_t h1 = min(h0 + 1, layout.dims[2] - 1);
    const size_t w1 = min(w0 + 1, layout.dims[3] - 1);
    const double weight_h = fmin(fmax(src_h - static_cast<double>(h0), 0.0), 1.0);
    const double weight_w = fmin(fmax(src_w - static_cast<double>(w0), 0.0), 1.0);

    const size_t base = start_offset + b_idx * layout.strides[0] + c_idx * layout.strides[1];
    const double v00 =
        static_cast<double>(to_f32(src[base + h0 * layout.strides[2] + w0 * layout.strides[3]]));
    const double v10 =
        static_cast<double>(to_f32(src[base + h0 * layout.strides[2] + w1 * layout.strides[3]]));
    const double v01 =
        static_cast<double>(to_f32(src[base + h1 * layout.strides[2] + w0 * layout.strides[3]]));
    const double v11 =
        static_cast<double>(to_f32(src[base + h1 * layout.strides[2] + w1 * layout.strides[3]]));
    const double v_top = v00 * (1.0 - weight_w) + v10 * weight_w;
    const double v_bottom = v01 * (1.0 - weight_w) + v11 * weight_w;
    dst[index] = from_f32<T>(static_cast<float>(v_top * (1.0 - weight_h) + v_bottom * weight_h));
}

int init_4d_layout(
    DeviceLayout& layout,
    const size_t* dims,
    const size_t* strides,
    size_t rank) {
    if (rank != 4) {
        return set_error("image layout rank", hipErrorInvalidValue);
    }
    return layout.init(dims, strides, rank);
}

} // namespace

extern "C" int hip_pool2d_f32(
    int ordinal,
    int op,
    const float* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    float* dst,
    size_t k_h,
    size_t k_w,
    size_t s_h,
    size_t s_w,
    size_t out_h,
    size_t out_w,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    DeviceLayout layout;
    rc = init_4d_layout(layout, dims, strides, rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        "pool2d_f32",
        elem_count,
        pool2d_kernel<float>,
        op,
        src,
        layout,
        start_offset,
        dst,
        k_h,
        k_w,
        s_h,
        s_w,
        out_h,
        out_w,
        elem_count);
}

extern "C" int hip_upsample_nearest1d_f32(
    int ordinal,
    const float* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    float* dst,
    size_t out_size,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    DeviceLayout layout;
    if (rank != 3) {
        return set_error("upsample_nearest1d layout rank", hipErrorInvalidValue);
    }
    rc = layout.init(dims, strides, rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        "upsample_nearest1d_f32",
        elem_count,
        upsample_nearest1d_kernel<float>,
        src,
        layout,
        start_offset,
        dst,
        out_size,
        elem_count);
}

extern "C" int hip_upsample_nearest2d_f32(
    int ordinal,
    const float* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    float* dst,
    size_t out_h,
    size_t out_w,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    DeviceLayout layout;
    rc = init_4d_layout(layout, dims, strides, rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        "upsample_nearest2d_f32",
        elem_count,
        upsample_nearest2d_kernel<float>,
        src,
        layout,
        start_offset,
        dst,
        out_h,
        out_w,
        elem_count);
}

extern "C" int hip_upsample_bilinear2d_f32(
    int ordinal,
    const float* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    float* dst,
    size_t out_h,
    size_t out_w,
    double scale_h,
    double scale_w,
    int align_corners,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    DeviceLayout layout;
    rc = init_4d_layout(layout, dims, strides, rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        "upsample_bilinear2d_f32",
        elem_count,
        upsample_bilinear2d_kernel<float>,
        src,
        layout,
        start_offset,
        dst,
        out_h,
        out_w,
        scale_h,
        scale_w,
        align_corners != 0,
        elem_count);
}

extern "C" int hip_pool2d_bf16(
    int ordinal,
    int op,
    const uint16_t* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    uint16_t* dst,
    size_t k_h,
    size_t k_w,
    size_t s_h,
    size_t s_w,
    size_t out_h,
    size_t out_w,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    DeviceLayout layout;
    rc = init_4d_layout(layout, dims, strides, rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        "pool2d_bf16",
        elem_count,
        pool2d_kernel<uint16_t>,
        op,
        src,
        layout,
        start_offset,
        dst,
        k_h,
        k_w,
        s_h,
        s_w,
        out_h,
        out_w,
        elem_count);
}

extern "C" int hip_upsample_nearest1d_bf16(
    int ordinal,
    const uint16_t* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    uint16_t* dst,
    size_t out_size,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    DeviceLayout layout;
    if (rank != 3) {
        return set_error("upsample_nearest1d layout rank", hipErrorInvalidValue);
    }
    rc = layout.init(dims, strides, rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        "upsample_nearest1d_bf16",
        elem_count,
        upsample_nearest1d_kernel<uint16_t>,
        src,
        layout,
        start_offset,
        dst,
        out_size,
        elem_count);
}

extern "C" int hip_upsample_nearest2d_bf16(
    int ordinal,
    const uint16_t* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    uint16_t* dst,
    size_t out_h,
    size_t out_w,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    DeviceLayout layout;
    rc = init_4d_layout(layout, dims, strides, rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        "upsample_nearest2d_bf16",
        elem_count,
        upsample_nearest2d_kernel<uint16_t>,
        src,
        layout,
        start_offset,
        dst,
        out_h,
        out_w,
        elem_count);
}

extern "C" int hip_upsample_bilinear2d_bf16(
    int ordinal,
    const uint16_t* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    uint16_t* dst,
    size_t out_h,
    size_t out_w,
    double scale_h,
    double scale_w,
    int align_corners,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    DeviceLayout layout;
    rc = init_4d_layout(layout, dims, strides, rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        "upsample_bilinear2d_bf16",
        elem_count,
        upsample_bilinear2d_kernel<uint16_t>,
        src,
        layout,
        start_offset,
        dst,
        out_h,
        out_w,
        scale_h,
        scale_w,
        align_corners != 0,
        elem_count);
}

extern "C" int hip_pool2d_f8e4m3(
    int ordinal,
    int op,
    const uint8_t* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    uint8_t* dst,
    size_t k_h,
    size_t k_w,
    size_t s_h,
    size_t s_w,
    size_t out_h,
    size_t out_w,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    DeviceLayout layout;
    rc = init_4d_layout(layout, dims, strides, rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        "pool2d_f8e4m3",
        elem_count,
        pool2d_kernel<F8E4M3Storage>,
        op,
        as_f8e4m3(src),
        layout,
        start_offset,
        as_f8e4m3(dst),
        k_h,
        k_w,
        s_h,
        s_w,
        out_h,
        out_w,
        elem_count);
}

extern "C" int hip_upsample_nearest1d_f8e4m3(
    int ordinal,
    const uint8_t* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    uint8_t* dst,
    size_t out_size,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    DeviceLayout layout;
    if (rank != 3) {
        return set_error("upsample_nearest1d layout rank", hipErrorInvalidValue);
    }
    rc = layout.init(dims, strides, rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        "upsample_nearest1d_f8e4m3",
        elem_count,
        upsample_nearest1d_kernel<F8E4M3Storage>,
        as_f8e4m3(src),
        layout,
        start_offset,
        as_f8e4m3(dst),
        out_size,
        elem_count);
}

extern "C" int hip_upsample_nearest2d_f8e4m3(
    int ordinal,
    const uint8_t* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    uint8_t* dst,
    size_t out_h,
    size_t out_w,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    DeviceLayout layout;
    rc = init_4d_layout(layout, dims, strides, rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        "upsample_nearest2d_f8e4m3",
        elem_count,
        upsample_nearest2d_kernel<F8E4M3Storage>,
        as_f8e4m3(src),
        layout,
        start_offset,
        as_f8e4m3(dst),
        out_h,
        out_w,
        elem_count);
}

extern "C" int hip_upsample_bilinear2d_f8e4m3(
    int ordinal,
    const uint8_t* src,
    const size_t* dims,
    const size_t* strides,
    size_t rank,
    size_t start_offset,
    uint8_t* dst,
    size_t out_h,
    size_t out_w,
    double scale_h,
    double scale_w,
    int align_corners,
    size_t elem_count) {
    int rc = select_device(ordinal);
    if (rc != 0 || elem_count == 0) {
        return rc;
    }
    DeviceLayout layout;
    rc = init_4d_layout(layout, dims, strides, rank);
    if (rc != 0) {
        return rc;
    }
    return launch_1d(
        "upsample_bilinear2d_f8e4m3",
        elem_count,
        upsample_bilinear2d_kernel<F8E4M3Storage>,
        as_f8e4m3(src),
        layout,
        start_offset,
        as_f8e4m3(dst),
        out_h,
        out_w,
        scale_h,
        scale_w,
        align_corners != 0,
        elem_count);
}
