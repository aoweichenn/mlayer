// Copyright 2021 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_POOLING3D_HPP
#define MLAYER_POOLING3D_HPP

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>

namespace mlayer {

// 3D pooling over (w, h, d, channels) blobs.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate = false>
class Pooling3D : public LayerT {
public:
    Pooling3D();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

public:
    std::int32_t pooling_type;
    std::int32_t kernel_w;
    std::int32_t kernel_h;
    std::int32_t kernel_d;
    std::int32_t stride_w;
    std::int32_t stride_h;
    std::int32_t stride_d;
    std::int32_t pad_left;
    std::int32_t pad_right;
    std::int32_t pad_top;
    std::int32_t pad_bottom;
    std::int32_t pad_front;
    std::int32_t pad_behind;
    std::int32_t global_pooling;
    std::int32_t pad_mode;
    std::int32_t avgpool_count_include_pad;
    std::int32_t adaptive_pooling;
    std::int32_t out_w;
    std::int32_t out_h;
    std::int32_t out_d;
};

#define MLAYER_POOLING3D_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate>

MLAYER_POOLING3D_TEMPLATE
Pooling3D<LayerT, MatT, OptionT, ParamDictT, Validate>::Pooling3D() {
    this->one_blob_only = true;
    this->support_inplace = false;
    this->pooling_type = 0;
    this->kernel_w = 0;
    this->kernel_h = 0;
    this->kernel_d = 0;
    this->stride_w = 1;
    this->stride_h = 1;
    this->stride_d = 1;
    this->pad_left = 0;
    this->pad_right = 0;
    this->pad_top = 0;
    this->pad_bottom = 0;
    this->pad_front = 0;
    this->pad_behind = 0;
    this->global_pooling = 0;
    this->pad_mode = 0;
    this->avgpool_count_include_pad = 0;
    this->adaptive_pooling = 0;
    this->out_w = 0;
    this->out_h = 0;
    this->out_d = 0;
}

MLAYER_POOLING3D_TEMPLATE
std::int32_t Pooling3D<LayerT, MatT, OptionT, ParamDictT, Validate>::load_param(const ParamDictT& pd) {
    this->pooling_type = pd.get(0, 0);
    this->kernel_w = pd.get(1, 0);
    this->kernel_h = pd.get(11, this->kernel_w);
    this->kernel_d = pd.get(21, this->kernel_w);
    this->stride_w = pd.get(2, 1);
    this->stride_h = pd.get(12, this->stride_w);
    this->stride_d = pd.get(22, this->stride_w);
    this->pad_left = pd.get(3, 0);
    this->pad_right = pd.get(14, this->pad_left);
    this->pad_top = pd.get(13, this->pad_left);
    this->pad_bottom = pd.get(15, this->pad_top);
    this->pad_front = pd.get(23, this->pad_left);
    this->pad_behind = pd.get(16, this->pad_front);
    this->global_pooling = pd.get(4, 0);
    this->pad_mode = pd.get(5, 0);
    this->avgpool_count_include_pad = pd.get(6, 0);
    this->adaptive_pooling = pd.get(7, 0);
    this->out_w = pd.get(8, 0);
    this->out_h = pd.get(18, this->out_w);
    this->out_d = pd.get(28, this->out_w);
    return 0;
}

MLAYER_POOLING3D_TEMPLATE
std::int32_t Pooling3D<LayerT, MatT, OptionT, ParamDictT, Validate>::forward(const MatT& bottom_blob,
                                                                           MatT& top_blob,
                                                                           const OptionT& opt) const {
    const std::int32_t w = bottom_blob.w;
    const std::int32_t h = bottom_blob.h;
    const std::int32_t d = bottom_blob.d;
    const std::int32_t channels = bottom_blob.c;
    const std::size_t elemsize = bottom_blob.elemsize;

    if (this->global_pooling != 0) {
        top_blob.create(channels, elemsize, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }
#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; ++q) {
            const float* ptr = bottom_blob.channel(q);
            if (this->pooling_type == 0) {
                float m = ptr[0];
                for (std::int32_t i = 1; i < w * h * d; ++i) {
                    m = std::max(m, ptr[i]);
                }
                top_blob[q] = m;
            } else {
                float s = 0.0F;
                for (std::int32_t i = 0; i < w * h * d; ++i) {
                    s += ptr[i];
                }
                top_blob[q] = s / static_cast<float>(w * h * d);
            }
        }
        return 0;
    }

    std::int32_t outw = 0;
    std::int32_t outh = 0;
    std::int32_t outd = 0;
    std::int32_t padtail_w = 0;
    std::int32_t padtail_h = 0;
    std::int32_t padtail_d = 0;
    if (this->adaptive_pooling != 0) {
        outw = this->out_w;
        outh = this->out_h;
        outd = this->out_d;
    } else if (this->pad_mode == 0) {
        if (this->kernel_w <= 0) {
            return -1;
        }
        // ncnn "full" padding extends the right/bottom/behind side so the
        // window fits an integer number of strides.
        const std::int32_t wtail = (w + this->pad_left + this->pad_right - this->kernel_w) % this->stride_w;
        const std::int32_t htail = (h + this->pad_top + this->pad_bottom - this->kernel_h) % this->stride_h;
        const std::int32_t dtail = (d + this->pad_front + this->pad_behind - this->kernel_d) % this->stride_d;
        padtail_w = wtail != 0 ? this->stride_w - wtail : 0;
        padtail_h = htail != 0 ? this->stride_h - htail : 0;
        padtail_d = dtail != 0 ? this->stride_d - dtail : 0;
        outw = (w + this->pad_left + this->pad_right + padtail_w - this->kernel_w) / this->stride_w + 1;
        outh = (h + this->pad_top + this->pad_bottom + padtail_h - this->kernel_h) / this->stride_h + 1;
        outd = (d + this->pad_front + this->pad_behind + padtail_d - this->kernel_d) / this->stride_d + 1;
    } else if (this->pad_mode == 1) {
        outw = (w - this->kernel_w) / this->stride_w + 1;
        outh = (h - this->kernel_h) / this->stride_h + 1;
        outd = (d - this->kernel_d) / this->stride_d + 1;
    } else {
        return -1;
    }
    if (outw <= 0 || outh <= 0 || outd <= 0) {
        return -1;
    }

    top_blob.create(outw, outh, outd, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty()) {
        return -100;
    }

#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t q = 0; q < channels; ++q) {
        const MatT image = bottom_blob.channel(q);
        MatT out = top_blob.channel(q);
        for (std::int32_t oz = 0; oz < outd; ++oz) {
            for (std::int32_t oy = 0; oy < outh; ++oy) {
                for (std::int32_t ox = 0; ox < outw; ++ox) {
                    std::int32_t ws = 0;
                    std::int32_t hs = 0;
                    std::int32_t ds = 0;
                    std::int32_t we = 0;
                    std::int32_t he = 0;
                    std::int32_t de = 0;
                    std::int32_t ksize = 0;
                    if (this->adaptive_pooling != 0) {
                        ws = static_cast<std::int32_t>(std::floor(static_cast<float>(ox) * w / outw));
                        we = static_cast<std::int32_t>(std::ceil(static_cast<float>(ox + 1) * w / outw));
                        hs = static_cast<std::int32_t>(std::floor(static_cast<float>(oy) * h / outh));
                        he = static_cast<std::int32_t>(std::ceil(static_cast<float>(oy + 1) * h / outh));
                        ds = static_cast<std::int32_t>(std::floor(static_cast<float>(oz) * d / outd));
                        de = static_cast<std::int32_t>(std::ceil(static_cast<float>(oz + 1) * d / outd));
                        ksize = (we - ws) * (he - hs) * (de - ds);
                    } else {
                        ws = ox * this->stride_w - this->pad_left;
                        hs = oy * this->stride_h - this->pad_top;
                        ds = oz * this->stride_d - this->pad_front;
                        we = ws + this->kernel_w;
                        he = hs + this->kernel_h;
                        de = ds + this->kernel_d;
                        static_cast<void>(padtail_w);
                        static_cast<void>(padtail_h);
                        static_cast<void>(padtail_d);
                        ksize = this->kernel_w * this->kernel_h * this->kernel_d;
                    }
                    const std::int32_t ws2 = std::max(ws, 0);
                    const std::int32_t hs2 = std::max(hs, 0);
                    const std::int32_t ds2 = std::max(ds, 0);
                    const std::int32_t we2 = std::min(we, w);
                    const std::int32_t he2 = std::min(he, h);
                    const std::int32_t de2 = std::min(de, d);
                    float acc = this->pooling_type == 0 ? -FLT_MAX : 0.0F;
                    for (std::int32_t z = ds2; z < de2; ++z) {
                        for (std::int32_t y = hs2; y < he2; ++y) {
                            for (std::int32_t x = ws2; x < we2; ++x) {
                                const float v = image.depth(z).row(y)[x];
                                acc = this->pooling_type == 0 ? std::max(acc, v) : acc + v;
                            }
                        }
                    }
                    if (this->pooling_type == 1) {
                        std::int32_t count = (we2 - ws2) * (he2 - hs2) * (de2 - ds2);
                        if (this->avgpool_count_include_pad != 0) {
                            count = ksize;
                        }
                        acc = count > 0 ? acc / static_cast<float>(count) : 0.0F;
                    } else if (we2 <= ws2 || he2 <= hs2 || de2 <= ds2) {
                        acc = 0.0F;
                    }
                    out.depth(oz).row(oy)[ox] = acc;
                }
            }
        }
    }
    return 0;
}

}  // namespace mlayer

#endif  // MLAYER_POOLING3D_HPP
