// Copyright 2021 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_POOLING3D_HPP
#define MLAYER_POOLING3D_HPP

#include <algorithm>
#include <limits>
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

    constexpr std::int32_t kAdaptiveKeep = -233;
    if (this->adaptive_pooling != 0) {
        const std::int32_t outw = this->out_w == kAdaptiveKeep ? w : this->out_w;
        const std::int32_t outh = this->out_h == kAdaptiveKeep ? h : this->out_h;
        const std::int32_t outd = this->out_d == kAdaptiveKeep ? d : this->out_d;
        if (outw == w && outh == h && outd == d) {
            top_blob = bottom_blob;
            return 0;
        }
        top_blob.create(outw, outh, outd, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }
#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; ++q) {
            const float* inptr = bottom_blob.channel(q);
            MatT out = top_blob.channel(q);
            for (std::int32_t oz = 0; oz < outd; ++oz) {
                const std::int32_t id0 = d * oz / outd;
                const std::int32_t id1 = (d * (oz + 1) + outd - 1) / outd;
                const std::int32_t dk = id1 - id0;
                for (std::int32_t oy = 0; oy < outh; ++oy) {
                    const std::int32_t ih0 = h * oy / outh;
                    const std::int32_t ih1 = (h * (oy + 1) + outh - 1) / outh;
                    const std::int32_t hk = ih1 - ih0;
                    for (std::int32_t ox = 0; ox < outw; ++ox) {
                        const std::int32_t iw0 = w * ox / outw;
                        const std::int32_t iw1 = (w * (ox + 1) + outw - 1) / outw;
                        const std::int32_t wk = iw1 - iw0;
                        float acc = this->pooling_type == 0
                                        ? inptr[id0 * w * h + ih0 * w + iw0]
                                        : 0.0F;
                        for (std::int32_t id = id0; id < id1; ++id) {
                            for (std::int32_t ih = ih0; ih < ih1; ++ih) {
                                for (std::int32_t iw = iw0; iw < iw1; ++iw) {
                                    const float v = inptr[id * w * h + ih * w + iw];
                                    acc = this->pooling_type == 0 ? std::max(acc, v) : acc + v;
                                }
                            }
                        }
                        if (this->pooling_type == 1) {
                            acc = acc / static_cast<float>(hk) / static_cast<float>(wk) /
                                  static_cast<float>(dk);
                        }
                        out.depth(oz).row(oy)[ox] = acc;
                    }
                }
            }
        }
        return 0;
    }

    std::int32_t padtail_w = 0;
    std::int32_t padtail_h = 0;
    std::int32_t padtail_d = 0;
    std::int32_t padded_w = 0;
    std::int32_t padded_h = 0;
    std::int32_t padded_d = 0;
    std::int32_t left_eff = 0;
    std::int32_t top_eff = 0;
    std::int32_t front_eff = 0;
    if (this->pad_mode == 0) {
        if (this->kernel_w <= 0 || this->stride_w <= 0) {
            return -1;
        }
        const std::int32_t wtail = (w + this->pad_left + this->pad_right - this->kernel_w) % this->stride_w;
        const std::int32_t htail = (h + this->pad_top + this->pad_bottom - this->kernel_h) % this->stride_h;
        const std::int32_t dtail = (d + this->pad_front + this->pad_behind - this->kernel_d) % this->stride_d;
        padtail_w = wtail != 0 ? this->stride_w - wtail : 0;
        padtail_h = htail != 0 ? this->stride_h - htail : 0;
        padtail_d = dtail != 0 ? this->stride_d - dtail : 0;
        padded_w = w + this->pad_left + this->pad_right + padtail_w;
        padded_h = h + this->pad_top + this->pad_bottom + padtail_h;
        padded_d = d + this->pad_front + this->pad_behind + padtail_d;
        left_eff = this->pad_left;
        top_eff = this->pad_top;
        front_eff = this->pad_front;
    } else if (this->pad_mode == 1) {
        padded_w = w + this->pad_left + this->pad_right;
        padded_h = h + this->pad_top + this->pad_bottom;
        padded_d = d + this->pad_front + this->pad_behind;
        left_eff = this->pad_left;
        top_eff = this->pad_top;
        front_eff = this->pad_front;
    } else if (this->pad_mode == 2 || this->pad_mode == 3) {
        std::int32_t wpad = this->kernel_w + (w - 1) / this->stride_w * this->stride_w - w;
        std::int32_t hpad = this->kernel_h + (h - 1) / this->stride_h * this->stride_h - h;
        std::int32_t dpad = this->kernel_d + (d - 1) / this->stride_d * this->stride_d - d;
        wpad = std::max(wpad, 0);
        hpad = std::max(hpad, 0);
        dpad = std::max(dpad, 0);
        padded_w = w + wpad;
        padded_h = h + hpad;
        padded_d = d + dpad;
        left_eff = this->pad_mode == 2 ? wpad / 2 : wpad - wpad / 2;
        top_eff = this->pad_mode == 2 ? hpad / 2 : hpad - hpad / 2;
        front_eff = this->pad_mode == 2 ? dpad / 2 : dpad - dpad / 2;
    } else {
        return -1;
    }
    if (padded_w < this->kernel_w || padded_h < this->kernel_h || padded_d < this->kernel_d) {
        return -1;
    }

    const std::int32_t outw = (padded_w - this->kernel_w) / this->stride_w + 1;
    const std::int32_t outh = (padded_h - this->kernel_h) / this->stride_h + 1;
    const std::int32_t outd = (padded_d - this->kernel_d) / this->stride_d + 1;
    if (outw <= 0 || outh <= 0 || outd <= 0) {
        return -1;
    }

    top_blob.create(outw, outh, outd, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty()) {
        return -100;
    }

    const float pad_value =
        this->pooling_type == 0 ? -std::numeric_limits<float>::max() : 0.0F;
    const std::int32_t maxk = this->kernel_w * this->kernel_h * this->kernel_d;
    const std::int32_t count_x0 = this->pad_left;
    const std::int32_t count_y0 = this->pad_top;
    const std::int32_t count_z0 = this->pad_front;
    const std::int32_t count_x1 = padded_w - this->pad_right - padtail_w;
    const std::int32_t count_y1 = padded_h - this->pad_bottom - padtail_h;
    const std::int32_t count_z1 = padded_d - this->pad_behind - padtail_d;

#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t q = 0; q < channels; ++q) {
        const MatT image = bottom_blob.channel(q);
        MatT out = top_blob.channel(q);
        for (std::int32_t oz = 0; oz < outd; ++oz) {
            const std::int32_t z0 = oz * this->stride_d;
            for (std::int32_t oy = 0; oy < outh; ++oy) {
                const std::int32_t y0 = oy * this->stride_h;
                for (std::int32_t ox = 0; ox < outw; ++ox) {
                    const std::int32_t x0 = ox * this->stride_w;
                    float acc = this->pooling_type == 0 ? -std::numeric_limits<float>::max() : 0.0F;
                    std::int32_t area = 0;
                    for (std::int32_t kz = 0; kz < this->kernel_d; ++kz) {
                        const std::int32_t spz = z0 + kz;
                        const std::int32_t sz = spz - front_eff;
                        const bool z_ok = sz >= 0 && sz < d;
                        for (std::int32_t ky = 0; ky < this->kernel_h; ++ky) {
                            const std::int32_t spy = y0 + ky;
                            const std::int32_t sy = spy - top_eff;
                            const bool y_ok = sy >= 0 && sy < h;
                            for (std::int32_t kx = 0; kx < this->kernel_w; ++kx) {
                                const std::int32_t spx = x0 + kx;
                                const std::int32_t sx = spx - left_eff;
                                const bool in_range = z_ok && y_ok && sx >= 0 && sx < w;
                                const float v = in_range ? image.depth(sz).row(sy)[sx] : pad_value;
                                if (this->pooling_type == 0) {
                                    acc = std::max(acc, v);
                                } else if (this->avgpool_count_include_pad != 0) {
                                    acc += v;
                                } else if (spz >= count_z0 && spz < count_z1 && spy >= count_y0 &&
                                           spy < count_y1 && spx >= count_x0 && spx < count_x1) {
                                    acc += v;
                                    ++area;
                                }
                            }
                        }
                    }
                    if (this->pooling_type == 1) {
                        acc = this->avgpool_count_include_pad != 0
                                  ? acc / static_cast<float>(maxk)
                                  : acc / static_cast<float>(area);
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
