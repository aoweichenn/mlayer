// Copyright 2021 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_POOLING1D_HPP
#define MLAYER_POOLING1D_HPP

#include <algorithm>
#include <limits>
#include <cmath>
#include <cstdint>

namespace mlayer {

// 1D pooling over the width of a (w, channels) blob.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate = false>
class Pooling1D : public LayerT {
public:
    Pooling1D();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

public:
    std::int32_t pooling_type;  // 0 = max, 1 = average
    std::int32_t kernel_w;
    std::int32_t stride_w;
    std::int32_t pad_left;
    std::int32_t pad_right;
    std::int32_t global_pooling;
    std::int32_t pad_mode;
    std::int32_t avgpool_count_include_pad;
    std::int32_t adaptive_pooling;
    std::int32_t out_w;
};

#define MLAYER_POOLING1D_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate>

MLAYER_POOLING1D_TEMPLATE
Pooling1D<LayerT, MatT, OptionT, ParamDictT, Validate>::Pooling1D() {
    this->one_blob_only = true;
    this->support_inplace = false;
    this->pooling_type = 0;
    this->kernel_w = 0;
    this->stride_w = 1;
    this->pad_left = 0;
    this->pad_right = 0;
    this->global_pooling = 0;
    this->pad_mode = 0;
    this->avgpool_count_include_pad = 0;
    this->adaptive_pooling = 0;
    this->out_w = 0;
}

MLAYER_POOLING1D_TEMPLATE
std::int32_t Pooling1D<LayerT, MatT, OptionT, ParamDictT, Validate>::load_param(const ParamDictT& pd) {
    this->pooling_type = pd.get(0, 0);
    this->kernel_w = pd.get(1, 0);
    this->stride_w = pd.get(2, 1);
    this->pad_left = pd.get(3, 0);
    this->pad_right = pd.get(14, this->pad_left);
    this->global_pooling = pd.get(4, 0);
    this->pad_mode = pd.get(5, 0);
    this->avgpool_count_include_pad = pd.get(6, 0);
    this->adaptive_pooling = pd.get(7, 0);
    this->out_w = pd.get(8, 0);
    return 0;
}

MLAYER_POOLING1D_TEMPLATE
std::int32_t Pooling1D<LayerT, MatT, OptionT, ParamDictT, Validate>::forward(const MatT& bottom_blob,
                                                                           MatT& top_blob,
                                                                           const OptionT& opt) const {
    const std::int32_t w = bottom_blob.w;
    const std::int32_t channels = bottom_blob.h;
    const std::size_t elemsize = bottom_blob.elemsize;

    if (this->global_pooling != 0) {
        top_blob.create(channels, elemsize, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }
#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; ++q) {
            const float* ptr = bottom_blob.row(q);
            if (this->pooling_type == 0) {
                float m = ptr[0];
                for (std::int32_t i = 1; i < w; ++i) {
                    m = std::max(m, ptr[i]);
                }
                top_blob[q] = m;
            } else {
                float s = 0.0F;
                for (std::int32_t i = 0; i < w; ++i) {
                    s += ptr[i];
                }
                top_blob[q] = s / static_cast<float>(w);
            }
        }
        return 0;
    }

    if (this->adaptive_pooling != 0) {
        const std::int32_t outw = this->out_w;
        if (outw <= 0) {
            return -1;
        }
        top_blob.create(outw, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }
#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; ++q) {
            const float* inptr = bottom_blob.row(q);
            float* outptr = top_blob.row(q);
            for (std::int32_t j = 0; j < outw; ++j) {
                const std::int32_t iw0 = w * j / outw;
                const std::int32_t iw1 = (w * (j + 1) + outw - 1) / outw;
                const std::int32_t wk = iw1 - iw0;
                float acc = this->pooling_type == 0 ? inptr[iw0] : 0.0F;
                for (std::int32_t iw = iw0; iw < iw1; ++iw) {
                    acc = this->pooling_type == 0 ? std::max(acc, inptr[iw]) : acc + inptr[iw];
                }
                outptr[j] = this->pooling_type == 0 ? acc : acc / static_cast<float>(wk);
            }
        }
        return 0;
    }

    if (this->kernel_w <= 0 || this->stride_w <= 0) {
        return -1;
    }
    std::int32_t padtail_w = 0;
    std::int32_t padded_w = 0;
    std::int32_t left_eff = 0;
    if (this->pad_mode == 0) {
        const std::int32_t wtail = (w + this->pad_left + this->pad_right - this->kernel_w) % this->stride_w;
        padtail_w = wtail != 0 ? this->stride_w - wtail : 0;
        padded_w = w + this->pad_left + this->pad_right + padtail_w;
        left_eff = this->pad_left;
    } else if (this->pad_mode == 1) {
        padded_w = w + this->pad_left + this->pad_right;
        left_eff = this->pad_left;
    } else if (this->pad_mode == 2 || this->pad_mode == 3) {
        std::int32_t wpad = this->kernel_w + (w - 1) / this->stride_w * this->stride_w - w;
        wpad = std::max(wpad, 0);
        padded_w = w + wpad;
        left_eff = this->pad_mode == 2 ? wpad / 2 : wpad - wpad / 2;
    } else {
        return -1;
    }
    if (padded_w < this->kernel_w) {
        return -1;
    }
    const std::int32_t outw = (padded_w - this->kernel_w) / this->stride_w + 1;
    if (outw <= 0) {
        return -1;
    }

    top_blob.create(outw, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty()) {
        return -100;
    }

    const float pad_value = this->pooling_type == 0 ? -std::numeric_limits<float>::max() : 0.0F;
    const std::int32_t count_x0 = this->pad_left;
    const std::int32_t count_x1 = padded_w - this->pad_right - padtail_w;

#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t q = 0; q < channels; ++q) {
        const float* inptr = bottom_blob.row(q);
        float* outptr = top_blob.row(q);
        for (std::int32_t j = 0; j < outw; ++j) {
            const std::int32_t x0 = j * this->stride_w;
            float acc = this->pooling_type == 0 ? -std::numeric_limits<float>::max() : 0.0F;
            std::int32_t area = 0;
            for (std::int32_t k = 0; k < this->kernel_w; ++k) {
                const std::int32_t sp = x0 + k;
                const std::int32_t sx = sp - left_eff;
                const float v = (sx >= 0 && sx < w) ? inptr[sx] : pad_value;
                if (this->pooling_type == 0) {
                    acc = std::max(acc, v);
                } else if (this->avgpool_count_include_pad != 0) {
                    acc += v;
                } else if (sp >= count_x0 && sp < count_x1) {
                    acc += v;
                    ++area;
                }
            }
            if (this->pooling_type == 1) {
                acc = this->avgpool_count_include_pad != 0 ? acc / static_cast<float>(this->kernel_w)
                                                           : acc / static_cast<float>(area);
            }
            outptr[j] = acc;
        }
    }
    return 0;
}

}  // namespace mlayer

#endif  // MLAYER_POOLING1D_HPP
