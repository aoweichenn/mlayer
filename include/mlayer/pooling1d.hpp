// Copyright 2021 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_POOLING1D_HPP
#define MLAYER_POOLING1D_HPP

#include <algorithm>
#include <cfloat>
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

    std::int32_t outw = 0;
    if (this->adaptive_pooling != 0) {
        outw = this->out_w;
    } else if (this->pad_mode == 0) {
        if (this->kernel_w <= 0) {
            return -1;
        }
        outw = (w + this->pad_left + this->pad_right - this->kernel_w) / this->stride_w + 1;
    } else if (this->pad_mode == 1) {
        if (this->kernel_w <= 0) {
            return -1;
        }
        outw = (w - this->kernel_w) / this->stride_w + 1;
    } else {
        // SAME modes are not needed by the supported models.
        return -1;
    }
    if (outw <= 0) {
        return -1;
    }

    top_blob.create(outw, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty()) {
        return -100;
    }

#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t q = 0; q < channels; ++q) {
        const float* ptr = bottom_blob.row(q);
        float* outptr = top_blob.row(q);
        for (std::int32_t i = 0; i < outw; ++i) {
            std::int32_t wstart = 0;
            std::int32_t wend = 0;
            std::int32_t ksize = 0;
            if (this->adaptive_pooling != 0) {
                const std::int32_t start = static_cast<std::int32_t>(std::floor(static_cast<float>(i) * w / outw));
                const std::int32_t end =
                    static_cast<std::int32_t>(std::ceil(static_cast<float>(i + 1) * w / outw));
                wstart = start;
                wend = end;
                ksize = end - start;
            } else {
                wstart = i * this->stride_w - this->pad_left;
                wend = wstart + this->kernel_w;
                ksize = this->kernel_w;
            }
            std::int32_t sstart = std::max(wstart, 0);
            std::int32_t send = std::min(wend, w);
            if (this->adaptive_pooling != 0) {
                sstart = wstart;
                send = wend;
            }
            float acc = this->pooling_type == 0 ? -FLT_MAX : 0.0F;
            for (std::int32_t x = sstart; x < send; ++x) {
                acc = this->pooling_type == 0 ? std::max(acc, ptr[x]) : acc + ptr[x];
            }
            if (this->pooling_type == 1) {
                const std::int32_t count = this->avgpool_count_include_pad != 0 ? ksize : (send - sstart);
                acc = count > 0 ? acc / static_cast<float>(count) : 0.0F;
            }
            outptr[i] = this->pooling_type == 0 && (send <= sstart) ? 0.0F : acc;
        }
    }
    return 0;
}

}  // namespace mlayer

#endif  // MLAYER_POOLING1D_HPP
