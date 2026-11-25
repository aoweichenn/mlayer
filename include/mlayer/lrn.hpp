// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_LRN_HPP
#define MLAYER_LRN_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace mlayer {

// Local response normalisation, matching ncnn's LRN semantics:
//   region_type 0: sum the squares across the local channel window
//   region_type 1: sum the squares across a local_size x local_size window
// The window is zero padded, so clamped coordinates simply contribute 0.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class LRN : public LayerT {
public:
    LRN();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const override;

public:
    // 0 = across channels, 1 = within each channel.
    std::int32_t region_type;
    std::int32_t local_size;
    float alpha;
    float beta;
    float bias;
};

#define MLAYER_LRN_TEMPLATE template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>

MLAYER_LRN_TEMPLATE
LRN<LayerT, MatT, OptionT, ParamDictT>::LRN() {
    this->one_blob_only = true;
    this->support_inplace = true;
    this->region_type = 0;
    this->local_size = 5;
    this->alpha = 1.0F;
    this->beta = 0.75F;
    this->bias = 1.0F;
}

MLAYER_LRN_TEMPLATE
std::int32_t LRN<LayerT, MatT, OptionT, ParamDictT>::load_param(const ParamDictT& pd) {
    this->region_type = pd.get(0, 0);
    this->local_size = pd.get(1, 5);
    this->alpha = pd.get(2, 1.0F);
    this->beta = pd.get(3, 0.75F);
    this->bias = pd.get(4, 1.0F);
    return 0;
}

MLAYER_LRN_TEMPLATE
std::int32_t LRN<LayerT, MatT, OptionT, ParamDictT>::forward_inplace(MatT& bottom_top_blob,
                                                                    const OptionT& opt) const {
    const std::int32_t w = bottom_top_blob.w;
    const std::int32_t h = bottom_top_blob.h;
    const std::int32_t channels = bottom_top_blob.c;
    const std::size_t elemsize = bottom_top_blob.elemsize;
    const std::int32_t size = w * h;

    MatT square_blob;
    square_blob.create(w, h, channels, elemsize, opt.workspace_allocator);
    if (square_blob.empty()) {
        return -100;
    }

#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t q = 0; q < channels; ++q) {
        const auto* ptr = static_cast<const float*>(bottom_top_blob.channel(q));
        auto* outptr = static_cast<float*>(square_blob.channel(q));
        for (std::int32_t i = 0; i < size; ++i) {
            outptr[i] = ptr[i] * ptr[i];
        }
    }

    if (this->region_type == 0) {
        const float alpha_div_size = this->alpha / static_cast<float>(this->local_size);

#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; ++q) {
            auto* ptr = static_cast<float*>(bottom_top_blob.channel(q));
            const std::int32_t begin = q - this->local_size / 2;
            const std::int32_t end = q + (this->local_size - 1) / 2;
            for (std::int32_t i = 0; i < size; ++i) {
                float sum = 0.0F;
                for (std::int32_t p = begin; p <= end; ++p) {
                    if (p < 0 || p >= channels) {
                        continue;
                    }
                    sum += static_cast<const float*>(square_blob.channel(p))[i];
                }
                ptr[i] *= std::pow(this->bias + alpha_div_size * sum, -this->beta);
            }
        }
        return 0;
    }

    const std::int32_t pad = this->local_size / 2;
    const float alpha_div_size = this->alpha / static_cast<float>(this->local_size * this->local_size);

#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t q = 0; q < channels; ++q) {
        auto* ptr = static_cast<float*>(bottom_top_blob.channel(q));
        const auto* sqptr = static_cast<const float*>(square_blob.channel(q));
        for (std::int32_t y = 0; y < h; ++y) {
            for (std::int32_t x = 0; x < w; ++x) {
                float sum = 0.0F;
                for (std::int32_t ky = -pad; ky < this->local_size - pad; ++ky) {
                    const std::int32_t sy = y + ky;
                    if (sy < 0 || sy >= h) {
                        continue;
                    }
                    for (std::int32_t kx = -pad; kx < this->local_size - pad; ++kx) {
                        const std::int32_t sx = x + kx;
                        if (sx < 0 || sx >= w) {
                            continue;
                        }
                        sum += sqptr[sy * w + sx];
                    }
                }
                const std::int32_t index = y * w + x;
                ptr[index] *= std::pow(this->bias + alpha_div_size * sum, -this->beta);
            }
        }
    }
    return 0;
}

}  // namespace mlayer

#endif  // MLAYER_LRN_HPP
