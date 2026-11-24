// Copyright 2020 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_SWISH_HPP
#define MLAYER_SWISH_HPP

#include <cmath>
#include <cstdint>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT>
class Swish : public LayerT {
public:
    Swish() {
        this->one_blob_only = true;
        this->support_inplace = true;
    }

    std::int32_t forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const override {
        const std::int32_t w = bottom_top_blob.w;
        const std::int32_t h = bottom_top_blob.h;
        const std::int32_t d = bottom_top_blob.d;
        const std::int32_t channels = bottom_top_blob.c;
        const std::int32_t size = w * h * d;

#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; ++q) {
            auto* ptr = static_cast<float*>(bottom_top_blob.channel(q));
            for (std::int32_t i = 0; i < size; ++i) {
                const float x = ptr[i];
                ptr[i] = x / (1.0F + std::exp(-x));
            }
        }

        return 0;
    }
};

}  // namespace mlayer

#endif  // MLAYER_SWISH_HPP
