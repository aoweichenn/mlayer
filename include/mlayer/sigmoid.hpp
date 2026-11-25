// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_SIGMOID_HPP
#define MLAYER_SIGMOID_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT>
class Sigmoid : public LayerT {
public:
    Sigmoid() {
        this->one_blob_only = true;
        this->support_inplace = true;
    }

    std::int32_t forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const override {
        const std::int32_t w = bottom_top_blob.w;
        const std::int32_t h = bottom_top_blob.h;
        const std::int32_t d = bottom_top_blob.d;
        const std::int32_t channels = bottom_top_blob.c;
        const std::int32_t size = w * h * d;
        constexpr float kClamp = 88.3762626647949F;

#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; ++q) {
            auto* ptr = static_cast<float*>(bottom_top_blob.channel(q));
            for (std::int32_t i = 0; i < size; ++i) {
                float value = ptr[i];
                value = std::min(value, kClamp);
                value = std::max(value, -kClamp);
                ptr[i] = 1.0F / (1.0F + std::exp(-value));
            }
        }

        return 0;
    }

    std::int32_t get_fused_activation(std::int32_t& type, MatT& params) const override {
        type = 4;
        params = MatT();
        return 0;
    }
};

}  // namespace mlayer

#endif  // MLAYER_SIGMOID_HPP
