// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_LOG_HPP
#define MLAYER_LOG_HPP

#include <cmath>
#include <cstdint>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class Log : public LayerT {
public:
    Log() {
        this->one_blob_only = true;
        this->support_inplace = true;
        this->base = -1.0F;
        this->scale = 1.0F;
        this->shift = 0.0F;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->base = pd.get(0, -1.0F);
        this->scale = pd.get(1, 1.0F);
        this->shift = pd.get(2, 0.0F);
        return 0;
    }

    std::int32_t forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const override {
        const std::int32_t w = bottom_top_blob.w;
        const std::int32_t h = bottom_top_blob.h;
        const std::int32_t d = bottom_top_blob.d;
        const std::int32_t channels = bottom_top_blob.c;
        const std::int32_t size = w * h * d;
        const float log_base_inv = this->base == -1.0F ? 1.0F : 1.0F / std::log(this->base);

#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; ++q) {
            auto* ptr = static_cast<float*>(bottom_top_blob.channel(q));
            for (std::int32_t i = 0; i < size; ++i) {
                ptr[i] = std::log(this->shift + ptr[i] * this->scale) * log_base_inv;
            }
        }

        return 0;
    }

public:
    float base;
    float scale;
    float shift;
};

}  // namespace mlayer

#endif  // MLAYER_LOG_HPP
