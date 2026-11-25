// Copyright 2018 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_CLIP_HPP
#define MLAYER_CLIP_HPP

#include <cfloat>
#include <cstdint>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class Clip : public LayerT {
public:
    Clip() {
        this->one_blob_only = true;
        this->support_inplace = true;
        this->min = -FLT_MAX;
        this->max = FLT_MAX;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->min = pd.get(0, -FLT_MAX);
        this->max = pd.get(1, FLT_MAX);
        return 0;
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
                if (ptr[i] < this->min) {
                    ptr[i] = this->min;
                }
                if (ptr[i] > this->max) {
                    ptr[i] = this->max;
                }
            }
        }

        return 0;
    }

public:
    float min;
    float max;

    std::int32_t get_fused_activation(std::int32_t& type, MatT& params) const override {
        type = 3;
        params = MatT(2);
        params[0] = min;
        params[1] = max;
        return 0;
    }
};

}  // namespace mlayer

#endif  // MLAYER_CLIP_HPP
