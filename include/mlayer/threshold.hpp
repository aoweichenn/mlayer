// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_THRESHOLD_HPP
#define MLAYER_THRESHOLD_HPP

#include <cstdint>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class Threshold : public LayerT {
public:
    Threshold() {
        this->one_blob_only = true;
        this->support_inplace = true;
        this->threshold = 0.0F;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->threshold = pd.get(0, 0.0F);
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
                ptr[i] = ptr[i] > this->threshold ? 1.0F : 0.0F;
            }
        }

        return 0;
    }

public:
    float threshold;
};

}  // namespace mlayer

#endif  // MLAYER_THRESHOLD_HPP
