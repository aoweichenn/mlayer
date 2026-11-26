// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_RANDOMNORMALLIKE_HPP
#define MLAYER_RANDOMNORMALLIKE_HPP

#include <cstdint>
#include <random>

namespace mlayer {

// ONNX RandomNormalLike. Only the input shape is used. A fixed seed keeps mnet
// inference deterministic for repeated runs.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class RandomNormalLike : public LayerT {
public:
    RandomNormalLike() {
        this->one_blob_only = true;
        this->support_inplace = false;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->mean = pd.get(0, 0.0F);
        this->scale = pd.get(1, 1.0F);
        this->seed = pd.get(2, 0);
        return 0;
    }

    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override {
        const std::size_t elemsize = bottom_blob.elemsize;
        if (bottom_blob.dims == 1) {
            top_blob.create(bottom_blob.w, elemsize, opt.blob_allocator);
        } else if (bottom_blob.dims == 2) {
            top_blob.create(bottom_blob.w, bottom_blob.h, elemsize, opt.blob_allocator);
        } else if (bottom_blob.dims == 3) {
            top_blob.create(bottom_blob.w, bottom_blob.h, bottom_blob.c, elemsize, opt.blob_allocator);
        } else {
            top_blob.create(bottom_blob.w, bottom_blob.h, bottom_blob.d, bottom_blob.c, elemsize,
                            opt.blob_allocator);
        }
        if (top_blob.empty()) {
            return -100;
        }
        std::mt19937 generator(static_cast<std::mt19937::result_type>(this->seed));
        std::normal_distribution<float> distribution(this->mean, this->scale);
        auto* values = static_cast<float*>(top_blob);
        const std::size_t count = top_blob.total();
        for (std::size_t i = 0; i < count; ++i) {
            values[i] = distribution(generator);
        }
        return 0;
    }

public:
    float mean;
    float scale;
    std::int32_t seed;
};

}  // namespace mlayer

#endif  // MLAYER_RANDOMNORMALLIKE_HPP
