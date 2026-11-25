// Copyright 2021 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_INSTANCENORM_HPP
#define MLAYER_INSTANCENORM_HPP

#include <cmath>
#include <cstdint>

namespace mlayer {

// Per-channel instance normalization with optional affine weights.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT>
class InstanceNorm : public LayerT {
public:
    InstanceNorm();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t load_model(const ModelBinT& mb) override;
    std::int32_t forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const override;

public:
    std::int32_t channels;
    float eps;
    std::int32_t affine;
    MatT gamma_data;
    MatT beta_data;
};

#define MLAYER_INSTANCENORM_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT>

MLAYER_INSTANCENORM_TEMPLATE
InstanceNorm<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::InstanceNorm() {
    this->one_blob_only = true;
    this->support_inplace = true;
    this->channels = 0;
    this->eps = 0.001F;
    this->affine = 1;
}

MLAYER_INSTANCENORM_TEMPLATE
std::int32_t InstanceNorm<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::load_param(const ParamDictT& pd) {
    this->channels = pd.get(0, 0);
    this->eps = pd.get(1, 0.001F);
    this->affine = pd.get(2, 1);
    return 0;
}

MLAYER_INSTANCENORM_TEMPLATE
std::int32_t InstanceNorm<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::load_model(const ModelBinT& mb) {
    if (this->affine != 0) {
        this->gamma_data = mb.load(this->channels, 1);
        this->beta_data = mb.load(this->channels, 1);
        if (this->gamma_data.empty() || this->beta_data.empty()) {
            return -100;
        }
    }
    return 0;
}

MLAYER_INSTANCENORM_TEMPLATE
std::int32_t InstanceNorm<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::forward_inplace(
    MatT& bottom_top_blob, const OptionT& opt) const {
    const std::int32_t w = bottom_top_blob.w;
    const std::int32_t h = bottom_top_blob.h;
    const std::int32_t d = bottom_top_blob.d;
    const std::int32_t c = bottom_top_blob.c;
    const std::int32_t size = w * h * d;

#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t q = 0; q < c; ++q) {
        float* ptr = bottom_top_blob.channel(q);
        float sum = 0.0F;
        for (std::int32_t i = 0; i < size; ++i) {
            sum += ptr[i];
        }
        const float mean = sum / static_cast<float>(size);
        float sqsum = 0.0F;
        for (std::int32_t i = 0; i < size; ++i) {
            const float t = ptr[i] - mean;
            sqsum += t * t;
        }
        const float var = sqsum / static_cast<float>(size);
        const float a = 1.0F / std::sqrt(var + this->eps);
        float gamma = 1.0F;
        float beta = 0.0F;
        if (this->affine != 0) {
            gamma = this->gamma_data[q];
            beta = this->beta_data[q];
        }
        for (std::int32_t i = 0; i < size; ++i) {
            ptr[i] = (ptr[i] - mean) * a * gamma + beta;
        }
    }
    return 0;
}

}  // namespace mlayer

#endif  // MLAYER_INSTANCENORM_HPP
