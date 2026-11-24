// Copyright 2020 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_GROUPNORM_HPP
#define MLAYER_GROUPNORM_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT, bool Validate>
class GroupNorm : public LayerT {
public:
    GroupNorm();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t load_model(const ModelBinT& mb) override;
    std::int32_t forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const override;

public:
    std::int32_t group;
    std::int32_t channels;
    float eps;
    std::int32_t affine;
    MatT gamma_data;
    MatT beta_data;
};

#define MLAYER_GROUPNORM_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT, bool Validate>

MLAYER_GROUPNORM_TEMPLATE
GroupNorm<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate>::GroupNorm()
{
    this->one_blob_only = true;
    this->support_inplace = true;
}

MLAYER_GROUPNORM_TEMPLATE
std::int32_t GroupNorm<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate>::load_param(const ParamDictT& pd)
{
    group = pd.get(0, 1);
    channels = pd.get(1, 0);
    eps = pd.get(2, 0.001f);
    affine = pd.get(3, 1);

if constexpr (Validate) {
    if (group <= 0)
        return -1;

    if (channels < 0 || (channels > 0 && channels % group != 0) || (affine && channels == 0))
        return -1;
}

    return 0;
}

MLAYER_GROUPNORM_TEMPLATE
std::int32_t GroupNorm<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate>::load_model(const ModelBinT& mb)
{
    if (affine == 0)
        return 0;

    gamma_data = mb.load(channels, 1);
    if (gamma_data.empty())
        return -100;

    beta_data = mb.load(channels, 1);
    if (beta_data.empty())
        return -100;

    return 0;
}

inline void groupnorm(float* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, std::int32_t channels, std::int32_t size, std::size_t cstep)
{
    float sum = 0.f;
    for (std::int32_t q = 0; q < channels; q++)
    {
        const float* ptr0 = ptr + cstep * q;
        for (std::int32_t i = 0; i < size; i++)
        {
            sum += ptr0[i];
        }
    }

    float mean = sum / (channels * size);

    float sqsum = 0.f;
    for (std::int32_t q = 0; q < channels; q++)
    {
        const float* ptr0 = ptr + cstep * q;
        for (std::int32_t i = 0; i < size; i++)
        {
            float v = ptr0[i] - mean;
            sqsum += v * v;
        }
    }

    float var = sqsum / (channels * size);

    float a = 1.f / std::sqrt(var + eps);
    float b = -mean * a;

    if (gamma_ptr && beta_ptr)
    {
        for (std::int32_t q = 0; q < channels; q++)
        {
            float* ptr0 = ptr + cstep * q;
            const float gamma = gamma_ptr[q];
            const float beta = beta_ptr[q];
            for (std::int32_t i = 0; i < size; i++)
            {
                ptr0[i] = (ptr0[i] * a + b) * gamma + beta;
            }
        }
    }
    else
    {
        for (std::int32_t q = 0; q < channels; q++)
        {
            float* ptr0 = ptr + cstep * q;
            for (std::int32_t i = 0; i < size; i++)
            {
                ptr0[i] = ptr0[i] * a + b;
            }
        }
    }
}

MLAYER_GROUPNORM_TEMPLATE
std::int32_t GroupNorm<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate>::forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const
{
    const std::int32_t dims = bottom_top_blob.dims;
    const std::int32_t channels_g = channels / group;

    if (dims == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t g = 0; g < group; g++)
        {
            MatT bottom_top_blob_g = bottom_top_blob.range(g * channels_g, channels_g);
            const float* gamma_ptr = affine ? (const float*)gamma_data + g * channels_g : 0;
            const float* beta_ptr = affine ? (const float*)beta_data + g * channels_g : 0;
            groupnorm(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g, 1, 1);
        }
    }

    if (dims == 2)
    {
        const std::int32_t w = bottom_top_blob.w;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t g = 0; g < group; g++)
        {
            MatT bottom_top_blob_g = bottom_top_blob.row_range(g * channels_g, channels_g);
            const float* gamma_ptr = affine ? (const float*)gamma_data + g * channels_g : 0;
            const float* beta_ptr = affine ? (const float*)beta_data + g * channels_g : 0;
            groupnorm(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g, w, w);
        }
    }

    if (dims == 3 || dims == 4)
    {
        const std::int32_t size = bottom_top_blob.w * bottom_top_blob.h * bottom_top_blob.d;
        const std::size_t cstep = bottom_top_blob.cstep;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t g = 0; g < group; g++)
        {
            MatT bottom_top_blob_g = bottom_top_blob.channel_range(g * channels_g, channels_g);
            const float* gamma_ptr = affine ? (const float*)gamma_data + g * channels_g : 0;
            const float* beta_ptr = affine ? (const float*)beta_data + g * channels_g : 0;
            groupnorm(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g, size, cstep);
        }
    }

    return 0;
}


#undef MLAYER_GROUPNORM_TEMPLATE

}  // namespace mlayer

#endif  // MLAYER_GROUPNORM_HPP
