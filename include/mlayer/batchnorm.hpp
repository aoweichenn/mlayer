// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_BATCHNORM_HPP
#define MLAYER_BATCHNORM_HPP

#include <cmath>
#include <cstdint>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT>
class BatchNorm : public LayerT {
public:
    BatchNorm();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t load_model(const ModelBinT& mb) override;
    std::int32_t forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const override;

public:
    std::int32_t channels;
    float eps;

    MatT slope_data;
    MatT mean_data;
    MatT var_data;
    MatT bias_data;
    MatT a_data;
    MatT b_data;
};

#define MLAYER_BATCHNORM_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT>

MLAYER_BATCHNORM_TEMPLATE
BatchNorm<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::BatchNorm()
{
    this->one_blob_only = true;
    this->support_inplace = true;
}

MLAYER_BATCHNORM_TEMPLATE
std::int32_t BatchNorm<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::load_param(const ParamDictT& pd)
{
    channels = pd.get(0, 0);
    eps = pd.get(1, 0.f);

    return 0;
}

MLAYER_BATCHNORM_TEMPLATE
std::int32_t BatchNorm<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::load_model(const ModelBinT& mb)
{
    slope_data = mb.load(channels, 1);
    if (slope_data.empty())
        return -100;

    mean_data = mb.load(channels, 1);
    if (mean_data.empty())
        return -100;

    var_data = mb.load(channels, 1);
    if (var_data.empty())
        return -100;

    bias_data = mb.load(channels, 1);
    if (bias_data.empty())
        return -100;

    a_data.create(channels);
    if (a_data.empty())
        return -100;
    b_data.create(channels);
    if (b_data.empty())
        return -100;

    for (std::int32_t i = 0; i < channels; i++)
    {
        float sqrt_var = sqrtf(var_data[i] + eps);
        if (sqrt_var == 0.f)
            sqrt_var = 0.0001f; // sanitize divide by zero
        a_data[i] = bias_data[i] - slope_data[i] * mean_data[i] / sqrt_var;
        b_data[i] = slope_data[i] / sqrt_var;
    }

    return 0;
}

MLAYER_BATCHNORM_TEMPLATE
std::int32_t BatchNorm<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const
{
    // a = bias - slope * mean / sqrt(var)
    // b = slope / sqrt(var)
    // value = b * value + a

    std::int32_t dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        std::int32_t w = bottom_top_blob.w;

        float* ptr = bottom_top_blob;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t i = 0; i < w; i++)
        {
            ptr[i] = b_data[i] * ptr[i] + a_data[i];
        }
    }

    if (dims == 2)
    {
        std::int32_t w = bottom_top_blob.w;
        std::int32_t h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            float a = a_data[i];
            float b = b_data[i];

            for (std::int32_t j = 0; j < w; j++)
            {
                ptr[j] = b * ptr[j] + a;
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        std::int32_t w = bottom_top_blob.w;
        std::int32_t h = bottom_top_blob.h;
        std::int32_t d = bottom_top_blob.d;
        std::int32_t c = bottom_top_blob.c;
        std::int32_t size = w * h * d;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < c; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float a = a_data[q];
            float b = b_data[q];

            for (std::int32_t i = 0; i < size; i++)
            {
                ptr[i] = b * ptr[i] + a;
            }
        }
    }

    return 0;
}


#undef MLAYER_BATCHNORM_TEMPLATE

}  // namespace mlayer

#endif  // MLAYER_BATCHNORM_HPP
