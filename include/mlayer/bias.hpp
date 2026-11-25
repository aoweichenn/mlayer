// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_BIAS_HPP
#define MLAYER_BIAS_HPP

#include <cstdint>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT>
class Bias : public LayerT {
public:
    Bias();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t load_model(const ModelBinT& mb) override;
    std::int32_t forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const override;

public:
    std::int32_t bias_data_size;
    MatT bias_data;
};

#define MLAYER_BIAS_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT>

MLAYER_BIAS_TEMPLATE
Bias<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::Bias()
{
    this->one_blob_only = true;
    this->support_inplace = true;
}

MLAYER_BIAS_TEMPLATE
std::int32_t Bias<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::load_param(const ParamDictT& pd)
{
    bias_data_size = pd.get(0, 0);

    return 0;
}

MLAYER_BIAS_TEMPLATE
std::int32_t Bias<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::load_model(const ModelBinT& mb)
{
    bias_data = mb.load(bias_data_size, 1);
    if (bias_data.empty())
        return -100;

    return 0;
}

MLAYER_BIAS_TEMPLATE
std::int32_t Bias<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const
{
    std::int32_t w = bottom_top_blob.w;
    std::int32_t h = bottom_top_blob.h;
    std::int32_t d = bottom_top_blob.d;
    std::int32_t channels = bottom_top_blob.c;
    std::int32_t size = w * h * d;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        float bias = bias_data[static_cast<std::size_t>(q)];

        for (std::int32_t i = 0; i < size; i++)
        {
            ptr[i] += bias;
        }
    }

    return 0;
}


#undef MLAYER_BIAS_TEMPLATE

}  // namespace mlayer

#endif  // MLAYER_BIAS_HPP
