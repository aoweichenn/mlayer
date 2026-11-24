// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_DROPOUT_HPP
#define MLAYER_DROPOUT_HPP

#include <cstdint>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class Dropout : public LayerT {
public:
    Dropout();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const override;

public:
    float scale;
};

#define MLAYER_DROPOUT_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>

MLAYER_DROPOUT_TEMPLATE
Dropout<LayerT, MatT, OptionT, ParamDictT>::Dropout()
{
    this->one_blob_only = true;
    this->support_inplace = true;
}

MLAYER_DROPOUT_TEMPLATE
std::int32_t Dropout<LayerT, MatT, OptionT, ParamDictT>::load_param(const ParamDictT& pd)
{
    this->scale = pd.get(0, 1.0F);

    return 0;
}

MLAYER_DROPOUT_TEMPLATE
std::int32_t Dropout<LayerT, MatT, OptionT, ParamDictT>::forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const
{
    if (this->scale == 1.0F)
    {
        return 0;
    }

    std::int32_t w = bottom_top_blob.w;
    std::int32_t h = bottom_top_blob.h;
    std::int32_t d = bottom_top_blob.d;
    std::int32_t channels = bottom_top_blob.c;
    std::int32_t size = w * h * d;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        for (std::int32_t i = 0; i < size; i++)
        {
            ptr[i] = ptr[i] * this->scale;
        }
    }

    return 0;
}


#undef MLAYER_DROPOUT_TEMPLATE

}  // namespace mlayer

#endif  // MLAYER_DROPOUT_HPP
