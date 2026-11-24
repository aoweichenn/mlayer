// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_PRELU_HPP
#define MLAYER_PRELU_HPP

#include <cstdint>


namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT>
class PReLU : public LayerT {
public:
    PReLU();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t load_model(const ModelBinT& mb) override;
    std::int32_t forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const override;

public:
    std::int32_t num_slope;
    MatT slope_data;
};

#define MLAYER_PRELU_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT>

MLAYER_PRELU_TEMPLATE
PReLU<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::PReLU()
{
    this->one_blob_only = true;
    this->support_inplace = true;
}

MLAYER_PRELU_TEMPLATE
std::int32_t PReLU<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::load_param(const ParamDictT& pd)
{
    num_slope = pd.get(0, 0);

    return 0;
}

MLAYER_PRELU_TEMPLATE
std::int32_t PReLU<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::load_model(const ModelBinT& mb)
{
    slope_data = mb.load(num_slope, 1);
    if (slope_data.empty())
        return -100;

    return 0;
}

MLAYER_PRELU_TEMPLATE
std::int32_t PReLU<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const
{
    std::int32_t dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        std::int32_t w = bottom_top_blob.w;

        float* ptr = bottom_top_blob;

        if (num_slope > 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t i = 0; i < w; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope_data[i];
            }
        }
        else
        {
            float slope = slope_data[0];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t i = 0; i < w; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope;
            }
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
            float slope = num_slope > 1 ? slope_data[i] : slope_data[0];

            for (std::int32_t j = 0; j < w; j++)
            {
                if (ptr[j] < 0)
                    ptr[j] *= slope;
            }
        }
    }

    if (dims == 3 || dims == 4)
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
            float slope = num_slope > 1 ? slope_data[q] : slope_data[0];

            for (std::int32_t i = 0; i < size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope;
            }
        }
    }

    return 0;
}


#undef MLAYER_PRELU_TEMPLATE

}  // namespace mlayer

#endif  // MLAYER_PRELU_HPP
