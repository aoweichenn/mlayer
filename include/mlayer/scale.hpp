// Copyright 2021 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_SCALE_HPP
#define MLAYER_SCALE_HPP

#include <cstdint>
#include <vector>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT>
class Scale : public LayerT {
public:
    Scale();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t load_model(const ModelBinT& mb) override;
    std::int32_t forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const override;
    std::int32_t forward_inplace(std::vector<MatT>& bottom_top_blobs, const OptionT& opt) const override;

public:
    std::int32_t scale_data_size;
    std::int32_t bias_term;
    MatT scale_data;
    MatT bias_data;
};

#define MLAYER_SCALE_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT>

MLAYER_SCALE_TEMPLATE
Scale<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::Scale()
{
    this->one_blob_only = true;
    this->support_inplace = true;
}

MLAYER_SCALE_TEMPLATE
std::int32_t Scale<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::load_param(const ParamDictT& pd)
{
    scale_data_size = pd.get(0, 0);
    bias_term = pd.get(1, 0);

    if (scale_data_size == -233)
        this->one_blob_only = false;

    return 0;
}

MLAYER_SCALE_TEMPLATE
std::int32_t Scale<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::load_model(const ModelBinT& mb)
{
    if (scale_data_size == -233)
        return 0;

    scale_data = mb.load(scale_data_size, 1);
    if (scale_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(scale_data_size, 1);
        if (bias_data.empty())
            return -100;
    }

    return 0;
}

MLAYER_SCALE_TEMPLATE
std::int32_t Scale<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::forward_inplace(std::vector<MatT>& bottom_top_blobs, const OptionT& opt) const
{
    MatT& bottom_top_blob = bottom_top_blobs[0];
    const MatT& scale_blob = bottom_top_blobs[1];

    std::int32_t dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        std::int32_t w = bottom_top_blob.w;

        float* ptr = bottom_top_blob;

        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t i = 0; i < w; i++)
            {
                ptr[i] = ptr[i] * scale_blob[static_cast<std::size_t>(i)] + bias_data[static_cast<std::size_t>(i)];
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t i = 0; i < w; i++)
            {
                ptr[i] *= scale_blob[static_cast<std::size_t>(i)];
            }
        }
    }

    if (dims == 2)
    {
        std::int32_t w = bottom_top_blob.w;
        std::int32_t h = bottom_top_blob.h;

        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
                float s = scale_blob[static_cast<std::size_t>(i)];
                float bias = bias_data[static_cast<std::size_t>(i)];

                for (std::int32_t j = 0; j < w; j++)
                {
                    ptr[j] = ptr[j] * s + bias;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
                float s = scale_blob[static_cast<std::size_t>(i)];

                for (std::int32_t j = 0; j < w; j++)
                {
                    ptr[j] *= s;
                }
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

        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                float s = scale_blob[static_cast<std::size_t>(q)];
                float bias = bias_data[static_cast<std::size_t>(q)];

                for (std::int32_t i = 0; i < size; i++)
                {
                    ptr[i] = ptr[i] * s + bias;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                float s = scale_blob[static_cast<std::size_t>(q)];

                for (std::int32_t i = 0; i < size; i++)
                {
                    ptr[i] *= s;
                }
            }
        }
    }

    return 0;
}

MLAYER_SCALE_TEMPLATE
std::int32_t Scale<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const
{
    std::vector<MatT> bottom_top_blobs(2);
    bottom_top_blobs[0] = bottom_top_blob;
    bottom_top_blobs[1] = scale_data;

    return forward_inplace(bottom_top_blobs, opt);
}


#undef MLAYER_SCALE_TEMPLATE

}  // namespace mlayer

#endif  // MLAYER_SCALE_HPP
