// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_NORMALIZE_HPP
#define MLAYER_NORMALIZE_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>


namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT>
class Normalize : public LayerT {
public:
    Normalize();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t load_model(const ModelBinT& mb) override;
    std::int32_t forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const override;

public:
    std::int32_t across_spatial;
    std::int32_t across_channel;
    std::int32_t channel_shared;
    float eps;
    std::int32_t eps_mode;
    std::int32_t scale_data_size;
    MatT scale_data;
};

#define MLAYER_NORMALIZE_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT>

MLAYER_NORMALIZE_TEMPLATE
Normalize<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::Normalize()
{
    this->one_blob_only = true;
    this->support_inplace = true;
}

MLAYER_NORMALIZE_TEMPLATE
std::int32_t Normalize<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::load_param(const ParamDictT& pd)
{
    across_spatial = pd.get(0, 0);
    across_channel = pd.get(4, 1);
    channel_shared = pd.get(1, 0);
    eps = pd.get(2, 0.0001f);
    eps_mode = pd.get(9, 0);
    scale_data_size = pd.get(3, 0);

    return 0;
}

MLAYER_NORMALIZE_TEMPLATE
std::int32_t Normalize<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::load_model(const ModelBinT& mb)
{
    scale_data = mb.load(scale_data_size, 1);
    if (scale_data.empty())
        return -100;

    return 0;
}

MLAYER_NORMALIZE_TEMPLATE
std::int32_t Normalize<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const
{
    std::int32_t w = bottom_top_blob.w;
    std::int32_t h = bottom_top_blob.h;
    std::int32_t d = bottom_top_blob.d;
    std::int32_t channels = bottom_top_blob.c;
    std::size_t elemsize = bottom_top_blob.elemsize;
    std::int32_t size = w * h * d;

    if (across_spatial && across_channel)
    {
        // square
        MatT square_sum_blob;
        square_sum_blob.create(channels, elemsize, opt.workspace_allocator);
        if (square_sum_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; q++)
        {
            const float* ptr = bottom_top_blob.channel(q);

            float ssum = 0.f;
            for (std::int32_t i = 0; i < size; i++)
            {
                ssum += ptr[i] * ptr[i];
            }

            square_sum_blob[q] = ssum;
        }

        float ssum = 0.f;
        for (std::int32_t q = 0; q < channels; q++)
        {
            ssum += square_sum_blob[q];
        }

        float a;
        if (eps_mode == 0) // caffe/mxnet
        {
            a = 1.f / sqrtf(ssum + eps);
        }
        else if (eps_mode == 1) // pytorch
        {
            a = 1.f / std::max(sqrtf(ssum), eps);
        }
        else //if (eps_mode == 2) // tensorflow
        {
            a = 1.f / sqrt(std::max(ssum, eps));
        }

        if (channel_shared)
        {
            float scale = a * scale_data[0];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                for (std::int32_t i = 0; i < size; i++)
                {
                    ptr[i] = ptr[i] * scale;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                float scale = a * scale_data[q];

                for (std::int32_t i = 0; i < size; i++)
                {
                    ptr[i] = ptr[i] * scale;
                }
            }
        }

        return 0;
    }

    if (across_spatial && !across_channel)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float ssum = 0.f;
            for (std::int32_t i = 0; i < size; i++)
            {
                ssum += ptr[i] * ptr[i];
            }

            float a;
            if (eps_mode == 0) // caffe/mxnet
            {
                a = 1.f / sqrtf(ssum + eps);
            }
            else if (eps_mode == 1) // pytorch
            {
                a = 1.f / std::max(sqrtf(ssum), eps);
            }
            else //if (eps_mode == 2) // tensorflow
            {
                a = 1.f / sqrtf(std::max(ssum, eps));
            }

            float scale = a * (channel_shared ? scale_data[0] : scale_data[q]);

            for (std::int32_t i = 0; i < size; i++)
            {
                ptr[i] = ptr[i] * scale;
            }
        }

        return 0;
    }

    if (!across_spatial && across_channel)
    {
        // square sum, 1 / sqrt(ssum)
        MatT square_sum_blob;
        square_sum_blob.create(size, elemsize, opt.workspace_allocator);
        if (square_sum_blob.empty())
            return -100;

        if (channel_shared)
        {
            float scale = scale_data[0];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t i = 0; i < size; i++)
            {
                float ssum = 0.f;
                for (std::int32_t q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_top_blob.channel(q);
                    ssum += ptr[i] * ptr[i];
                }

                float a;
                if (eps_mode == 0) // caffe/mxnet
                {
                    a = 1.f / sqrtf(ssum + eps);
                }
                else if (eps_mode == 1) // pytorch
                {
                    a = 1.f / std::max((float)sqrt(ssum), eps);
                }
                else //if (eps_mode == 2) // tensorflow
                {
                    a = 1.f / sqrtf(std::max(ssum, eps));
                }

                square_sum_blob[i] = a * scale;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                for (std::int32_t i = 0; i < size; i++)
                {
                    ptr[i] = ptr[i] * square_sum_blob[i];
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t i = 0; i < size; i++)
            {
                float ssum = 0.f;
                for (std::int32_t q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_top_blob.channel(q);
                    ssum += ptr[i] * ptr[i];
                }

                float a;
                if (eps_mode == 0) // caffe/mxnet
                {
                    a = 1.f / sqrtf(ssum + eps);
                }
                else if (eps_mode == 1) // pytorch
                {
                    a = 1.f / std::max(sqrtf(ssum), eps);
                }
                else //if (eps_mode == 2) // tensorflow
                {
                    a = 1.f / sqrtf(std::max(ssum, eps));
                }

                square_sum_blob[i] = a;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                float scale = scale_data[q];

                for (std::int32_t i = 0; i < size; i++)
                {
                    ptr[i] = ptr[i] * square_sum_blob[i] * scale;
                }
            }
        }

        return 0;
    }

    return 0;
}


#undef MLAYER_NORMALIZE_TEMPLATE

}  // namespace mlayer

#endif  // MLAYER_NORMALIZE_HPP
