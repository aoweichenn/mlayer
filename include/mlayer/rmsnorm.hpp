// Copyright 2024 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_RMSNORM_HPP
#define MLAYER_RMSNORM_HPP

#include <cmath>
#include <cstdint>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT>
class RMSNorm : public LayerT {
public:
    RMSNorm();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t load_model(const ModelBinT& mb) override;
    std::int32_t forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const override;

public:
    std::int32_t affine_size;
    float eps;
    std::int32_t affine;
    MatT gamma_data;
};

#define MLAYER_RMSNORM_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT>

MLAYER_RMSNORM_TEMPLATE
RMSNorm<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::RMSNorm()
{
    this->one_blob_only = true;
    this->support_inplace = true;
}

MLAYER_RMSNORM_TEMPLATE
std::int32_t RMSNorm<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::load_param(const ParamDictT& pd)
{
    affine_size = pd.get(0, 0);
    eps = pd.get(1, 0.001f);
    affine = pd.get(2, 1);

    return 0;
}

MLAYER_RMSNORM_TEMPLATE
std::int32_t RMSNorm<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::load_model(const ModelBinT& mb)
{
    if (affine == 0)
        return 0;

    gamma_data = mb.load(affine_size, 1);
    if (gamma_data.empty())
        return -100;

    return 0;
}

inline void rmsnorm(float* ptr, const float* gamma_ptr, float eps, std::int32_t size)
{
    float sqsum = 0.f;
    for (std::int32_t i = 0; i < size; i++)
    {
        sqsum += ptr[i] * ptr[i];
    }

    float rms = sqsum / size;

    float a = 1.f / std::sqrt(rms + eps);

    if (gamma_ptr)
    {
        for (std::int32_t i = 0; i < size; i++)
        {
            ptr[i] = (ptr[i] * a) * gamma_ptr[i];
        }
    }
    else
    {
        for (std::int32_t i = 0; i < size; i++)
        {
            ptr[i] = ptr[i] * a;
        }
    }
}

MLAYER_RMSNORM_TEMPLATE
std::int32_t RMSNorm<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const
{
    // x = x / sqrt(rms + eps) * gamma

    std::int32_t dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        std::int32_t w = bottom_top_blob.w;
        // assert affine_size == w

        float* ptr = bottom_top_blob;
        rmsnorm(ptr, gamma_data, eps, w);
    }

    if (dims == 2)
    {
        std::int32_t w = bottom_top_blob.w;
        std::int32_t h = bottom_top_blob.h;
        // assert affine_size == w

        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            rmsnorm(ptr, gamma_data, eps, w);
        }
    }

    if (dims == 3)
    {
        std::int32_t w = bottom_top_blob.w;
        std::int32_t h = bottom_top_blob.h;
        std::int32_t channels = bottom_top_blob.c;

        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < channels; q++)
            {
                for (std::int32_t i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.channel(q).row(i);
                    rmsnorm(ptr, gamma_data, eps, w);
                }
            }
        }
        else // if (affine_size == size)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                rmsnorm(ptr, gamma_data, eps, w * h);
            }
        }
    }

    if (dims == 4)
    {
        std::int32_t w = bottom_top_blob.w;
        std::int32_t h = bottom_top_blob.h;
        std::int32_t d = bottom_top_blob.d;
        std::int32_t channels = bottom_top_blob.c;

        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < channels; q++)
            {
                for (std::int32_t z = 0; z < d; z++)
                {
                    for (std::int32_t i = 0; i < h; i++)
                    {
                        float* ptr = bottom_top_blob.channel(q).depth(z).row(i);
                        rmsnorm(ptr, gamma_data, eps, w);
                    }
                }
            }
        }
        else if (affine_size == w * h)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < channels; q++)
            {
                for (std::int32_t z = 0; z < d; z++)
                {
                    float* ptr = bottom_top_blob.channel(q).depth(z);
                    rmsnorm(ptr, gamma_data, eps, w * h);
                }
            }
        }
        else // if (affine_size == w * h * d)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                rmsnorm(ptr, gamma_data, eps, w * h * d);
            }
        }
    }

    return 0;
}


#undef MLAYER_RMSNORM_TEMPLATE

}  // namespace mlayer

#endif  // MLAYER_RMSNORM_HPP
