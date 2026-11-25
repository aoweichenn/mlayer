// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_SOFTMAX_HPP
#define MLAYER_SOFTMAX_HPP

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class Softmax : public LayerT {
public:
    Softmax();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const override;

public:
    std::int32_t axis;
};

#define MLAYER_SOFTMAX_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>

MLAYER_SOFTMAX_TEMPLATE
Softmax<LayerT, MatT, OptionT, ParamDictT>::Softmax()
{
    this->one_blob_only = true;
    this->support_inplace = true;
}

MLAYER_SOFTMAX_TEMPLATE
std::int32_t Softmax<LayerT, MatT, OptionT, ParamDictT>::load_param(const ParamDictT& pd)
{
    axis = pd.get(0, 0);

    // the original softmax handle axis on 3-dim blob incorrectly
    // ask user to regenerate param instead of producing wrong result
    const std::int32_t fixbug0 = pd.get(1, 0);
    if (fixbug0 == 0 && axis != 0)
    {
        return -1;
    }

    return 0;
}

inline void softmax(float* _ptr, std::int32_t size)
{
    // reduce max
    float max = -FLT_MAX;
    {
        const float* ptr = _ptr;
        for (std::int32_t i = 0; i < size; i++)
        {
            max = std::max(max, *ptr++);
        }
    }

    // reduce exp(x - max)
    float sum = 0.f;
    {
        float* ptr = _ptr;
        for (std::int32_t i = 0; i < size; i++)
        {
            *ptr = std::exp(*ptr - max);
            sum += *ptr;
            ptr++;
        }
    }

    // div sum
    {
        float* ptr = _ptr;
        for (std::int32_t i = 0; i < size; i++)
        {
            *ptr++ /= sum;
        }
    }
}

inline void softmax(float* _ptr, std::int32_t size, std::int32_t stride)
{
    // reduce max
    float max = -FLT_MAX;
    {
        const float* ptr = _ptr;
        for (std::int32_t i = 0; i < size; i++)
        {
            max = std::max(max, *ptr);
            ptr += stride;
        }
    }

    // reduce exp(x - max)
    float sum = 0.f;
    {
        float* ptr = _ptr;
        for (std::int32_t i = 0; i < size; i++)
        {
            *ptr = std::exp(*ptr - max);
            sum += *ptr;
            ptr += stride;
        }
    }

    // div sum
    {
        float* ptr = _ptr;
        for (std::int32_t i = 0; i < size; i++)
        {
            *ptr /= sum;
            ptr += stride;
        }
    }
}

MLAYER_SOFTMAX_TEMPLATE
std::int32_t Softmax<LayerT, MatT, OptionT, ParamDictT>::forward_inplace(MatT& bottom_top_blob, const OptionT& opt) const
{
    // value = std::exp( value - global max value )
    // sum all value
    // value = value / sum

    const std::int32_t dims = bottom_top_blob.dims;
    const std::int32_t w = bottom_top_blob.w;
    const std::int32_t h = bottom_top_blob.h;
    const std::int32_t d = bottom_top_blob.d;
    const std::int32_t channels = bottom_top_blob.c;
    const std::int32_t positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        float* ptr = bottom_top_blob;

        softmax(ptr, w);
    }

    if (dims == 2 && positive_axis == 0)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t i = 0; i < w; i++)
        {
            float* ptr = static_cast<float*>(bottom_top_blob) + i;

            softmax(ptr, h, w);
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);

            softmax(ptr, w);
        }
    }

    if (dims == 3 && positive_axis == 0)
    {
        const std::int32_t size = w * h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t i = 0; i < size; i++)
        {
            float* ptr = static_cast<float*>(bottom_top_blob) + i;

            softmax(ptr, channels, static_cast<std::int32_t>(bottom_top_blob.cstep));
        }
    }

    if (dims == 3 && positive_axis == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (std::int32_t i = 0; i < w; i++)
            {
                softmax(ptr, h, w);
                ptr += 1;
            }
        }
    }

    if (dims == 3 && positive_axis == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (std::int32_t i = 0; i < h; i++)
            {
                softmax(ptr, w);
                ptr += w;
            }
        }
    }

    if (dims == 4 && positive_axis == 0)
    {
        const std::int32_t size = w * h * d;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t i = 0; i < size; i++)
        {
            float* ptr = static_cast<float*>(bottom_top_blob) + i;

            softmax(ptr, channels, static_cast<std::int32_t>(bottom_top_blob.cstep));
        }
    }

    if (dims == 4 && positive_axis == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (std::int32_t i = 0; i < w * h; i++)
            {
                softmax(ptr, d, w * h);
                ptr += 1;
            }
        }
    }

    if (dims == 4 && positive_axis == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; q++)
        {
            for (std::int32_t i = 0; i < d; i++)
            {
                float* ptr = bottom_top_blob.channel(q).depth(i);

                for (std::int32_t j = 0; j < w; j++)
                {
                    softmax(ptr, h, w);
                    ptr += 1;
                }
            }
        }
    }

    if (dims == 4 && positive_axis == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (std::int32_t i = 0; i < d; i++)
            {
                for (std::int32_t j = 0; j < h; j++)
                {
                    softmax(ptr, w);
                    ptr += w;
                }
            }
        }
    }

    return 0;
}


#undef MLAYER_SOFTMAX_TEMPLATE

}  // namespace mlayer

#endif  // MLAYER_SOFTMAX_HPP
