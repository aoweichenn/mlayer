// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_ELTWISE_HPP
#define MLAYER_ELTWISE_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate>
class Eltwise : public LayerT {
public:
    Eltwise();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs, const OptionT& opt) const override;

    enum Operation {
        Operation_PROD = 0,
        Operation_SUM = 1,
        Operation_MAX = 2
    };

public:
    std::int32_t op_type;
    MatT coeffs;
};

#define MLAYER_ELTWISE_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate>

MLAYER_ELTWISE_TEMPLATE
Eltwise<LayerT, MatT, OptionT, ParamDictT, Validate>::Eltwise()
{
    this->one_blob_only = false;
    this->support_inplace = false; // TODO inplace reduction
}

MLAYER_ELTWISE_TEMPLATE
std::int32_t Eltwise<LayerT, MatT, OptionT, ParamDictT, Validate>::load_param(const ParamDictT& pd)
{
    op_type = pd.get(0, 0);
    coeffs = pd.get(1, MatT());

    const std::int32_t coeffs_type = pd.type(1);

if constexpr (Validate) {
    if (op_type < Operation_PROD || op_type > Operation_MAX)
        return -1;

    if (coeffs_type != 0 && coeffs_type != 4 && coeffs_type != 5 && coeffs_type != 6)
        return -1;

    if ((coeffs.dims != 0 || coeffs.w != 0 || coeffs.data) && (coeffs.dims != 1 || coeffs.w < 0 || coeffs.elempack != 1 || coeffs.elemsize != 4u || (coeffs.w > 0 && !coeffs.data)))
        return -1;
}

    // convert integer text arrays without modifying the shared data
    if (coeffs_type == 5 && !coeffs.empty())
    {
        MatT converted(coeffs.w);
        if (converted.empty())
            return -100;

        const std::int32_t* p = coeffs;
        for (std::int32_t i = 0; i < coeffs.w; i++)
            converted[i] = (float)p[i];

        coeffs = converted;
    }

    return 0;
}

MLAYER_ELTWISE_TEMPLATE
std::int32_t Eltwise<LayerT, MatT, OptionT, ParamDictT, Validate>::forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs, const OptionT& opt) const
{
    const MatT& bottom_blob = bottom_blobs[0];
    std::int32_t w = bottom_blob.w;
    std::int32_t h = bottom_blob.h;
    std::int32_t d = bottom_blob.d;
    std::int32_t channels = bottom_blob.c;
    std::int32_t size = w * h * d;

    MatT& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (op_type == Operation_PROD)
    {
        // first blob
        const MatT& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

            for (std::int32_t i = 0; i < size; i++)
            {
                outptr[i] = ptr[i] * ptr1[i];
            }
        }

        for (std::size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const MatT& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                for (std::int32_t i = 0; i < size; i++)
                {
                    outptr[i] *= ptr[i];
                }
            }
        }
    }
    else if (op_type == Operation_SUM)
    {
        if (coeffs.w == 0)
        {
            // first blob
            const MatT& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                for (std::int32_t i = 0; i < size; i++)
                {
                    outptr[i] = ptr[i] + ptr1[i];
                }
            }

            for (std::size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const MatT& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (std::int32_t q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);

                    for (std::int32_t i = 0; i < size; i++)
                    {
                        outptr[i] += ptr[i];
                    }
                }
            }
        }
        else
        {
            // first blob
            const MatT& bottom_blob1 = bottom_blobs[1];
            float coeff0 = coeffs[0];
            float coeff1 = coeffs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                for (std::int32_t i = 0; i < size; i++)
                {
                    outptr[i] = ptr[i] * coeff0 + ptr1[i] * coeff1;
                }
            }

            for (std::size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const MatT& bottom_blob1 = bottom_blobs[b];
                float coeff = coeffs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (std::int32_t q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);

                    for (std::int32_t i = 0; i < size; i++)
                    {
                        outptr[i] += ptr[i] * coeff;
                    }
                }
            }
        }
    }
    else if (op_type == Operation_MAX)
    {
        // first blob
        const MatT& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

            for (std::int32_t i = 0; i < size; i++)
            {
                outptr[i] = std::max(ptr[i], ptr1[i]);
            }
        }

        for (std::size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const MatT& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                for (std::int32_t i = 0; i < size; i++)
                {
                    outptr[i] = std::max(outptr[i], ptr[i]);
                }
            }
        }
    }

    return 0;
}


#undef MLAYER_ELTWISE_TEMPLATE

}  // namespace mlayer

#endif  // MLAYER_ELTWISE_HPP
