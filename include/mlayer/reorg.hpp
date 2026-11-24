// Copyright 2018 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_REORG_HPP
#define MLAYER_REORG_HPP

#include <climits>
#include <cstddef>
#include <cstdint>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate>
class Reorg : public LayerT {
public:
    Reorg();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

public:
    std::int32_t stride;
    std::int32_t mode;
};

#define MLAYER_REORG_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate>

MLAYER_REORG_TEMPLATE
Reorg<LayerT, MatT, OptionT, ParamDictT, Validate>::Reorg()
{
    this->one_blob_only = true;
    this->support_inplace = false;
}

MLAYER_REORG_TEMPLATE
std::int32_t Reorg<LayerT, MatT, OptionT, ParamDictT, Validate>::load_param(const ParamDictT& pd)
{
    stride = pd.get(0, 1);
    mode = pd.get(1, 0);

if constexpr (Validate) {
    if (mode < 0 || mode > 1)
        return -1;

    if (stride <= 0 || stride > INT_MAX / stride)
        return -1;
}

    return 0;
}

MLAYER_REORG_TEMPLATE
std::int32_t Reorg<LayerT, MatT, OptionT, ParamDictT, Validate>::forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const
{
    std::int32_t w = bottom_blob.w;
    std::int32_t h = bottom_blob.h;
    std::int32_t channels = bottom_blob.c;
    std::size_t elemsize = bottom_blob.elemsize;

    std::int32_t outw = w / stride;
    std::int32_t outh = h / stride;
    std::int32_t outc = channels * stride * stride;

    top_blob.create(outw, outh, outc, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t q = 0; q < channels; q++)
    {
        const MatT m = bottom_blob.channel(q);

        for (std::int32_t sh = 0; sh < stride; sh++)
        {
            for (std::int32_t sw = 0; sw < stride; sw++)
            {
                std::int32_t p;
                if (mode == 0)
                    p = q * stride * stride + sh * stride + sw;
                else // if (mode == 1)
                    p = (sh * stride + sw) * channels + q;

                float* outptr = top_blob.channel(p);

                for (std::int32_t i = 0; i < outh; i++)
                {
                    const float* sptr = m.row(i * stride + sh) + sw;
                    for (std::int32_t j = 0; j < outw; j++)
                    {
                        outptr[0] = sptr[0];

                        sptr += stride;
                        outptr++;
                    }
                }
            }
        }
    }

    return 0;
}


#undef MLAYER_REORG_TEMPLATE

}  // namespace mlayer

#endif  // MLAYER_REORG_HPP
