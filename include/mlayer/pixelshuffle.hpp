// Copyright 2020 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_PIXELSHUFFLE_HPP
#define MLAYER_PIXELSHUFFLE_HPP

#include <climits>
#include <cstddef>
#include <cstdint>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate>
class PixelShuffle : public LayerT {
public:
    PixelShuffle();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

public:
    std::int32_t upscale_factor;
    std::int32_t mode;
};

#define MLAYER_PIXELSHUFFLE_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate>

MLAYER_PIXELSHUFFLE_TEMPLATE
PixelShuffle<LayerT, MatT, OptionT, ParamDictT, Validate>::PixelShuffle()
{
    this->one_blob_only = true;
    this->support_inplace = false;
}

MLAYER_PIXELSHUFFLE_TEMPLATE
std::int32_t PixelShuffle<LayerT, MatT, OptionT, ParamDictT, Validate>::load_param(const ParamDictT& pd)
{
    upscale_factor = pd.get(0, 1);
    mode = pd.get(1, 0);

if constexpr (Validate) {
    if (mode < 0 || mode > 1)
        return -1;

    if (upscale_factor <= 0 || upscale_factor > INT_MAX / upscale_factor)
        return -1;
}

    return 0;
}

MLAYER_PIXELSHUFFLE_TEMPLATE
std::int32_t PixelShuffle<LayerT, MatT, OptionT, ParamDictT, Validate>::forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const
{
    std::int32_t w = bottom_blob.w;
    std::int32_t h = bottom_blob.h;
    std::int32_t channels = bottom_blob.c;
    std::size_t elemsize = bottom_blob.elemsize;

    std::int32_t outw = w * upscale_factor;
    std::int32_t outh = h * upscale_factor;
    std::int32_t outc = channels / (upscale_factor * upscale_factor);

    top_blob.create(outw, outh, outc, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t p = 0; p < outc; p++)
    {
        MatT m = top_blob.channel(p);

        for (std::int32_t sh = 0; sh < upscale_factor; sh++)
        {
            for (std::int32_t sw = 0; sw < upscale_factor; sw++)
            {
                std::int32_t q;
                if (mode == 0)
                    q = p * upscale_factor * upscale_factor + sh * upscale_factor + sw;
                else // if (mode == 1)
                    q = (sh * upscale_factor + sw) * outc + p;

                const float* sptr = bottom_blob.channel(q);

                for (std::int32_t i = 0; i < h; i++)
                {
                    float* outptr = m.row(i * upscale_factor + sh) + sw;
                    for (std::int32_t j = 0; j < w; j++)
                    {
                        outptr[0] = sptr[0];

                        sptr++;
                        outptr += upscale_factor;
                    }
                }
            }
        }
    }

    return 0;
}


#undef MLAYER_PIXELSHUFFLE_TEMPLATE

}  // namespace mlayer

#endif  // MLAYER_PIXELSHUFFLE_HPP
