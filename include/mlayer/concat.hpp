// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_CONCAT_HPP
#define MLAYER_CONCAT_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class Concat : public LayerT {
public:
    Concat();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs, const OptionT& opt) const override;

public:
    std::int32_t axis;
};

#define MLAYER_CONCAT_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>

MLAYER_CONCAT_TEMPLATE
Concat<LayerT, MatT, OptionT, ParamDictT>::Concat()
{
    this->one_blob_only = false;
    this->support_inplace = false;
}

MLAYER_CONCAT_TEMPLATE
std::int32_t Concat<LayerT, MatT, OptionT, ParamDictT>::load_param(const ParamDictT& pd)
{
    axis = pd.get(0, 0);

    return 0;
}

MLAYER_CONCAT_TEMPLATE
std::int32_t Concat<LayerT, MatT, OptionT, ParamDictT>::forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs, const OptionT& opt) const
{
    std::int32_t dims = bottom_blobs[0].dims;
    std::size_t elemsize = bottom_blobs[0].elemsize;
    std::int32_t positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        // concat vector
        // total length
        std::int32_t top_w = 0;
        for (std::size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const MatT& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        MatT& top_blob = top_blobs[0];
        top_blob.create(top_w, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        unsigned char* outptr = top_blob;
        for (std::size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const MatT& bottom_blob = bottom_blobs[b];

            std::int32_t w = bottom_blob.w;

            const unsigned char* ptr = bottom_blob;
            memcpy(outptr, ptr, w * elemsize);

            outptr += w * elemsize;
        }
    }

    if (dims == 2 && positive_axis == 0)
    {
        // concat image
        std::int32_t w = bottom_blobs[0].w;

        // total height
        std::int32_t top_h = 0;
        for (std::size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const MatT& bottom_blob = bottom_blobs[b];
            top_h += bottom_blob.h;
        }

        MatT& top_blob = top_blobs[0];
        top_blob.create(w, top_h, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        unsigned char* outptr = top_blob;
        for (std::size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const MatT& bottom_blob = bottom_blobs[b];

            std::int32_t size = w * bottom_blob.h;

            const unsigned char* ptr = bottom_blob;
            memcpy(outptr, ptr, size * elemsize);

            outptr += size * elemsize;
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        // interleave image row
        std::int32_t h = bottom_blobs[0].h;

        // total width
        std::int32_t top_w = 0;
        for (std::size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const MatT& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        MatT& top_blob = top_blobs[0];
        top_blob.create(top_w, h, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t i = 0; i < h; i++)
        {
            unsigned char* outptr = top_blob.template row<unsigned char>(i);
            for (std::size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const MatT& bottom_blob = bottom_blobs[b];

                const unsigned char* ptr = bottom_blob.template row<const unsigned char>(i);
                memcpy(outptr, ptr, bottom_blob.w * elemsize);

                outptr += bottom_blob.w * elemsize;
            }
        }
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
        // concat dim
        std::int32_t w = bottom_blobs[0].w;
        std::int32_t h = bottom_blobs[0].h;
        std::int32_t d = bottom_blobs[0].d;

        // total channels
        std::int32_t top_channels = 0;
        for (std::size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const MatT& bottom_blob = bottom_blobs[b];
            top_channels += bottom_blob.c;
        }

        MatT& top_blob = top_blobs[0];
        // 3-D blobs have d == 0; using the 4-D overload would allocate nothing.
        if (dims == 3)
            top_blob.create(w, h, top_channels, elemsize, opt.blob_allocator);
        else
            top_blob.create(w, h, d, top_channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        top_blob.dims = dims;

        std::int32_t q = 0;
        for (std::size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const MatT& bottom_blob = bottom_blobs[b];

            const std::int32_t channels = bottom_blob.c;
            const std::size_t channel_bytes =
                static_cast<std::size_t>(w) * h * d * elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t c = 0; c < channels; c++)
            {
                std::memcpy(top_blob.channel(q + c).data, bottom_blob.channel(c).data, channel_bytes);
            }

            q += channels;
        }
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        // interleave dim height
        std::int32_t w = bottom_blobs[0].w;
        std::int32_t d = bottom_blobs[0].d;
        std::int32_t channels = bottom_blobs[0].c;

        // total height
        std::int32_t top_h = 0;
        for (std::size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const MatT& bottom_blob = bottom_blobs[b];
            top_h += bottom_blob.h;
        }

        MatT& top_blob = top_blobs[0];
        if (dims == 3)
            top_blob.create(w, top_h, channels, elemsize, opt.blob_allocator);
        else
            top_blob.create(w, top_h, d, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        top_blob.dims = dims;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; q++)
        {
            unsigned char* outptr = top_blob.channel(q);

            for (std::int32_t i = 0; i < d; i++)
            {
                for (std::size_t b = 0; b < bottom_blobs.size(); b++)
                {
                    const MatT& bottom_blob = bottom_blobs[b];

                    std::int32_t size = bottom_blob.w * bottom_blob.h;

                    const unsigned char* ptr = bottom_blob.channel(q).depth(i);
                    memcpy(outptr, ptr, size * elemsize);

                    outptr += size * elemsize;
                }
            }
        }
    }

    if ((dims == 3 && positive_axis == 2) || (dims == 4 && positive_axis == 3))
    {
        // interleave dim width
        std::int32_t h = bottom_blobs[0].h;
        std::int32_t d = bottom_blobs[0].d;
        std::int32_t channels = bottom_blobs[0].c;

        // total width
        std::int32_t top_w = 0;
        for (std::size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const MatT& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        MatT& top_blob = top_blobs[0];
        top_blob.create(top_w, h, d, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        top_blob.dims = dims;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; q++)
        {
            unsigned char* outptr = top_blob.channel(q);

            for (std::int32_t i = 0; i < d; i++)
            {
                for (std::int32_t j = 0; j < h; j++)
                {
                    for (std::size_t b = 0; b < bottom_blobs.size(); b++)
                    {
                        const MatT& bottom_blob = bottom_blobs[b];

                        const unsigned char* ptr = bottom_blob.channel(q).depth(i).template row<const unsigned char>(j);
                        memcpy(outptr, ptr, bottom_blob.w * elemsize);

                        outptr += bottom_blob.w * elemsize;
                    }
                }
            }
        }
    }

    if (dims == 4 && positive_axis == 1)
    {
        // interleave dim depth
        std::int32_t w = bottom_blobs[0].w;
        std::int32_t h = bottom_blobs[0].h;
        std::int32_t channels = bottom_blobs[0].c;

        // total depth
        std::int32_t top_d = 0;
        for (std::size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const MatT& bottom_blob = bottom_blobs[b];
            top_d += bottom_blob.d;
        }

        MatT& top_blob = top_blobs[0];
        top_blob.create(w, h, top_d, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; q++)
        {
            unsigned char* outptr = top_blob.channel(q);

            for (std::size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const MatT& bottom_blob = bottom_blobs[b];

                std::int32_t size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

                const unsigned char* ptr = bottom_blob.channel(q);
                memcpy(outptr, ptr, size * elemsize);

                outptr += size * elemsize;
            }
        }
    }

    return 0;
}


#undef MLAYER_CONCAT_TEMPLATE

}  // namespace mlayer

#endif  // MLAYER_CONCAT_HPP
