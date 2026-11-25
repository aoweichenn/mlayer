// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_SLICE_HPP
#define MLAYER_SLICE_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate>
class Slice : public LayerT {
public:
    Slice();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs, const OptionT& opt) const override;

public:
    MatT slices;
    std::int32_t axis;
    MatT indices;
};

#define MLAYER_SLICE_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate>

MLAYER_SLICE_TEMPLATE
Slice<LayerT, MatT, OptionT, ParamDictT, Validate>::Slice()
{
}

MLAYER_SLICE_TEMPLATE
std::int32_t Slice<LayerT, MatT, OptionT, ParamDictT, Validate>::load_param(const ParamDictT& pd)
{
    slices = pd.get(0, MatT());
    axis = pd.get(1, 0);
    indices = pd.get(2, MatT());

if constexpr (Validate) {
    if (axis < -4 || axis > 3)
        return -1;

    {
        const std::int32_t slices_type = pd.type(0);
        if (slices_type != 0 && slices_type != 4 && slices_type != 5)
            return -1;

        if ((slices.dims != 0 || slices.w != 0 || slices.data) && (slices.dims != 1 || slices.w < 0 || slices.elempack != 1 || slices.elemsize != 4u || (slices.w > 0 && !slices.data)))
            return -1;
    }

    {
        const std::int32_t indices_type = pd.type(2);
        if (indices_type != 0 && indices_type != 4 && indices_type != 5)
            return -1;

        if ((indices.dims != 0 || indices.w != 0 || indices.data) && (indices.dims != 1 || indices.w < 0 || indices.elempack != 1 || indices.elemsize != 4u || (indices.w > 0 && !indices.data)))
            return -1;
    }

    const std::int32_t* slices_ptr = static_cast<const std::int32_t*>(static_cast<const void*>(slices.data));
    for (std::int32_t i = 0; i < slices.w; i++)
    {
        if (slices_ptr[i] <= 0 && slices_ptr[i] != -233)
            return -1;
    }
}

    return 0;
}

MLAYER_SLICE_TEMPLATE
std::int32_t Slice<LayerT, MatT, OptionT, ParamDictT, Validate>::forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs, const OptionT& opt) const
{
    const MatT& bottom_blob = bottom_blobs[0];
    std::int32_t dims = bottom_blob.dims;
    std::size_t elemsize = bottom_blob.elemsize;
    const std::int32_t* slices_ptr = static_cast<const std::int32_t*>(static_cast<const void*>(slices.data));
    const std::int32_t* indices_ptr = static_cast<const std::int32_t*>(static_cast<const void*>(indices.data));
    std::int32_t positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        std::int32_t w = bottom_blob.w;

        std::int32_t q = 0;
        for (std::size_t i = 0; i < top_blobs.size(); i++)
        {
            std::int32_t slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = w - q;
                }
                else
                {
                    std::int32_t indice = indices_ptr[i];
                    std::int32_t positive_indice = indice < 0 ? w + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<std::int32_t>((w - q) / (top_blobs.size() - i));
                }
            }

            MatT& top_blob = top_blobs[i];
            top_blob.create(slice, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            const auto* ptr = static_cast<const std::uint8_t*>(bottom_blob.data) + q * elemsize;
            auto* outptr = static_cast<std::uint8_t*>(top_blob.data);
            memcpy(outptr, ptr, slice * elemsize);

            q += slice;
        }

        return 0;
    }

    if (dims == 2 && positive_axis == 0)
    {
        std::int32_t w = bottom_blob.w;
        std::int32_t h = bottom_blob.h;

        std::int32_t q = 0;
        for (std::size_t i = 0; i < top_blobs.size(); i++)
        {
            std::int32_t slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = h - q;
                }
                else
                {
                    std::int32_t indice = indices_ptr[i];
                    std::int32_t positive_indice = indice < 0 ? h + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<std::int32_t>((h - q) / (top_blobs.size() - i));
                }
            }

            MatT& top_blob = top_blobs[i];
            top_blob.create(w, slice, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            std::size_t size = (std::size_t)w * slice;

            const auto* ptr = bottom_blob.template row<const std::uint8_t>(q);
            auto* outptr = static_cast<std::uint8_t*>(top_blob.data);
            memcpy(outptr, ptr, size * elemsize);

            q += slice;
        }

        return 0;
    }

    if (dims == 2 && positive_axis == 1)
    {
        std::int32_t w = bottom_blob.w;
        std::int32_t h = bottom_blob.h;

        std::int32_t q = 0;
        for (std::size_t i = 0; i < top_blobs.size(); i++)
        {
            std::int32_t slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = w - q;
                }
                else
                {
                    std::int32_t indice = indices_ptr[i];
                    std::int32_t positive_indice = indice < 0 ? w + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<std::int32_t>((w - q) / (top_blobs.size() - i));
                }
            }

            MatT& top_blob = top_blobs[i];
            top_blob.create(slice, h, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t j = 0; j < h; j++)
            {
                auto* outptr = top_blob.template row<std::uint8_t>(j);
                const auto* ptr = bottom_blob.template row<const std::uint8_t>(j) + q * elemsize;
                memcpy(outptr, ptr, slice * elemsize);
            }

            q += slice;
        }

        return 0;
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
        std::int32_t w = bottom_blob.w;
        std::int32_t h = bottom_blob.h;
        std::int32_t d = bottom_blob.d;
        std::int32_t channels = bottom_blob.c;

        std::int32_t q = 0;
        for (std::size_t i = 0; i < top_blobs.size(); i++)
        {
            std::int32_t slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = channels - q;
                }
                else
                {
                    std::int32_t indice = indices_ptr[i];
                    std::int32_t positive_indice = indice < 0 ? channels + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<std::int32_t>((channels - q) / (top_blobs.size() - i));
                }
            }

            MatT& top_blob = top_blobs[i];
            top_blob.create(w, h, d, slice, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            top_blob.dims = dims;

            std::size_t size = bottom_blob.cstep * slice;

            const auto* ptr = static_cast<const std::uint8_t*>(bottom_blob.channel(q).data);
            auto* outptr = static_cast<std::uint8_t*>(top_blob.data);
            memcpy(outptr, ptr, size * elemsize);

            q += slice;
        }

        return 0;
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        std::int32_t w = bottom_blob.w;
        std::int32_t h = bottom_blob.h;
        std::int32_t d = bottom_blob.d;
        std::int32_t channels = bottom_blob.c;

        std::int32_t q = 0;
        for (std::size_t i = 0; i < top_blobs.size(); i++)
        {
            std::int32_t slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = h - q;
                }
                else
                {
                    std::int32_t indice = indices_ptr[i];
                    std::int32_t positive_indice = indice < 0 ? h + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<std::int32_t>((h - q) / (top_blobs.size() - i));
                }
            }

            MatT& top_blob = top_blobs[i];
            top_blob.create(w, slice, d, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            top_blob.dims = dims;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t p = 0; p < channels; p++)
            {
                for (std::int32_t j = 0; j < d; j++)
                {
                    std::size_t size = (std::size_t)w * slice;

                    auto* outptr = static_cast<std::uint8_t*>(top_blob.channel(p).depth(j).data);
                    const auto* ptr = bottom_blob.channel(p).depth(j).template row<const std::uint8_t>(q);
                    memcpy(outptr, ptr, size * elemsize);
                }
            }

            q += slice;
        }

        return 0;
    }

    if ((dims == 3 && positive_axis == 2) || (dims == 4 && positive_axis == 3))
    {
        std::int32_t w = bottom_blob.w;
        std::int32_t h = bottom_blob.h;
        std::int32_t d = bottom_blob.d;
        std::int32_t channels = bottom_blob.c;

        std::int32_t q = 0;
        for (std::size_t i = 0; i < top_blobs.size(); i++)
        {
            std::int32_t slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = w - q;
                }
                else
                {
                    std::int32_t indice = indices_ptr[i];
                    std::int32_t positive_indice = indice < 0 ? w + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<std::int32_t>((w - q) / (top_blobs.size() - i));
                }
            }

            MatT& top_blob = top_blobs[i];
            top_blob.create(slice, h, d, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            top_blob.dims = dims;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t p = 0; p < channels; p++)
            {
                auto* outptr = static_cast<std::uint8_t*>(top_blob.channel(p).data);
                const MatT m = bottom_blob.channel(p);

                for (std::int32_t j = 0; j < d; j++)
                {
                    for (std::int32_t k = 0; k < h; k++)
                    {
                        const auto* ptr = m.depth(j).template row<const std::uint8_t>(k) + q * elemsize;
                        memcpy(outptr, ptr, slice * elemsize);

                        outptr += slice * elemsize;
                    }
                }
            }

            q += slice;
        }

        return 0;
    }

    if (dims == 4 && positive_axis == 1)
    {
        std::int32_t w = bottom_blob.w;
        std::int32_t h = bottom_blob.h;
        std::int32_t d = bottom_blob.d;
        std::int32_t channels = bottom_blob.c;

        std::int32_t q = 0;
        for (std::size_t i = 0; i < top_blobs.size(); i++)
        {
            std::int32_t slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = d - q;
                }
                else
                {
                    std::int32_t indice = indices_ptr[i];
                    std::int32_t positive_indice = indice < 0 ? d + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<std::int32_t>((d - q) / (top_blobs.size() - i));
                }
            }

            MatT& top_blob = top_blobs[i];
            top_blob.create(w, h, slice, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t p = 0; p < channels; p++)
            {
                std::size_t size = (std::size_t)w * h * slice;

                auto* outptr = static_cast<std::uint8_t*>(top_blob.channel(p).data);
                const auto* ptr = static_cast<const std::uint8_t*>(bottom_blob.channel(p).depth(q).data);
                memcpy(outptr, ptr, size * elemsize);
            }

            q += slice;
        }

        return 0;
    }

    return 0;
}


#undef MLAYER_SLICE_TEMPLATE

}  // namespace mlayer

#endif  // MLAYER_SLICE_HPP
