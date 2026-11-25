// Copyright 2020 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_PACKING_HPP
#define MLAYER_PACKING_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace mlayer {

// elempack conversion: regroup scalar lanes into `out_elempack` wide elements
// (and back again), matching ncnn's interleaved layout.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class Packing : public LayerT {
public:
    Packing();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

public:
    std::int32_t out_elempack;
    std::int32_t use_padding;
    // Retained for param compatibility; CPU ncnn does not act on them.
    std::int32_t cast_type_from;
    std::int32_t cast_type_to;
};

#define MLAYER_PACKING_TEMPLATE template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>

MLAYER_PACKING_TEMPLATE
Packing<LayerT, MatT, OptionT, ParamDictT>::Packing() {
    this->one_blob_only = true;
    this->support_inplace = false;
    this->out_elempack = 1;
    this->use_padding = 0;
    this->cast_type_from = 0;
    this->cast_type_to = 0;
}

MLAYER_PACKING_TEMPLATE
std::int32_t Packing<LayerT, MatT, OptionT, ParamDictT>::load_param(const ParamDictT& pd) {
    this->out_elempack = pd.get(0, 1);
    this->use_padding = pd.get(1, 0);
    this->cast_type_from = pd.get(2, 0);
    this->cast_type_to = pd.get(3, 0);
    return this->out_elempack > 0 ? 0 : -1;
}

MLAYER_PACKING_TEMPLATE
std::int32_t Packing<LayerT, MatT, OptionT, ParamDictT>::forward(const MatT& bottom_blob, MatT& top_blob,
                                                                const OptionT& opt) const {
    const std::int32_t in_pack = bottom_blob.elempack;
    if (in_pack == this->out_elempack) {
        top_blob = bottom_blob;
        return 0;
    }
    // ncnn's CPU Packing layer ignores cast_type_from/to: the fields are only
    // consumed by the Vulkan packing shader, and CPU models use an explicit Cast
    // layer for storage conversions. Keep the same behaviour so params that
    // carry these hints still run.

    const std::int32_t w = bottom_blob.w;
    const std::int32_t h = bottom_blob.h;
    const std::int32_t d = bottom_blob.d;
    const std::int32_t channels = bottom_blob.c;
    const std::int32_t dims = bottom_blob.dims;
    const std::size_t elemsize = bottom_blob.elemsize;
    const std::size_t scalar_size = elemsize / static_cast<std::size_t>(in_pack);
    if (scalar_size == 0u) {
        return -1;
    }

    // Source logical layout: `src_channels` scalar lanes of `plane` positions.
    const std::int32_t plane = dims == 1 ? w : (dims == 2 ? w : w * h * d);
    const std::int32_t src_channels = dims <= 2 ? (dims == 1 ? in_pack : h * in_pack) : channels * in_pack;
    if (!this->use_padding) {
        if (src_channels % this->out_elempack != 0) {
            top_blob = bottom_blob;
            return 0;
        }
    }
    const std::int32_t dst_channels = (src_channels + this->out_elempack - 1) / this->out_elempack;

    const std::int32_t total_scalars = src_channels * plane;
    const std::int32_t padded_scalars =
        (total_scalars + this->out_elempack - 1) / this->out_elempack * this->out_elempack;
    if (dims == 1) {
        const std::int32_t outw = total_scalars / this->out_elempack;
        top_blob.create(outw, scalar_size * static_cast<std::size_t>(this->out_elempack), this->out_elempack,
                        opt.blob_allocator);
    } else if (dims == 2) {
        top_blob.create(w, dst_channels, scalar_size * static_cast<std::size_t>(this->out_elempack),
                        this->out_elempack, opt.blob_allocator);
    } else if (dims == 3) {
        top_blob.create(w, h, dst_channels, scalar_size * static_cast<std::size_t>(this->out_elempack),
                        this->out_elempack, opt.blob_allocator);
    } else {
        top_blob.create(w, h, d, dst_channels, scalar_size * static_cast<std::size_t>(this->out_elempack),
                        this->out_elempack, opt.blob_allocator);
    }
    if (top_blob.empty()) {
        return -100;
    }

    const auto* src = static_cast<const std::uint8_t*>(bottom_blob.data);
    auto* dst = static_cast<std::uint8_t*>(top_blob.data);
    if (padded_scalars != total_scalars) {
        std::memset(dst, 0, static_cast<std::size_t>(padded_scalars) * scalar_size);
    }
    for (std::int32_t c = 0; c < src_channels; ++c) {
        const std::int32_t sc = c / in_pack;
        const std::int32_t sl = c % in_pack;
        const std::int32_t dc = c / this->out_elempack;
        const std::int32_t dl = c % this->out_elempack;
        const auto* src_ch = dims >= 3 ? static_cast<const std::uint8_t*>(bottom_blob.channel(sc)) : src;
        auto* dst_ch = dims >= 3 ? static_cast<std::uint8_t*>(top_blob.channel(dc)) : dst;
        for (std::int32_t i = 0; i < plane; ++i) {
            const std::size_t src_off =
                ((dims >= 3 ? 0u : static_cast<std::size_t>(sc) * plane * in_pack) +
                 static_cast<std::size_t>(i) * in_pack + static_cast<std::size_t>(sl)) *
                scalar_size;
            const std::size_t dst_off =
                (dims == 1
                     ? (static_cast<std::size_t>(c) * plane + static_cast<std::size_t>(i))
                     : (dims >= 3 ? 0u
                                  : static_cast<std::size_t>(dc) * plane * this->out_elempack) +
                           static_cast<std::size_t>(i) * this->out_elempack +
                           static_cast<std::size_t>(dl)) *
                scalar_size;
            std::memcpy(dst_ch + dst_off, src_ch + src_off, scalar_size);
        }
    }
    return 0;
}

}  // namespace mlayer

#endif  // MLAYER_PACKING_HPP
