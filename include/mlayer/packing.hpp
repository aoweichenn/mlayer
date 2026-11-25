// Copyright 2020 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_PACKING_HPP
#define MLAYER_PACKING_HPP

#include <cstddef>
#include <cstdint>

namespace mlayer {

// elempack conversion: interleave `in_elempack` channels into `out_elempack`
// (and back). The graph packing pass is not enabled, so this is mainly the
// compatibility layer a param file may reference.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class Packing : public LayerT {
public:
    Packing();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

public:
    std::int32_t out_elempack;
    std::int32_t use_padding;
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
    return 0;
}

MLAYER_PACKING_TEMPLATE
std::int32_t Packing<LayerT, MatT, OptionT, ParamDictT>::forward(const MatT& bottom_blob, MatT& top_blob,
                                                                const OptionT& opt) const {
    const std::int32_t in_pack = bottom_blob.elempack;
    if (in_pack == this->out_elempack) {
        top_blob = bottom_blob;
        return 0;
    }
    if (this->cast_type_from != 0 || this->cast_type_to != 0) {
        // storage casts are handled by the Cast layer
        return -1;
    }

    const std::int32_t w = bottom_blob.w;
    const std::int32_t h = bottom_blob.h;
    const std::int32_t d = bottom_blob.d;
    const std::int32_t channels = bottom_blob.c;
    const std::int32_t dims = bottom_blob.dims;
    const std::size_t elemsize = bottom_blob.elemsize / static_cast<std::size_t>(in_pack);

    if (!this->use_padding) {
        if (dims == 1 && w * in_pack % this->out_elempack != 0) {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims == 2 && h * in_pack % this->out_elempack != 0) {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims >= 3 && channels * in_pack % this->out_elempack != 0) {
            top_blob = bottom_blob;
            return 0;
        }
    }

    // Only scalar (elempack 1) blobs are produced by this runtime, so both the
    // pack and unpack directions operate on channel-major data.
    const std::int32_t total_channels = channels * in_pack;
    const std::int32_t out_channels = total_channels / this->out_elempack;
    const std::int32_t plane = w * h * d;

    if (dims == 1) {
        top_blob.create(w * in_pack / this->out_elempack, elemsize * this->out_elempack, opt.blob_allocator);
    } else if (dims == 2) {
        top_blob.create(w, h * in_pack / this->out_elempack, elemsize * this->out_elempack, opt.blob_allocator);
    } else if (dims == 3) {
        top_blob.create(w, h, out_channels, elemsize * this->out_elempack, opt.blob_allocator);
    } else {
        top_blob.create(w, h, d, out_channels, elemsize * this->out_elempack, opt.blob_allocator);
    }
    if (top_blob.empty()) {
        return -100;
    }
    if (elemsize != 4u) {
        return -1;
    }

    const float* src = bottom_blob;
    float* dst = top_blob;
    // channel-major -> interleaved: dst[c * plane * pack + i * pack + k]
    for (std::int32_t c = 0; c < out_channels; ++c) {
        for (std::int32_t i = 0; i < plane; ++i) {
            for (std::int32_t k = 0; k < this->out_elempack; ++k) {
                const std::int32_t sc = c * this->out_elempack + k;
                dst[static_cast<std::size_t>(c) * plane * this->out_elempack +
                    static_cast<std::size_t>(i) * this->out_elempack + k] =
                    src[static_cast<std::size_t>(sc) * plane + i];
            }
        }
    }
    return 0;
}

}  // namespace mlayer

#endif  // MLAYER_PACKING_HPP
