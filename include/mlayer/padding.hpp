// Copyright 2018 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_PADDING_HPP
#define MLAYER_PADDING_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mlayer {
namespace detail {

// ncnn's padding border types: 0 = constant, 1 = replicate, 2 = reflect
// (mirror without repeating the edge sample).
template<typename T>
inline void pad_channel(const T* src, T* dst, std::int32_t w, std::int32_t h, std::int32_t pad_top,
                        std::int32_t pad_left, std::int32_t outw, std::int32_t outh, std::int32_t type, T v) {
    for (std::int32_t y = 0; y < outh; ++y) {
        std::int32_t sy = y - pad_top;
        if (type == 1) {
            sy = sy < 0 ? 0 : (sy >= h ? h - 1 : sy);
        } else if (type == 2) {
            sy = sy < 0 ? -sy : (sy >= h ? 2 * (h - 1) - sy : sy);
        }
        const bool row_outside = sy < 0 || sy >= h;
        const T* srow = row_outside ? nullptr : src + static_cast<std::size_t>(sy) * static_cast<std::size_t>(w);
        T* drow = dst + static_cast<std::size_t>(y) * static_cast<std::size_t>(outw);
        for (std::int32_t x = 0; x < outw; ++x) {
            std::int32_t sx = x - pad_left;
            if (type == 1) {
                sx = sx < 0 ? 0 : (sx >= w ? w - 1 : sx);
            } else if (type == 2) {
                sx = sx < 0 ? -sx : (sx >= w ? 2 * (w - 1) - sx : sx);
            }
            if (srow == nullptr || sx < 0 || sx >= w) {
                drow[x] = v;
            } else {
                drow[x] = srow[sx];
            }
        }
    }
}

}  // namespace detail

// Padding with per-dimension borders and an optional per-channel pad value.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT>
class Padding : public LayerT {
public:
    Padding();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t load_model(const ModelBinT& mb) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

public:
    std::int32_t top;
    std::int32_t bottom;
    std::int32_t left;
    std::int32_t right;
    std::int32_t type;
    float value;
    std::int32_t per_channel_pad_data_size;
    std::int32_t front;
    std::int32_t behind;
    MatT per_channel_pad_data;
};

#define MLAYER_PADDING_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT>

MLAYER_PADDING_TEMPLATE
Padding<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::Padding() {
    this->one_blob_only = true;
    this->support_inplace = false;
    this->top = 0;
    this->bottom = 0;
    this->left = 0;
    this->right = 0;
    this->type = 0;
    this->value = 0.0F;
    this->per_channel_pad_data_size = 0;
    this->front = 0;
    this->behind = 0;
}

MLAYER_PADDING_TEMPLATE
std::int32_t Padding<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::load_param(const ParamDictT& pd) {
    this->top = pd.get(0, 0);
    this->bottom = pd.get(1, 0);
    this->left = pd.get(2, 0);
    this->right = pd.get(3, 0);
    this->type = pd.get(4, 0);
    this->value = pd.get(5, 0.0F);
    this->per_channel_pad_data_size = pd.get(6, 0);
    this->front = pd.get(7, 0);
    this->behind = pd.get(8, 0);
    return 0;
}

MLAYER_PADDING_TEMPLATE
std::int32_t Padding<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::load_model(const ModelBinT& mb) {
    if (this->per_channel_pad_data_size > 0) {
        this->per_channel_pad_data = mb.load(this->per_channel_pad_data_size, 1);
        if (this->per_channel_pad_data.empty()) {
            return -100;
        }
    }
    return 0;
}

MLAYER_PADDING_TEMPLATE
std::int32_t Padding<LayerT, MatT, OptionT, ParamDictT, ModelBinT>::forward(const MatT& bottom_blob, MatT& top_blob,
                                                                           const OptionT& opt) const {
    if (this->top == 0 && this->bottom == 0 && this->left == 0 && this->right == 0 && this->front == 0 &&
        this->behind == 0) {
        top_blob = bottom_blob;
        return 0;
    }

    const std::int32_t w = bottom_blob.w;
    const std::int32_t h = bottom_blob.h;
    const std::int32_t d = bottom_blob.d;
    const std::int32_t channels = bottom_blob.c;
    const std::int32_t dims = bottom_blob.dims;
    const std::size_t elemsize = bottom_blob.elemsize;
    const std::int32_t outw = w + this->left + this->right;

    if (dims == 1) {
        top_blob.create(outw, elemsize, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }
        detail::pad_channel<float>(bottom_blob, top_blob, w, 1, 0, this->left, outw, 1, this->type, this->value);
        return 0;
    }

    const std::int32_t outh = h + this->top + this->bottom;

    if (dims == 2) {
        top_blob.create(outw, outh, elemsize, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }
        detail::pad_channel<float>(bottom_blob, top_blob, w, h, this->top, this->left, outw, outh, this->type,
                                   this->value);
        return 0;
    }

    if (dims == 3) {
        const std::int32_t outc = channels + this->front + this->behind;
        top_blob.create(outw, outh, outc, elemsize, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }
#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < outc; ++q) {
            MatT borderm = top_blob.channel(q);
            const float pad_value = this->per_channel_pad_data_size > 0 ? this->per_channel_pad_data[q] : this->value;
            const bool channel_pad = (q < this->front) || (q >= channels + this->front);
            if (channel_pad && this->type == 0) {
                borderm.fill(pad_value);
                continue;
            }
            std::int32_t qs = q - this->front;
            if (this->type == 1) {
                qs = qs <= 0 ? 0 : qs;
                qs = qs >= channels - 1 ? channels - 1 : qs;
            } else if (this->type == 2) {
                qs = std::abs(qs);
                qs = (channels - 1) - std::abs(qs - (channels - 1));
            }
            detail::pad_channel<float>(bottom_blob.channel(qs), borderm, w, h, this->top, this->left, outw, outh,
                                       this->type, pad_value);
        }
        return 0;
    }

    // dims == 4
    const std::int32_t outd = d + this->front + this->behind;
    top_blob.create(outw, outh, outd, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty()) {
        return -100;
    }
#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t q = 0; q < channels; ++q) {
        const float pad_value = this->per_channel_pad_data_size > 0 ? this->per_channel_pad_data[q] : this->value;
        for (std::int32_t z = 0; z < outd; ++z) {
            MatT borderm = top_blob.channel(q).depth(z);
            const bool depth_pad = (z < this->front) || (z >= d + this->front);
            if (depth_pad && this->type == 0) {
                borderm.fill(pad_value);
                continue;
            }
            std::int32_t zs = z - this->front;
            if (this->type == 1) {
                zs = zs <= 0 ? 0 : zs;
                zs = zs >= d - 1 ? d - 1 : zs;
            } else if (this->type == 2) {
                zs = std::abs(zs);
                zs = (d - 1) - std::abs(zs - (d - 1));
            }
            detail::pad_channel<float>(bottom_blob.channel(q).depth(zs), borderm, w, h, this->top, this->left, outw,
                                       outh, this->type, pad_value);
        }
    }
    return 0;
}

}  // namespace mlayer

#endif  // MLAYER_PADDING_HPP
