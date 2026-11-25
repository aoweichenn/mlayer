// Copyright 2021 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_COPYTO_HPP
#define MLAYER_COPYTO_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mlayer {

// Copy `src` into a copy of `self` at the given offsets. The expression form
// (starts/axes) is not needed by the supported models and returns -1.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class CopyTo : public LayerT {
public:
    CopyTo();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override;

public:
    std::int32_t woffset;
    std::int32_t hoffset;
    std::int32_t doffset;
    std::int32_t coffset;
    MatT starts;
    MatT axes;
};

#define MLAYER_COPYTO_TEMPLATE template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>

MLAYER_COPYTO_TEMPLATE
CopyTo<LayerT, MatT, OptionT, ParamDictT>::CopyTo() {
    this->one_blob_only = false;
    this->support_inplace = false;
    this->woffset = 0;
    this->hoffset = 0;
    this->doffset = 0;
    this->coffset = 0;
}

MLAYER_COPYTO_TEMPLATE
std::int32_t CopyTo<LayerT, MatT, OptionT, ParamDictT>::load_param(const ParamDictT& pd) {
    this->woffset = pd.get(0, 0);
    this->hoffset = pd.get(1, 0);
    this->doffset = pd.get(13, 0);
    this->coffset = pd.get(2, 0);
    this->starts = pd.get(9, MatT());
    this->axes = pd.get(11, MatT());
    return 0;
}

MLAYER_COPYTO_TEMPLATE
std::int32_t CopyTo<LayerT, MatT, OptionT, ParamDictT>::forward(const std::vector<MatT>& bottom_blobs,
                                                               std::vector<MatT>& top_blobs,
                                                               const OptionT& opt) const {
    if (bottom_blobs.size() < 2 || top_blobs.empty()) {
        return -1;
    }
    const MatT& self_blob = bottom_blobs[0];
    const MatT& src_blob = bottom_blobs[1];
    MatT& top_blob = top_blobs[0];

    const std::int32_t w = self_blob.w;
    const std::int32_t h = self_blob.h;
    const std::int32_t d = self_blob.d;
    const std::int32_t channels = self_blob.c;
    const std::int32_t dims = self_blob.dims;

    if (!this->starts.empty() || !this->axes.empty()) {
        return -1;
    }
    if (src_blob.dims == dims && src_blob.w == w && src_blob.h == h && src_blob.d == d &&
        src_blob.c == channels) {
        top_blob = src_blob;
        return 0;
    }

    top_blob = self_blob.clone(opt.blob_allocator);
    if (top_blob.empty()) {
        return -100;
    }

    const std::int32_t sw = src_blob.w;
    const std::int32_t sh = src_blob.h;
    const std::int32_t sd = src_blob.d;
    const std::int32_t sc = src_blob.c;
    const std::size_t elemsize = src_blob.elemsize;

    for (std::int32_t q = 0; q < sc; ++q) {
        const std::int32_t tq = q + this->coffset;
        if (tq < 0 || tq >= channels) {
            continue;
        }
        for (std::int32_t z = 0; z < (dims == 4 ? sd : 1); ++z) {
            const std::int32_t tz = z + this->doffset;
            if (dims == 4 && (tz < 0 || tz >= d)) {
                continue;
            }
            for (std::int32_t y = 0; y < sh; ++y) {
                const std::int32_t ty = y + this->hoffset;
                if (ty < 0 || ty >= h) {
                    continue;
                }
                const float* srow = dims == 4 ? src_blob.channel(q).depth(z).row(y) : src_blob.channel(q).row(y);
                float* drow = dims == 4 ? top_blob.channel(tq).depth(tz).row(ty) : top_blob.channel(tq).row(ty);
                for (std::int32_t x = 0; x < sw; ++x) {
                    const std::int32_t tx = x + this->woffset;
                    if (tx < 0 || tx >= w) {
                        continue;
                    }
                    if (elemsize == 4u) {
                        drow[tx] = srow[x];
                    } else {
                        std::memcpy(static_cast<std::uint8_t*>(static_cast<void*>(drow)) + tx * elemsize,
                                    static_cast<const std::uint8_t*>(static_cast<const void*>(srow)) + x * elemsize,
                                    elemsize);
                    }
                }
            }
        }
    }
    return 0;
}

}  // namespace mlayer

#endif  // MLAYER_COPYTO_HPP
