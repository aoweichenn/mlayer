// Copyright 2019 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_TILE_HPP
#define MLAYER_TILE_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace mlayer {

// numpy-style tile: repeats the tensor along the requested axes. Supports both
// the legacy (axis, tiles) form and the (repeats array) form used by pnnx.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class Tile : public LayerT {
public:
    Tile();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

public:
    std::int32_t axis;
    std::int32_t tiles;
    MatT repeats;
};

#define MLAYER_TILE_TEMPLATE template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>

MLAYER_TILE_TEMPLATE
Tile<LayerT, MatT, OptionT, ParamDictT>::Tile() {
    this->one_blob_only = true;
    this->support_inplace = false;
    this->axis = 0;
    this->tiles = 1;
}

MLAYER_TILE_TEMPLATE
std::int32_t Tile<LayerT, MatT, OptionT, ParamDictT>::load_param(const ParamDictT& pd) {
    this->axis = pd.get(0, 0);
    this->tiles = pd.get(1, 1);
    this->repeats = pd.get(2, MatT());
    return 0;
}

MLAYER_TILE_TEMPLATE
std::int32_t Tile<LayerT, MatT, OptionT, ParamDictT>::forward(const MatT& bottom_blob, MatT& top_blob,
                                                             const OptionT& opt) const {
    const std::int32_t dims = bottom_blob.dims;
    std::int32_t repeat[4] = {1, 1, 1, 1};  // w, h, d, c

    const std::int32_t repeats_num = this->repeats.w;
    if (this->repeats.empty()) {
        if (dims == 1) {
            repeat[0] = this->tiles;
        } else if (dims == 2) {
            if (this->axis == 0) {
                repeat[1] = this->tiles;
            }
            if (this->axis == 1) {
                repeat[0] = this->tiles;
            }
        } else if (dims == 3) {
            if (this->axis == 0) {
                repeat[3] = this->tiles;
            }
            if (this->axis == 1) {
                repeat[1] = this->tiles;
            }
            if (this->axis == 2) {
                repeat[0] = this->tiles;
            }
        } else if (dims == 4) {
            if (this->axis == 0) {
                repeat[3] = this->tiles;
            }
            if (this->axis == 1) {
                repeat[2] = this->tiles;
            }
            if (this->axis == 2) {
                repeat[1] = this->tiles;
            }
            if (this->axis == 3) {
                repeat[0] = this->tiles;
            }
        }
    } else {
        const auto* r = static_cast<const std::int32_t*>(this->repeats);
        if (repeats_num == 1) {
            repeat[0] = r[0];
        } else if (repeats_num == 2) {
            repeat[1] = r[0];
            repeat[0] = r[1];
        } else if (repeats_num == 3) {
            if (dims == 4) {
                repeat[2] = r[0];
            } else {
                repeat[3] = r[0];
            }
            repeat[1] = r[1];
            repeat[0] = r[2];
        } else if (repeats_num == 4) {
            repeat[3] = r[0];
            repeat[2] = r[1];
            repeat[1] = r[2];
            repeat[0] = r[3];
        }
    }

    const std::int32_t w = bottom_blob.w;
    const std::int32_t h = bottom_blob.h;
    const std::int32_t d = bottom_blob.d;
    const std::int32_t channels = bottom_blob.c;
    const std::size_t elemsize = bottom_blob.elemsize;
    const std::int32_t outdims = std::max(dims, repeats_num);

    const std::int32_t outw = w * repeat[0];
    const std::int32_t outh = h * repeat[1];
    const std::int32_t outd = d * repeat[2];
    const std::int32_t outc = channels * repeat[3];

    if (outdims == 1) {
        top_blob.create(outw, elemsize, opt.blob_allocator);
    } else if (outdims == 2) {
        top_blob.create(outw, outh, elemsize, opt.blob_allocator);
    } else if (outdims == 3) {
        top_blob.create(outw, outh, outc, elemsize, opt.blob_allocator);
    } else {
        top_blob.create(outw, outh, outd, outc, elemsize, opt.blob_allocator);
    }
    if (top_blob.empty()) {
        return -100;
    }

    // Element-wise copy with per-axis modulo: block i comes from block i % size.
#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t oc = 0; oc < outc; ++oc) {
        const MatT src_c = bottom_blob.channel(oc % channels);
        MatT dst_c = top_blob.channel(oc);
        for (std::int32_t oz = 0; oz < (outdims >= 4 ? outd : 1); ++oz) {
            const MatT src_d = outdims >= 4 ? src_c.depth(oz % d) : src_c;
            MatT dst_d = outdims >= 4 ? dst_c.depth(oz) : dst_c;
            for (std::int32_t oy = 0; oy < (outdims >= 2 ? outh : 1); ++oy) {
                const float* srow = src_d.row(oy % h);
                float* drow = dst_d.row(oy);
                for (std::int32_t ox = 0; ox < outw; ++ox) {
                    drow[ox] = srow[ox % w];
                }
            }
        }
    }

    return 0;
}

}  // namespace mlayer

#endif  // MLAYER_TILE_HPP
