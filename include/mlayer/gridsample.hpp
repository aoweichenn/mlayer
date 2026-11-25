// Copyright 2024 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_GRIDSAMPLE_HPP
#define MLAYER_GRIDSAMPLE_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlayer {
namespace detail {

inline float grid_unormalize(std::int32_t w, float coord, std::int32_t align_corner) {
    return align_corner != 0 ? (coord + 1.0F) / 2.0F * static_cast<float>(w - 1)
                             : ((coord + 1.0F) * static_cast<float>(w) - 1.0F) / 2.0F;
}

inline float grid_border_coord(float x, float border) {
    return std::min(border, std::max(x, 0.0F));
}

inline float grid_reflect_coord(float x, std::int32_t high) {
    x = std::fabs(x);
    x = static_cast<float>(high) - std::fabs(x - static_cast<float>(high));
    return x;
}

inline float grid_compute_coord(float sx, std::int32_t w, std::int32_t padding_mode, std::int32_t align_corner) {
    if (padding_mode == 2) {
        sx = grid_border_coord(sx, static_cast<float>(w - 1));
    } else if (padding_mode == 3) {
        if (align_corner != 0) {
            sx = grid_reflect_coord(sx, w - 1);
        } else {
            sx = grid_reflect_coord(sx + 0.5F, w) - 0.5F;
            sx = grid_border_coord(sx, static_cast<float>(w - 1));
        }
    }
    return sx;
}

}  // namespace detail

// GridSample (2D): bilinear or nearest sampling of a feature map at a grid of
// normalised coordinates, with zeros / border / reflection padding modes.
// permute_fusion=1 uses the natural (w=outw, h=outh, c=2) grid layout emitted by
// pnnx; permute_fusion=0 expects one channel per output row with packed pairs.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate = false>
class GridSample : public LayerT {
public:
    GridSample();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override;

public:
    // 1 = bilinear, 2 = nearest (ncnn ids; bicubic is not implemented)
    std::int32_t sample_type;
    std::int32_t padding_mode;
    std::int32_t align_corner;
    std::int32_t permute_fusion;
};

#define MLAYER_GRIDSAMPLE_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate>

MLAYER_GRIDSAMPLE_TEMPLATE
GridSample<LayerT, MatT, OptionT, ParamDictT, Validate>::GridSample() {
    this->one_blob_only = false;
    this->support_inplace = false;
    this->sample_type = 1;
    this->padding_mode = 1;
    this->align_corner = 0;
    this->permute_fusion = 0;
}

MLAYER_GRIDSAMPLE_TEMPLATE
std::int32_t GridSample<LayerT, MatT, OptionT, ParamDictT, Validate>::load_param(const ParamDictT& pd) {
    this->sample_type = pd.get(0, 1);
    this->padding_mode = pd.get(1, 1);
    this->align_corner = pd.get(2, 0);
    this->permute_fusion = pd.get(3, 0);
    return 0;
}

MLAYER_GRIDSAMPLE_TEMPLATE
std::int32_t GridSample<LayerT, MatT, OptionT, ParamDictT, Validate>::forward(const std::vector<MatT>& bottom_blobs,
                                                                             std::vector<MatT>& top_blobs,
                                                                             const OptionT& opt) const {
    if (bottom_blobs.size() < 2 || top_blobs.empty()) {
        return -1;
    }
    if (this->sample_type == 3) {
        // bicubic is not implemented
        return -1;
    }
    const MatT& bottom_blob = bottom_blobs[0];
    const MatT& grid = bottom_blobs[1];
    const std::int32_t w = bottom_blob.w;
    const std::int32_t h = bottom_blob.h;
    const std::int32_t channels = bottom_blob.c;
    const std::size_t elemsize = bottom_blob.elemsize;

    if (bottom_blob.dims != 3) {
        // 2D feature maps only
        return -1;
    }

    const std::int32_t outw = this->permute_fusion == 0 ? grid.h : grid.w;
    const std::int32_t outh = this->permute_fusion == 0 ? grid.c : grid.h;

    MatT& top_blob = top_blobs[0];
    top_blob.create(outw, outh, channels, elemsize, opt.blob_allocator);
    MatT offset_blob;
    offset_blob.create(outw, outh, 2, elemsize, opt.workspace_allocator);
    if (top_blob.empty() || offset_blob.empty()) {
        return -100;
    }

    // Unnormalise the grid once and share it across channels.
    for (std::int32_t y = 0; y < outh; ++y) {
        float* offsetptr_x = offset_blob.channel(0).row(y);
        float* offsetptr_y = offset_blob.channel(1).row(y);
        const float* gridptr = this->permute_fusion == 0 ? grid.channel(y) : grid.channel(0).row(y);
        const float* gridptr_y = this->permute_fusion == 0 ? nullptr : grid.channel(1).row(y);
        for (std::int32_t x = 0; x < outw; ++x) {
            const float sample_x = this->permute_fusion == 0 ? gridptr[2 * x] : gridptr[x];
            const float sample_y = this->permute_fusion == 0 ? gridptr[2 * x + 1] : gridptr_y[x];
            offsetptr_x[x] = detail::grid_unormalize(w, sample_x, this->align_corner);
            offsetptr_y[x] = detail::grid_unormalize(h, sample_y, this->align_corner);
        }
    }

    const auto sample = [&](const MatT& image, float sx, float sy) -> float {
        if (this->sample_type == 2) {
            sx = detail::grid_compute_coord(sx, w, this->padding_mode, this->align_corner);
            sy = detail::grid_compute_coord(sy, h, this->padding_mode, this->align_corner);
            const std::int32_t x0 = static_cast<std::int32_t>(std::floor(sx + 0.5F));
            const std::int32_t y0 = static_cast<std::int32_t>(std::floor(sy + 0.5F));
            if (x0 < 0 || y0 < 0 || x0 >= w || y0 >= h) {
                return 0.0F;
            }
            return image.row(y0)[x0];
        }

        sx = detail::grid_compute_coord(sx, w, this->padding_mode, this->align_corner);
        sy = detail::grid_compute_coord(sy, h, this->padding_mode, this->align_corner);
        const std::int32_t x0 = static_cast<std::int32_t>(std::floor(sx));
        const std::int32_t y0 = static_cast<std::int32_t>(std::floor(sy));
        const std::int32_t x1 = x0 + 1;
        const std::int32_t y1 = y0 + 1;
        const auto value = [&](std::int32_t vx, std::int32_t vy) -> float {
            return (vx < 0 || vy < 0 || vx >= w || vy >= h) ? 0.0F : image.row(vy)[vx];
        };
        const float v00 = value(x0, y0);
        const float v01 = value(x1, y0);
        const float v10 = value(x0, y1);
        const float v11 = value(x1, y1);
        const float alpha = sx - static_cast<float>(x0);
        const float beta = sy - static_cast<float>(y0);
        const float v0 = v00 * (1.0F - alpha) + v01 * alpha;
        const float v1 = v10 * (1.0F - alpha) + v11 * alpha;
        return v0 * (1.0F - beta) + v1 * beta;
    };

#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t q = 0; q < channels; ++q) {
        const MatT image = bottom_blob.channel(q);
        const float* offsetptr_x = offset_blob.channel(0);
        const float* offsetptr_y = offset_blob.channel(1);
        MatT out_c = top_blob.channel(q);
        for (std::int32_t y = 0; y < outh; ++y) {
            float* outptr = out_c.row(y);
            for (std::int32_t x = 0; x < outw; ++x) {
                outptr[x] = sample(image, offsetptr_x[y * outw + x], offsetptr_y[y * outw + x]);
            }
        }
    }
    return 0;
}

}  // namespace mlayer

#endif  // MLAYER_GRIDSAMPLE_HPP
