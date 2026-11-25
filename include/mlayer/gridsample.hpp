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

// Cubic convolution weights with a = -0.75 (ncnn / PyTorch bicubic).
inline void grid_cubic_coeffs(float fx, float* coeffs) {
    constexpr float kA = -0.75F;
    const float fx0 = fx + 1.0F;
    const float fx1 = fx;
    const float fx2 = 1.0F - fx;
    coeffs[0] = kA * fx0 * fx0 * fx0 - 5.0F * kA * fx0 * fx0 + 8.0F * kA * fx0 - 4.0F * kA;
    coeffs[1] = (kA + 2.0F) * fx1 * fx1 * fx1 - (kA + 3.0F) * fx1 * fx1 + 1.0F;
    coeffs[2] = (kA + 2.0F) * fx2 * fx2 * fx2 - (kA + 3.0F) * fx2 * fx2 + 1.0F;
    coeffs[3] = 1.0F - coeffs[0] - coeffs[1] - coeffs[2];
}

}  // namespace detail

// GridSample: 2D bilinear / nearest / bicubic and 3D bilinear / nearest
// sampling of a feature map at normalised grid coordinates, with zeros /
// border / reflection padding modes. permute_fusion=1 uses the natural
// (w, h[, d], coords) grid layout, 0 the channel-packed layout emitted by pnnx
// after a Permute. ncnn also rejects bicubic for 3D, so this layer does too.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate = false>
class GridSample : public LayerT {
public:
    GridSample();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override;

public:
    // 1 = bilinear/trilinear, 2 = nearest, 3 = bicubic (2D only)
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
    const MatT& bottom_blob = bottom_blobs[0];
    const MatT& grid = bottom_blobs[1];
    const std::int32_t w = bottom_blob.w;
    const std::int32_t h = bottom_blob.h;
    const std::int32_t d = bottom_blob.d;
    const std::int32_t channels = bottom_blob.c;
    const std::int32_t dims = bottom_blob.dims;
    const std::size_t elemsize = bottom_blob.elemsize;

    if (dims != 3 && dims != 4) {
        return -1;
    }
    if (dims == 4 && this->sample_type == 3) {
        // ncnn: unsupported bicubic when dims == 4
        return -1;
    }

    std::int32_t outw = 0;
    std::int32_t outh = 0;
    std::int32_t outd = 1;
    if (dims == 3) {
        outw = this->permute_fusion == 0 ? grid.h : grid.w;
        outh = this->permute_fusion == 0 ? grid.c : grid.h;
    } else {
        outw = this->permute_fusion == 0 ? grid.h : grid.w;
        outh = this->permute_fusion == 0 ? grid.d : grid.h;
        outd = this->permute_fusion == 0 ? grid.c : grid.d;
    }

    MatT& top_blob = top_blobs[0];
    if (dims == 3) {
        top_blob.create(outw, outh, channels, elemsize, opt.blob_allocator);
    } else {
        top_blob.create(outw, outh, outd, channels, elemsize, opt.blob_allocator);
    }

    const std::int32_t coords = dims == 3 ? 2 : 3;
    MatT offset_blob;
    if (dims == 3) {
        offset_blob.create(outw, outh, coords, elemsize, opt.workspace_allocator);
    } else {
        offset_blob.create(outw, outh, outd, coords, elemsize, opt.workspace_allocator);
    }
    if (top_blob.empty() || offset_blob.empty()) {
        return -100;
    }

    const std::int32_t count = outw * outh * outd;
    float* offsets[3] = {nullptr, nullptr, nullptr};
    for (std::int32_t c = 0; c < coords; ++c) {
        offsets[c] = static_cast<float*>(offset_blob.channel(c));
    }

    if (dims == 3) {
        for (std::int32_t y = 0; y < outh; ++y) {
            const float* gx = nullptr;
            const float* gy = nullptr;
            const float* gpairs = nullptr;
            if (this->permute_fusion == 0) {
                gpairs = grid.channel(y);
            } else {
                gx = static_cast<const float*>(grid.channel(0).row(y));
                gy = static_cast<const float*>(grid.channel(1).row(y));
            }
            for (std::int32_t x = 0; x < outw; ++x) {
                const float sx = this->permute_fusion == 0 ? gpairs[2 * x] : gx[x];
                const float sy = this->permute_fusion == 0 ? gpairs[2 * x + 1] : gy[x];
                const std::int32_t oi = y * outw + x;
                offsets[0][oi] = detail::grid_unormalize(w, sx, this->align_corner);
                offsets[1][oi] = detail::grid_unormalize(h, sy, this->align_corner);
            }
        }
    } else {
        for (std::int32_t z = 0; z < outd; ++z) {
            const float* gx = nullptr;
            const float* gy = nullptr;
            const float* gz = nullptr;
            const float* gtriples = nullptr;
            if (this->permute_fusion == 0) {
                gtriples = static_cast<const float*>(grid.channel(z));
            } else {
                gx = static_cast<const float*>(grid.channel(0));
                gy = static_cast<const float*>(grid.channel(1));
                gz = static_cast<const float*>(grid.channel(2));
            }
            for (std::int32_t y = 0; y < outh; ++y) {
                for (std::int32_t x = 0; x < outw; ++x) {
                    const std::int32_t src = (z * outh + y) * outw + x;
                    float sx = 0.0F;
                    float sy = 0.0F;
                    float sz = 0.0F;
                    if (this->permute_fusion == 0) {
                        sx = gtriples[3 * src];
                        sy = gtriples[3 * src + 1];
                        sz = gtriples[3 * src + 2];
                    } else {
                        sx = gx[src];
                        sy = gy[src];
                        sz = gz[src];
                    }
                    // 3D stores coordinates already unnormalised and padded.
                    offsets[0][src] =
                        detail::grid_compute_coord(detail::grid_unormalize(w, sx, this->align_corner), w,
                                                   this->padding_mode, this->align_corner);
                    offsets[1][src] =
                        detail::grid_compute_coord(detail::grid_unormalize(h, sy, this->align_corner), h,
                                                   this->padding_mode, this->align_corner);
                    offsets[2][src] =
                        detail::grid_compute_coord(detail::grid_unormalize(d, sz, this->align_corner), d,
                                                   this->padding_mode, this->align_corner);
                }
            }
        }
    }

    // In-bounds fetch used by the 3D paths (coordinates are already padded).
    const auto fetch3 = [&](const MatT& image, std::int32_t x, std::int32_t y, std::int32_t z) -> float {
        if (x < 0 || y < 0 || z < 0 || x >= w || y >= h || z >= d) {
            return 0.0F;
        }
        return image.channel(z).row(y)[x];
    };
    // Fetch with padding applied per coordinate (2D and bicubic).
    const auto fetch_padded = [&](const MatT& image, std::int32_t x, std::int32_t y) -> float {
        const float px = detail::grid_compute_coord(static_cast<float>(x), w, this->padding_mode,
                                                    this->align_corner);
        const float py = detail::grid_compute_coord(static_cast<float>(y), h, this->padding_mode,
                                                    this->align_corner);
        const auto ix = static_cast<std::int32_t>(px);
        const auto iy = static_cast<std::int32_t>(py);
        if (ix < 0 || iy < 0 || ix >= w || iy >= h) {
            return 0.0F;
        }
        return image.row(iy)[ix];
    };

#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t q = 0; q < channels; ++q) {
        const MatT image = bottom_blob.channel(q);
        float* outptr = static_cast<float*>(top_blob.channel(q));
        for (std::int32_t idx = 0; idx < count; ++idx) {
            const float sx = offsets[0][idx];
            const float sy = offsets[1][idx];
            const float sz = dims == 4 ? offsets[2][idx] : 0.0F;
            float v = 0.0F;

            if (dims == 3) {
                if (this->sample_type == 2) {
                    const float cx = detail::grid_compute_coord(sx, w, this->padding_mode, this->align_corner);
                    const float cy = detail::grid_compute_coord(sy, h, this->padding_mode, this->align_corner);
                    const auto x0 = static_cast<std::int32_t>(std::floor(cx + 0.5F));
                    const auto y0 = static_cast<std::int32_t>(std::floor(cy + 0.5F));
                    v = (x0 < 0 || y0 < 0 || x0 >= w || y0 >= h) ? 0.0F : image.row(y0)[x0];
                } else if (this->sample_type == 3) {
                    const auto x1 = static_cast<std::int32_t>(std::floor(sx));
                    const auto y1 = static_cast<std::int32_t>(std::floor(sy));
                    const std::int32_t xs[4] = {x1 - 1, x1, x1 + 1, x1 + 2};
                    const std::int32_t ys[4] = {y1 - 1, y1, y1 + 1, y1 + 2};
                    float xc[4];
                    float yc[4];
                    detail::grid_cubic_coeffs(sx - static_cast<float>(x1), xc);
                    detail::grid_cubic_coeffs(sy - static_cast<float>(y1), yc);
                    float rows[4];
                    for (std::int32_t r = 0; r < 4; ++r) {
                        const float v0 = fetch_padded(image, xs[0], ys[r]);
                        const float v1 = fetch_padded(image, xs[1], ys[r]);
                        const float v2 = fetch_padded(image, xs[2], ys[r]);
                        const float v3 = fetch_padded(image, xs[3], ys[r]);
                        rows[r] = v0 * xc[0] + v1 * xc[1] + v2 * xc[2] + v3 * xc[3];
                    }
                    v = rows[0] * yc[0] + rows[1] * yc[1] + rows[2] * yc[2] + rows[3] * yc[3];
                } else {
                    const float cx = detail::grid_compute_coord(sx, w, this->padding_mode, this->align_corner);
                    const float cy = detail::grid_compute_coord(sy, h, this->padding_mode, this->align_corner);
                    const auto x0 = static_cast<std::int32_t>(std::floor(cx));
                    const auto y0 = static_cast<std::int32_t>(std::floor(cy));
                    const std::int32_t x1 = x0 + 1;
                    const std::int32_t y1 = y0 + 1;
                    const auto value = [&](std::int32_t vx, std::int32_t vy) -> float {
                        return (vx < 0 || vy < 0 || vx >= w || vy >= h) ? 0.0F : image.row(vy)[vx];
                    };
                    const float alpha = cx - static_cast<float>(x0);
                    const float beta = cy - static_cast<float>(y0);
                    const float v0 = value(x0, y0) * (1.0F - alpha) + value(x1, y0) * alpha;
                    const float v1 = value(x0, y1) * (1.0F - alpha) + value(x1, y1) * alpha;
                    v = v0 * (1.0F - beta) + v1 * beta;
                }
            } else {
                const std::int32_t zidx = idx / (outw * outh);
                const std::int32_t rem = idx % (outw * outh);
                const std::int32_t yidx = rem / outw;
                const std::int32_t xidx = rem % outw;
                if (this->sample_type == 2) {
                    const auto x0 = static_cast<std::int32_t>(std::floor(sx + 0.5F));
                    const auto y0 = static_cast<std::int32_t>(std::floor(sy + 0.5F));
                    const auto z0 = static_cast<std::int32_t>(std::floor(sz + 0.5F));
                    v = fetch3(image, x0, y0, z0);
                } else {
                    const auto x0 = static_cast<std::int32_t>(std::floor(sx));
                    const auto y0 = static_cast<std::int32_t>(std::floor(sy));
                    const auto z0 = static_cast<std::int32_t>(std::floor(sz));
                    const std::int32_t x1 = x0 + 1;
                    const std::int32_t y1 = y0 + 1;
                    const std::int32_t z1 = z0 + 1;
                    const float v000 = fetch3(image, x0, y0, z0);
                    const float v001 = fetch3(image, x1, y0, z0);
                    const float v010 = fetch3(image, x0, y1, z0);
                    const float v011 = fetch3(image, x1, y1, z0);
                    const float v100 = fetch3(image, x0, y0, z1);
                    const float v101 = fetch3(image, x1, y0, z1);
                    const float v110 = fetch3(image, x0, y1, z1);
                    const float v111 = fetch3(image, x1, y1, z1);
                    const float alpha = sx - static_cast<float>(x0);
                    const float beta = sy - static_cast<float>(y0);
                    const float gamma = sz - static_cast<float>(z0);
                    const float v00 = v000 * (1.0F - alpha) + v001 * alpha;
                    const float v01 = v010 * (1.0F - alpha) + v011 * alpha;
                    const float v10 = v100 * (1.0F - alpha) + v101 * alpha;
                    const float v11 = v110 * (1.0F - alpha) + v111 * alpha;
                    const float r0 = v00 * (1.0F - beta) + v01 * beta;
                    const float r1 = v10 * (1.0F - beta) + v11 * beta;
                    v = r0 * (1.0F - gamma) + r1 * gamma;
                }
                (void)xidx;
                (void)yidx;
                (void)zidx;
            }
            outptr[idx] = v;
        }
    }
    return 0;
}

}  // namespace mlayer

#endif  // MLAYER_GRIDSAMPLE_HPP
