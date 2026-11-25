// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_CROP_HPP
#define MLAYER_CROP_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate = false>
class Crop : public LayerT {
public:
    Crop() {
        // Both the single-input and reference-input forms go through the
        // vector forward overload.
        this->one_blob_only = false;
        this->support_inplace = false;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->woffset = pd.get(0, 0);
        this->hoffset = pd.get(1, 0);
        this->doffset = pd.get(13, 0);
        this->coffset = pd.get(2, 0);
        // -233 means "take the whole remaining extent", as in ncnn.
        this->outw = pd.get(3, -233);
        this->outh = pd.get(4, -233);
        this->outd = pd.get(14, -233);
        this->outc = pd.get(5, -233);
        this->woffset2 = pd.get(6, 0);
        this->hoffset2 = pd.get(7, 0);
        this->doffset2 = pd.get(15, 0);
        this->coffset2 = pd.get(8, 0);

        this->starts = pd.get(9, MatT());
        this->ends = pd.get(10, MatT());
        this->axes = pd.get(11, MatT());

        if constexpr (Validate) {
            if (this->axes.w > 4) {
                return -1;
            }
        }

        return 0;
    }

    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override {
        if (bottom_blobs.empty() || top_blobs.empty()) {
            return -1;
        }
        const MatT& bottom = bottom_blobs[0];

        std::int32_t x0 = this->woffset;
        std::int32_t y0 = this->hoffset;
        std::int32_t z0 = this->doffset;
        std::int32_t q0 = this->coffset;
        std::int32_t ow = this->outw;
        std::int32_t oh = this->outh;
        std::int32_t od = this->outd > 0 ? this->outd : 1;
        std::int32_t oc = this->outc;

        if (bottom_blobs.size() > 1) {
            const MatT& reference = bottom_blobs[1];
            const std::int32_t dims = bottom.dims;
            ow = reference.w;
            oh = reference.h;
            od = reference.d > 0 ? reference.d : 1;
            oc = reference.dims >= 3 ? reference.c : bottom.c;
            if (dims == 1) {
                ow = reference.w;
            }
            if (dims == 2) {
                oh = reference.h;
            }
        } else {
            const std::int32_t dims = bottom.dims;
            if (dims == 1) {
                ow = bottom.w - this->woffset - this->woffset2;
                if (this->outw != -233) {
                    ow = std::min(this->outw, ow);
                }
            }
            if (dims == 2) {
                ow = bottom.w - this->woffset - this->woffset2;
                if (this->outw != -233) {
                    ow = std::min(this->outw, ow);
                }
                oh = bottom.h - this->hoffset - this->hoffset2;
                if (this->outh != -233) {
                    oh = std::min(this->outh, oh);
                }
            }
            if (dims == 3) {
                ow = bottom.w - this->woffset - this->woffset2;
                if (this->outw != -233) {
                    ow = std::min(this->outw, ow);
                }
                oh = bottom.h - this->hoffset - this->hoffset2;
                if (this->outh != -233) {
                    oh = std::min(this->outh, oh);
                }
                oc = bottom.c - this->coffset - this->coffset2;
                if (this->outc != -233) {
                    oc = std::min(this->outc, oc);
                }
            }
            if (dims == 4) {
                ow = bottom.w - this->woffset - this->woffset2;
                if (this->outw != -233) {
                    ow = std::min(this->outw, ow);
                }
                oh = bottom.h - this->hoffset - this->hoffset2;
                if (this->outh != -233) {
                    oh = std::min(this->outh, oh);
                }
                od = bottom.d - this->doffset - this->doffset2;
                if (this->outd != -233) {
                    od = std::min(this->outd, od);
                }
                oc = bottom.c - this->coffset - this->coffset2;
                if (this->outc != -233) {
                    oc = std::min(this->outc, oc);
                }
            }
        }

        MatT& top = top_blobs[0];
        const std::int32_t dims = bottom.dims;
        // Blobs below 3-D have implicit h/c of one; cropping a 1-D blob must not
        // fail because outh/outc were left unspecified.
        if (dims == 1) {
            oh = 1;
            oc = 1;
        } else if (dims == 2) {
            oc = 1;
        }
        if (ow <= 0 || oh <= 0 || od <= 0 || oc <= 0) {
            return -100;
        }

        if (dims == 1) {
            top.create(ow, bottom.elemsize, opt.blob_allocator);
            if (top.empty()) {
                return -100;
            }
            const auto* src = static_cast<const float*>(bottom);
            auto* dst = static_cast<float*>(top);
            for (std::int32_t x = 0; x < ow; ++x) {
                dst[x] = src[x0 + x];
            }
            return 0;
        }

        if (dims == 2) {
            top.create(ow, oh, bottom.elemsize, opt.blob_allocator);
            if (top.empty()) {
                return -100;
            }
            for (std::int32_t y = 0; y < oh; ++y) {
                const auto* src = static_cast<const float*>(bottom.row(y0 + y));
                auto* dst = static_cast<float*>(top.row(y));
                for (std::int32_t x = 0; x < ow; ++x) {
                    dst[x] = src[x0 + x];
                }
            }
            return 0;
        }

        if (dims == 3) {
            top.create(ow, oh, oc, bottom.elemsize, opt.blob_allocator);
            if (top.empty()) {
                return -100;
            }
            for (std::int32_t q = 0; q < oc; ++q) {
                for (std::int32_t y = 0; y < oh; ++y) {
                    const auto* src = static_cast<const float*>(bottom.channel(q0 + q).row(y0 + y));
                    auto* dst = static_cast<float*>(top.channel(q).row(y));
                    for (std::int32_t x = 0; x < ow; ++x) {
                        dst[x] = src[x0 + x];
                    }
                }
            }
            return 0;
        }

        top.create(ow, oh, od, oc, bottom.elemsize, opt.blob_allocator);
        if (top.empty()) {
            return -100;
        }
        for (std::int32_t q = 0; q < oc; ++q) {
            for (std::int32_t z = 0; z < od; ++z) {
                for (std::int32_t y = 0; y < oh; ++y) {
                    const auto* src =
                        static_cast<const float*>(bottom.channel(q0 + q).depth(z0 + z).row(y0 + y));
                    auto* dst = static_cast<float*>(top.channel(q).depth(z).row(y));
                    for (std::int32_t x = 0; x < ow; ++x) {
                        dst[x] = src[x0 + x];
                    }
                }
            }
        }
        return 0;
    }

public:
    std::int32_t woffset = 0;
    std::int32_t hoffset = 0;
    std::int32_t doffset = 0;
    std::int32_t coffset = 0;
    std::int32_t outw = 0;
    std::int32_t outh = 0;
    std::int32_t outd = 0;
    std::int32_t outc = 0;
    std::int32_t woffset2 = 0;
    std::int32_t hoffset2 = 0;
    std::int32_t doffset2 = 0;
    std::int32_t coffset2 = 0;
    MatT starts;
    MatT ends;
    MatT axes;
};

}  // namespace mlayer

#endif  // MLAYER_CROP_HPP
