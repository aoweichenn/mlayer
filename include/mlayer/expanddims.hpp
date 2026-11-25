// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_EXPANDDIMS_HPP
#define MLAYER_EXPANDDIMS_HPP

#include <cstdint>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate = false>
class ExpandDims : public LayerT {
public:
    ExpandDims() {
        this->one_blob_only = true;
        this->support_inplace = false;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->axes = pd.get(3, MatT());
        if constexpr (Validate) {
            if (this->axes.w > 4) {
                return -1;
            }
            const auto* axes_ptr = static_cast<const std::int32_t*>(this->axes);
            for (std::int32_t i = 0; i < this->axes.w; ++i) {
                const std::int32_t axis = axes_ptr[i];
                if (axis < -4 || axis > 3) {
                    return -1;
                }
            }
        }
        return 0;
    }

    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override {
        const std::int32_t w = bottom_blob.w;
        const std::int32_t h = bottom_blob.h;
        const std::int32_t channels = bottom_blob.c;
        const std::int32_t dims = bottom_blob.dims;

        const std::int32_t outdims = dims + this->axes.w;

        bool expand_w = false;
        bool expand_h = false;
        bool expand_d = false;
        bool expand_c = false;

        const auto* axes_ptr = static_cast<const std::int32_t*>(this->axes);
        for (std::int32_t i = 0; i < this->axes.w; ++i) {
            std::int32_t axis = axes_ptr[i];
            if (axis < 0) {
                axis = outdims + axis;
            }

            if (outdims == 2) {
                if (axis == 0) {
                    expand_h = true;
                }
                if (axis == 1) {
                    expand_w = true;
                }
            }
            if (outdims == 3) {
                if (axis == 0) {
                    expand_c = true;
                }
                if (axis == 1) {
                    expand_h = true;
                }
                if (axis == 2) {
                    expand_w = true;
                }
            }
            if (outdims == 4) {
                if (axis == 0) {
                    expand_c = true;
                }
                if (axis == 1) {
                    expand_d = true;
                }
                if (axis == 2) {
                    expand_h = true;
                }
                if (axis == 3) {
                    expand_w = true;
                }
            }
        }

        top_blob = bottom_blob;

        if (outdims == 2) {
            if (expand_w) {
                top_blob = bottom_blob.reshape(1, w, opt.blob_allocator);
            } else if (expand_h) {
                top_blob = bottom_blob.reshape(w, 1, opt.blob_allocator);
            }
        }
        if (outdims == 3) {
            if (expand_w && expand_h) {
                top_blob = bottom_blob.reshape(1, 1, w, opt.blob_allocator);
            } else if (expand_w && expand_c) {
                top_blob = bottom_blob.reshape(1, w, 1, opt.blob_allocator);
            } else if (expand_h && expand_c) {
                top_blob = bottom_blob.reshape(w, 1, 1, opt.blob_allocator);
            } else if (expand_w) {
                top_blob = bottom_blob.reshape(1, w, h, opt.blob_allocator);
            } else if (expand_h) {
                top_blob = bottom_blob.reshape(w, 1, h, opt.blob_allocator);
            } else if (expand_c) {
                top_blob = bottom_blob.reshape(w, h, 1, opt.blob_allocator);
            }
        }
        if (outdims == 4) {
            if (expand_w && expand_h && expand_d) {
                top_blob = bottom_blob.reshape(1, 1, 1, w, opt.blob_allocator);
            } else if (expand_w && expand_h && expand_c) {
                top_blob = bottom_blob.reshape(1, 1, w, 1, opt.blob_allocator);
            } else if (expand_w && expand_d && expand_c) {
                top_blob = bottom_blob.reshape(1, w, 1, 1, opt.blob_allocator);
            } else if (expand_h && expand_d && expand_c) {
                top_blob = bottom_blob.reshape(w, 1, 1, 1, opt.blob_allocator);
            } else if (expand_w && expand_h) {
                top_blob = bottom_blob.reshape(1, 1, w, h, opt.blob_allocator);
            } else if (expand_w && expand_c) {
                top_blob = bottom_blob.reshape(1, w, h, 1, opt.blob_allocator);
            } else if (expand_d && expand_c) {
                top_blob = bottom_blob.reshape(w, h, 1, 1, opt.blob_allocator);
            } else if (expand_w && expand_d) {
                top_blob = bottom_blob.reshape(1, w, 1, h, opt.blob_allocator);
            } else if (expand_h && expand_c) {
                top_blob = bottom_blob.reshape(w, 1, h, 1, opt.blob_allocator);
            } else if (expand_h && expand_d) {
                top_blob = bottom_blob.reshape(w, 1, 1, h, opt.blob_allocator);
            } else if (expand_w) {
                top_blob = bottom_blob.reshape(1, w, h, channels, opt.blob_allocator);
            } else if (expand_h) {
                top_blob = bottom_blob.reshape(w, 1, h, channels, opt.blob_allocator);
            } else if (expand_d) {
                top_blob = bottom_blob.reshape(w, h, 1, channels, opt.blob_allocator);
            } else if (expand_c) {
                top_blob = bottom_blob.reshape(w, h, channels, 1, opt.blob_allocator);
            }
        }

        return top_blob.empty() ? -100 : 0;
    }

public:
    MatT axes;
};

}  // namespace mlayer

#endif  // MLAYER_EXPANDDIMS_HPP
