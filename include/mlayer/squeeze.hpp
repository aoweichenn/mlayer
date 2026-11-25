// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_SQUEEZE_HPP
#define MLAYER_SQUEEZE_HPP

#include <cstdint>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate = false>
class Squeeze : public LayerT {
public:
    Squeeze() {
        this->one_blob_only = true;
        this->support_inplace = false;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->squeeze_w = pd.get(0, 0);
        this->squeeze_h = pd.get(1, 0);
        this->squeeze_d = pd.get(11, 0);
        this->squeeze_c = pd.get(2, 0);
        this->axes = pd.get(3, MatT());

        if constexpr (Validate) {
            if (this->axes.w > 4) {
                return -1;
            }
            const auto* axes_ptr = static_cast<const std::int32_t*>(this->axes);
            for (std::int32_t i = 0; i < this->axes.w; ++i) {
                if (axes_ptr[i] < -4 || axes_ptr[i] > 3) {
                    return -1;
                }
            }
        }
        return 0;
    }

    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override {
        const std::int32_t w = bottom_blob.w;
        const std::int32_t h = bottom_blob.h;
        const std::int32_t d = bottom_blob.d;
        const std::int32_t channels = bottom_blob.c;
        const std::int32_t dims = bottom_blob.dims;

        bool squeeze_w = false;
        bool squeeze_h = false;
        bool squeeze_d = false;
        bool squeeze_c = false;

        if (this->axes.empty()) {
            squeeze_w = w == 1 && this->squeeze_w != 0;
            squeeze_h = h == 1 && this->squeeze_h != 0;
            squeeze_d = d == 1 && this->squeeze_d != 0;
            squeeze_c = channels == 1 && this->squeeze_c != 0;
        } else {
            const auto* axes_ptr = static_cast<const std::int32_t*>(this->axes);
            for (std::int32_t i = 0; i < this->axes.w; ++i) {
                std::int32_t axis = axes_ptr[i];
                if (axis < 0) {
                    axis = dims + axis;
                }

                if (dims == 1 && axis == 0) {
                    squeeze_w = w == 1;
                }
                if (dims == 2 && axis == 0) {
                    squeeze_h = h == 1;
                }
                if (dims == 2 && axis == 1) {
                    squeeze_w = w == 1;
                }
                if (dims == 3 && axis == 0) {
                    squeeze_c = channels == 1;
                }
                if (dims == 3 && axis == 1) {
                    squeeze_h = h == 1;
                }
                if (dims == 3 && axis == 2) {
                    squeeze_w = w == 1;
                }
                if (dims == 4 && axis == 0) {
                    squeeze_c = channels == 1;
                }
                if (dims == 4 && axis == 1) {
                    squeeze_d = d == 1;
                }
                if (dims == 4 && axis == 2) {
                    squeeze_h = h == 1;
                }
                if (dims == 4 && axis == 3) {
                    squeeze_w = w == 1;
                }
            }
        }

        top_blob = bottom_blob;

        if (dims == 1) {
            if (squeeze_w) {
                top_blob = bottom_blob.reshape(1, opt.blob_allocator);
            }
        }
        if (dims == 2) {
            if (squeeze_w && squeeze_h) {
                top_blob = bottom_blob.reshape(1, opt.blob_allocator);
            } else if (squeeze_w) {
                top_blob = bottom_blob.reshape(h, opt.blob_allocator);
            } else if (squeeze_h) {
                top_blob = bottom_blob.reshape(w, opt.blob_allocator);
            }
        }
        if (dims == 3) {
            if (squeeze_w && squeeze_h && squeeze_c) {
                top_blob = bottom_blob.reshape(1, opt.blob_allocator);
            } else if (squeeze_w && squeeze_h) {
                top_blob = bottom_blob.reshape(channels, opt.blob_allocator);
            } else if (squeeze_h && squeeze_c) {
                top_blob = bottom_blob.reshape(w, opt.blob_allocator);
            } else if (squeeze_w && squeeze_c) {
                top_blob = bottom_blob.reshape(h, opt.blob_allocator);
            } else if (squeeze_w) {
                top_blob = bottom_blob.reshape(h, channels, opt.blob_allocator);
            } else if (squeeze_h) {
                top_blob = bottom_blob.reshape(w, channels, opt.blob_allocator);
            } else if (squeeze_c) {
                top_blob = bottom_blob.reshape(w, h, opt.blob_allocator);
            }
        }
        if (dims == 4) {
            if (squeeze_w && squeeze_h && squeeze_d && squeeze_c) {
                top_blob = bottom_blob.reshape(1, opt.blob_allocator);
            } else if (squeeze_w && squeeze_h && squeeze_d) {
                top_blob = bottom_blob.reshape(channels, opt.blob_allocator);
            } else if (squeeze_h && squeeze_d && squeeze_c) {
                top_blob = bottom_blob.reshape(w, opt.blob_allocator);
            } else if (squeeze_w && squeeze_d && squeeze_c) {
                top_blob = bottom_blob.reshape(h, opt.blob_allocator);
            } else if (squeeze_w && squeeze_h && squeeze_c) {
                top_blob = bottom_blob.reshape(d, opt.blob_allocator);
            } else if (squeeze_w && squeeze_h) {
                top_blob = bottom_blob.reshape(d, channels, opt.blob_allocator);
            } else if (squeeze_w && squeeze_d) {
                top_blob = bottom_blob.reshape(h, channels, opt.blob_allocator);
            } else if (squeeze_h && squeeze_d) {
                top_blob = bottom_blob.reshape(w, channels, opt.blob_allocator);
            } else if (squeeze_w && squeeze_c) {
                top_blob = bottom_blob.reshape(h, d, opt.blob_allocator);
            } else if (squeeze_h && squeeze_c) {
                top_blob = bottom_blob.reshape(w, d, opt.blob_allocator);
            } else if (squeeze_d && squeeze_c) {
                top_blob = bottom_blob.reshape(w, h, opt.blob_allocator);
            } else if (squeeze_w) {
                top_blob = bottom_blob.reshape(h, d, channels, opt.blob_allocator);
            } else if (squeeze_h) {
                top_blob = bottom_blob.reshape(w, d, channels, opt.blob_allocator);
            } else if (squeeze_d) {
                top_blob = bottom_blob.reshape(w, h, channels, opt.blob_allocator);
            } else if (squeeze_c) {
                top_blob = bottom_blob.reshape(w, h, d, opt.blob_allocator);
            }
        }

        return top_blob.empty() ? -100 : 0;
    }

public:
    std::int32_t squeeze_w = 0;
    std::int32_t squeeze_h = 0;
    std::int32_t squeeze_d = 0;
    std::int32_t squeeze_c = 0;
    MatT axes;
};

}  // namespace mlayer

#endif  // MLAYER_SQUEEZE_HPP
