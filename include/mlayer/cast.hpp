// Copyright 2021 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_CAST_HPP
#define MLAYER_CAST_HPP

#include <cmath>
#include <cstdint>
#include <cstring>

namespace mlayer {
namespace detail {

// IEEE half <-> float conversions (round to nearest even for f32 -> f16).
inline float cast_half_to_float(std::uint16_t h) {
    const std::uint32_t sign = (h & 0x8000u) << 16;
    const std::uint32_t exp = (h >> 10) & 0x1Fu;
    const std::uint32_t mant = h & 0x3FFu;
    std::uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            std::uint32_t e = 127 - 15 + 1;
            std::uint32_t m = mant;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                --e;
            }
            bits = sign | (e << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f = 0.0F;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline std::uint16_t cast_float_to_half(float f) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    std::int32_t exp = static_cast<std::int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t mant = bits & 0x7FFFFFu;
    if (((bits >> 23) & 0xFFu) == 0xFFu) {
        return static_cast<std::uint16_t>(sign | 0x7C00u | (mant != 0 ? 0x200u : 0u));
    }
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        mant |= 0x800000u;
        const std::uint32_t shift = static_cast<std::uint32_t>(14 - exp);
        const std::uint32_t m = mant >> shift;
        const std::uint32_t round = (mant >> (shift - 1)) & 1u;
        return static_cast<std::uint16_t>(sign | (m + round));
    }
    if (exp >= 0x1F) {
        return static_cast<std::uint16_t>(sign | 0x7C00u);
    }
    const std::uint32_t round = (mant >> 12) & 1u;
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13) + round);
}

}  // namespace detail

// Storage type cast: 1 = fp32, 2 = fp16, 3 = int8, 4 = bf16 (ncnn ids).
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class Cast : public LayerT {
public:
    Cast();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

private:
    static std::size_t elemsize_of(std::int32_t type) {
        return type == 1 ? 4u : (type == 3 ? 1u : 2u);
    }

public:
    std::int32_t type_from;
    std::int32_t type_to;
};

#define MLAYER_CAST_TEMPLATE template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>

MLAYER_CAST_TEMPLATE
Cast<LayerT, MatT, OptionT, ParamDictT>::Cast() {
    this->one_blob_only = true;
    this->support_inplace = false;
    this->type_from = 1;
    this->type_to = 1;
}

MLAYER_CAST_TEMPLATE
std::int32_t Cast<LayerT, MatT, OptionT, ParamDictT>::load_param(const ParamDictT& pd) {
    this->type_from = pd.get(0, 1);
    this->type_to = pd.get(1, 1);
    return 0;
}

MLAYER_CAST_TEMPLATE
std::int32_t Cast<LayerT, MatT, OptionT, ParamDictT>::forward(const MatT& bottom_blob, MatT& top_blob,
                                                             const OptionT& opt) const {
    if (this->type_from == this->type_to) {
        top_blob = bottom_blob;
        return 0;
    }

    const std::int32_t dims = bottom_blob.dims;
    const std::int32_t w = bottom_blob.w;
    const std::int32_t h = bottom_blob.h;
    const std::int32_t d = bottom_blob.d;
    const std::int32_t channels = bottom_blob.c;

    if (this->type_to != 1 && this->type_to != 2 && this->type_to != 3 && this->type_to != 4) {
        return -1;
    }
    if (this->type_from == 3 && this->type_to != 1) {
        return -1;
    }
    if (this->type_from != 1 && this->type_from != 3) {
        return -1;
    }

    const std::size_t out_elemsize = elemsize_of(this->type_to);
    if (dims == 1) {
        top_blob.create(w, out_elemsize, opt.blob_allocator);
    } else if (dims == 2) {
        top_blob.create(w, h, out_elemsize, opt.blob_allocator);
    } else if (dims == 3) {
        top_blob.create(w, h, channels, out_elemsize, opt.blob_allocator);
    } else if (dims == 4) {
        top_blob.create(w, h, d, channels, out_elemsize, opt.blob_allocator);
    } else {
        return -1;
    }
    if (top_blob.empty()) {
        return -100;
    }

    const std::int32_t plane = w * h * d;
    const std::int32_t groups = dims >= 3 ? channels : 1;
    if (this->type_from == 3) {
#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < groups; ++q) {
            const auto* src = dims >= 2 ? static_cast<const std::int8_t*>(bottom_blob.channel(q))
                                        : static_cast<const std::int8_t*>(bottom_blob.data);
            auto* dst = dims >= 2 ? static_cast<float*>(top_blob.channel(q))
                                  : static_cast<float*>(top_blob.data);
            for (std::int32_t i = 0; i < plane; ++i) {
                dst[i] = static_cast<float>(src[i]);
            }
        }
        return 0;
    }
    if (this->type_to == 3) {
        // ncnn round-half-away-from-zero, clamped to [-128, 127].
#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < groups; ++q) {
            const auto* src = dims >= 2 ? static_cast<const float*>(bottom_blob.channel(q))
                                        : static_cast<const float*>(bottom_blob.data);
            auto* dst = dims >= 2 ? static_cast<std::int8_t*>(top_blob.channel(q))
                                  : static_cast<std::int8_t*>(top_blob.data);
            for (std::int32_t i = 0; i < plane; ++i) {
                float tmp = src[i] >= 0.0F ? src[i] + 0.5F : src[i] - 0.5F;
                if (tmp > 127.0F) {
                    tmp = 127.0F;
                }
                if (tmp < -128.0F) {
                    tmp = -128.0F;
                }
                dst[i] = static_cast<std::int8_t>(tmp);
            }
        }
        return 0;
    }
#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t q = 0; q < groups; ++q) {
        const auto* src = dims >= 2 ? static_cast<const float*>(bottom_blob.channel(q))
                                    : static_cast<const float*>(bottom_blob.data);
        auto* dst = dims >= 2 ? static_cast<std::uint16_t*>(top_blob.channel(q))
                              : static_cast<std::uint16_t*>(top_blob.data);
        for (std::int32_t i = 0; i < plane; ++i) {
            if (this->type_to == 2) {
                dst[i] = detail::cast_float_to_half(src[i]);
            } else {
                // bfloat16: truncate the mantissa
                std::uint32_t bits = 0;
                std::memcpy(&bits, &src[i], sizeof(bits));
                dst[i] = static_cast<std::uint16_t>(bits >> 16);
            }
        }
    }
    return 0;
}

}  // namespace mlayer

#endif  // MLAYER_CAST_HPP
