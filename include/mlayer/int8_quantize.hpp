// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_INT8_QUANTIZE_HPP
#define MLAYER_INT8_QUANTIZE_HPP

#include <cmath>
#include <cstdint>

namespace mlayer {
namespace detail {

// ncnn float2int8: round half away from zero, clamp to [-127, 127].
inline std::int8_t quantize_int8_scalar(float value, float scale) {
    const int int32 = static_cast<int>(std::round(value * scale));
    if (int32 > 127) {
        return 127;
    }
    if (int32 < -127) {
        return -127;
    }
    return static_cast<std::int8_t>(int32);
}

inline void quantize_row_int8(const float* src, std::int8_t* dst, std::int32_t size, float scale) {
    for (std::int32_t i = 0; i < size; ++i) {
        dst[i] = quantize_int8_scalar(src[i], scale);
    }
}

inline void fill_int8(std::int8_t* dst, std::int32_t size, std::int8_t value) {
    for (std::int32_t i = 0; i < size; ++i) {
        dst[i] = value;
    }
}

}  // namespace detail
}  // namespace mlayer

#endif  // MLAYER_INT8_QUANTIZE_HPP
