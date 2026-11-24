// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_QUANTIZE_HPP
#define MLAYER_QUANTIZE_HPP

#include <cmath>
#include <cstdint>

namespace mlayer {

// Clamp a float to the symmetric int8 range, rounding to nearest.
inline std::int8_t float_to_int8(float v) {
    const auto int32 = static_cast<std::int32_t>(std::round(v));
    if (int32 > 127) {
        return 127;
    }
    if (int32 < -127) {
        return -127;
    }
    return static_cast<std::int8_t>(int32);
}

}  // namespace mlayer

#endif  // MLAYER_QUANTIZE_HPP
