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

// Unpack one signed value from ncnn's little-endian weight-block format.
inline std::int32_t unpack_weight_block(const unsigned char* ptr, std::int32_t k, std::int32_t bits) {
    if (bits == 8) {
        return static_cast<std::int32_t>(static_cast<std::int8_t>(ptr[k]));
    }
    const std::int32_t bit_offset = k * bits;
    const std::int32_t byte_offset = bit_offset / 8;
    const std::int32_t bit_shift = bit_offset % 8;
    unsigned int v = static_cast<unsigned int>(ptr[byte_offset]);
    if (bit_shift + bits > 8) {
        v |= static_cast<unsigned int>(ptr[byte_offset + 1]) << 8;
    }
    const std::int32_t mask = (1 << bits) - 1;
    const std::int32_t raw = static_cast<std::int32_t>((v >> bit_shift) & static_cast<unsigned int>(mask));
    const std::int32_t sign_bit = 1 << (bits - 1);
    return (raw ^ sign_bit) - sign_bit;
}

inline void fill_int8(std::int8_t* dst, std::int32_t size, std::int8_t value) {
    for (std::int32_t i = 0; i < size; ++i) {
        dst[i] = value;
    }
}

}  // namespace detail
}  // namespace mlayer

#endif  // MLAYER_INT8_QUANTIZE_HPP
