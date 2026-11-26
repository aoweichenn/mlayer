// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_STFT_HPP
#define MLAYER_STFT_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace mlayer {

namespace detail {

inline bool stft_is_power_of_two(std::int32_t value)
{
    return value > 0 && (value & (value - 1)) == 0;
}

inline std::uint32_t stft_bit_reverse(std::uint32_t value, unsigned significant_bits)
{
    std::uint32_t reversed = 0;
    for (unsigned bit = 0; bit < significant_bits; ++bit)
    {
        reversed = (reversed << 1) | (value & 1U);
        value >>= 1;
    }
    return reversed;
}

// Radix-2 FFT using the same bit-reversed Vandermonde table and butterfly
// ordering as ONNX Runtime's DFT kernel. The inverse call deliberately reuses
// the forward twiddles; Bluestein undoes the resulting reversal when reading
// the convolution output. This keeps the phase noise of near-zero bins as
// close as possible to the ONNX Runtime reference.
inline void stft_fft_radix2(std::vector<std::complex<float> >& data, bool inverse)
{
    const std::size_t size = data.size();
    unsigned significant_bits = 0;
    for (std::size_t value = size; value > 1; value >>= 1)
    {
        ++significant_bits;
    }

    constexpr float kTau = 6.28318530717958647692F;
    const float angular_velocity = -kTau / static_cast<float>(size);
    std::vector<std::complex<float> > vandermonde(size);
    for (std::size_t i = 0; i < size; ++i)
    {
        const float angle = static_cast<float>(i) * angular_velocity;
        vandermonde[stft_bit_reverse(static_cast<std::uint32_t>(i), significant_bits)] =
            std::complex<float>(std::cos(angle), std::sin(angle));
    }

    std::vector<std::complex<float> > values(size);
    for (std::size_t i = 0; i < size; ++i)
    {
        values[i] = data[stft_bit_reverse(static_cast<std::uint32_t>(i), significant_bits)];
    }

    unsigned current_bits = 0;
    for (std::size_t length = 2; length <= size; length <<= 1)
    {
        const std::size_t midpoint = length >> 1;
        ++current_bits;
        for (std::size_t k = 0; k < midpoint; ++k)
        {
            const auto first_index = stft_bit_reverse(static_cast<std::uint32_t>(k), current_bits);
            const auto second_index = stft_bit_reverse(static_cast<std::uint32_t>(midpoint + k), current_bits);
            for (std::size_t start = 0; start < size; start += length)
            {
                const std::size_t even_index = k + start;
                const std::size_t odd_index = k + start + midpoint;
                const std::complex<float> even = values[even_index];
                const std::complex<float> odd = values[odd_index];
                values[even_index] = even + vandermonde[first_index] * odd;
                values[odd_index] = even + vandermonde[second_index] * odd;
            }
        }
    }

    if (inverse)
    {
        const float scale = 1.0F / static_cast<float>(size);
        for (std::complex<float>& value : values)
        {
            value *= scale;
        }
    }
    data = std::move(values);
}

}  // namespace detail

// One-sided short-time Fourier transform (ONNX `STFT` with onesided=1).
//
// Inputs:  [0] signal   (length)
//          [1] window   (frame_length)
//          [2] frame_step (scalar)
// Output:  [0] (bins, frames, 2): channel 0 is the real part, channel 1 the
//              imaginary part, bins = frame_length / 2 + 1.
//
// The caller is responsible for any centre padding (ONNX graphs that use
// `center=1` put a Pad layer in front). The transform is a direct DFT so the
// layer stays exact for any frame length; the usual frame lengths for TTS
// decoders are small enough that this is not the bottleneck yet.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class Stft : public LayerT {
public:
    Stft() {
        this->one_blob_only = false;
        this->support_inplace = false;
    }

    std::int32_t load_param(const ParamDictT& /*pd*/) override { return 0; }

    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override {
        if (bottom_blobs.size() < 3 || top_blobs.empty()) {
            return -100;
        }
        const MatT& signal = bottom_blobs[0];
        const MatT& window = bottom_blobs[1];
        const MatT& step_blob = bottom_blobs[2];

        const std::int32_t length = signal.w * signal.h;
        const std::int32_t frame_length = window.w * window.h;
        const auto step = static_cast<std::int32_t>(step_blob[0]);
        if (frame_length <= 1 || step <= 0 || length < frame_length) {
            return -100;
        }

        const std::int32_t frames = (length - frame_length) / step + 1;
        const std::int32_t bins = frame_length / 2 + 1;

        MatT& top_blob = top_blobs[0];
        // ONNX STFT is [N, frames, bins, 2]; ncnn 4-D is (w,h,d,c), so
        // real/imag must be the fastest axis and frames the channel axis.
        top_blob.create(2, bins, 1, frames, 4u, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }
        // Real/imag views are replaced by direct interleaved writes below.

        const auto* win = static_cast<const float*>(window);
        if (!detail::stft_is_power_of_two(frame_length)) {
            constexpr float kPi = 3.14159265358979323846F;
            std::size_t fft_length = 1;
            while (fft_length < static_cast<std::size_t>(2 * frame_length - 1)) {
                fft_length <<= 1;
            }
            std::vector<std::complex<float> > chirp(static_cast<std::size_t>(frame_length));
            for (std::int32_t n = 0; n < frame_length; ++n) {
                const float phase = -kPi * static_cast<float>(n) * static_cast<float>(n) /
                                    static_cast<float>(frame_length);
                chirp[static_cast<std::size_t>(n)] = std::complex<float>(std::cos(phase), std::sin(phase));
            }
            std::vector<std::complex<float> > kernel(fft_length);
            kernel[0] = std::conj(chirp[0]);
            for (std::int32_t n = 1; n < frame_length; ++n) {
                kernel[static_cast<std::size_t>(n)] = std::conj(chirp[static_cast<std::size_t>(n)]);
                kernel[fft_length - static_cast<std::size_t>(n)] = std::conj(chirp[static_cast<std::size_t>(n)]);
            }
            detail::stft_fft_radix2(kernel, false);

            #pragma omp parallel num_threads(opt.num_threads)
            {
                std::vector<std::complex<float> > frame(fft_length);
                #pragma omp for
                for (std::int32_t f = 0; f < frames; ++f) {
                    std::fill(frame.begin(), frame.end(), std::complex<float>(0.0F, 0.0F));
                    const auto* base = static_cast<const float*>(signal) +
                                       static_cast<std::size_t>(f) * static_cast<std::size_t>(step);
                    for (std::int32_t n = 0; n < frame_length; ++n) {
                        frame[static_cast<std::size_t>(n)] = std::complex<float>(base[n] * win[n], 0.0F) *
                                                              chirp[static_cast<std::size_t>(n)];
                    }
                    detail::stft_fft_radix2(frame, false);
                    for (std::size_t i = 0; i < fft_length; ++i) {
                        frame[i] *= kernel[i];
                    }
                    detail::stft_fft_radix2(frame, true);
                    for (std::int32_t k = 0; k < bins; ++k) {
                        const auto index = static_cast<std::size_t>(k);
                        const std::complex<float> convolution =
                            k == 0 ? frame[0] : frame[fft_length - index];
                        const std::complex<float> value = convolution * chirp[index];
                        top_blob.channel(f).row(k)[0] = value.real();
                        top_blob.channel(f).row(k)[1] = value.imag();
                    }
                }
            }
            return 0;
        }

        // Twiddle factors for one frame, reused by every frame.
        std::vector<float> cos_table(static_cast<std::size_t>(bins) * static_cast<std::size_t>(frame_length));
        std::vector<float> sin_table(static_cast<std::size_t>(bins) * static_cast<std::size_t>(frame_length));
        for (std::int32_t k = 0; k < bins; ++k) {
            for (std::int32_t n = 0; n < frame_length; ++n) {
                const double angle = 2.0 * 3.14159265358979323846 * static_cast<double>(k) *
                                     static_cast<double>(n) / static_cast<double>(frame_length);
                const std::size_t index = static_cast<std::size_t>(k) * static_cast<std::size_t>(frame_length) +
                                          static_cast<std::size_t>(n);
                cos_table[index] = static_cast<float>(std::cos(angle));
                sin_table[index] = static_cast<float>(std::sin(angle));
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t f = 0; f < frames; ++f) {
            const auto* base = static_cast<const float*>(signal) +
                               static_cast<std::size_t>(f) * static_cast<std::size_t>(step);
            for (std::int32_t k = 0; k < bins; ++k) {
                double sum_re = 0.0;
                double sum_im = 0.0;
                for (std::int32_t n = 0; n < frame_length; ++n) {
                    const std::size_t index = static_cast<std::size_t>(k) * static_cast<std::size_t>(frame_length) +
                                              static_cast<std::size_t>(n);
                    const double value = static_cast<double>(base[n]) * static_cast<double>(win[n]);
                    sum_re += value * static_cast<double>(cos_table[index]);
                    sum_im -= value * static_cast<double>(sin_table[index]);
                }
                top_blob.channel(f).row(k)[0] = static_cast<float>(sum_re);
                top_blob.channel(f).row(k)[1] = static_cast<float>(sum_im);
            }
        }
        return 0;
    }
};

}  // namespace mlayer

#endif  // MLAYER_STFT_HPP
