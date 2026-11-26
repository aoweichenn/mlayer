// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_STFT_HPP
#define MLAYER_STFT_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlayer {

namespace detail {

inline bool stft_is_power_of_two(std::int32_t value)
{
    return value > 0 && (value & (value - 1)) == 0;
}

// In-place radix-2 FFT using float32 arithmetic. Bluestein needs a
// power-of-two transform length even when the frame length is not.
inline void stft_fft_radix2(std::vector<std::complex<float> >& data, bool inverse)
{
    const std::size_t size = data.size();
    for (std::size_t i = 1, j = 0; i < size; ++i)
    {
        std::size_t bit = size >> 1;
        for (; (j & bit) != 0; bit >>= 1)
        {
            j ^= bit;
        }
        j ^= bit;
        if (i < j)
        {
            std::swap(data[i], data[j]);
        }
    }

    constexpr float kPi = 3.14159265358979323846F;
    for (std::size_t length = 2; length <= size; length <<= 1)
    {
        const float angle = (inverse ? 2.0F : -2.0F) * kPi / static_cast<float>(length);
        const std::size_t half = length >> 1;
        for (std::size_t start = 0; start < size; start += length)
        {
            for (std::size_t k = 0; k < half; ++k)
            {
                const float phase = angle * static_cast<float>(k);
                const std::complex<float> twiddle(std::cos(phase), std::sin(phase));
                const std::complex<float> u = data[start + k];
                const std::complex<float> v = twiddle * data[start + k + half];
                data[start + k] = u + v;
                data[start + k + half] = u - v;
            }
        }
    }
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
            const float inverse_scale = 1.0F / static_cast<float>(fft_length);

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
                        const std::complex<float> value = frame[static_cast<std::size_t>(k)] *
                                                          chirp[static_cast<std::size_t>(k)] * inverse_scale;
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
