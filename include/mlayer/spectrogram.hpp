// Copyright 2024 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_SPECTROGRAM_HPP
#define MLAYER_SPECTROGRAM_HPP

#include <cmath>
#include <cstdint>
#include <vector>

namespace mlayer {

namespace detail {

// Constant / replicate / reflect padding for a 1D float blob, matching the
// ncnn Padding layer used by Spectrogram (BORDER_CONSTANT/REPLICATE/REFLECT).
inline void spectrogram_window(std::vector<float>& window, std::int32_t n_fft, std::int32_t winlen,
                               std::int32_t window_type, std::int32_t normalized, bool inverse) {
    constexpr double kTwoPi = 6.28318530717958647692;
    window.assign(static_cast<std::size_t>(n_fft) + (normalized == 2 ? 1 : 0), 0.0F);
    std::int32_t p = 0;
    const std::int32_t lead = (n_fft - winlen) / 2;
    p += lead;
    for (std::int32_t i = 0; i < winlen; ++i) {
        float v = 1.0F;
        if (window_type == 1) {
            v = 0.5F * static_cast<float>(1.0 - std::cos(kTwoPi * i / winlen));
        } else if (window_type == 2) {
            v = 0.54F - 0.46F * static_cast<float>(std::cos(kTwoPi * i / winlen));
        }
        window[static_cast<std::size_t>(p++)] = v;
    }
    if (normalized == 2) {
        float sqsum = 0.0F;
        for (std::int32_t i = 0; i < n_fft; ++i) {
            sqsum += window[static_cast<std::size_t>(i)] * window[static_cast<std::size_t>(i)];
        }
        window[static_cast<std::size_t>(n_fft)] =
            inverse ? std::sqrt(sqsum) : 1.0F / std::sqrt(sqsum);
    }
}

template<typename MatT, typename OptionT>
MatT spectrogram_pad(const MatT& src, std::int32_t left, std::int32_t right, std::int32_t type,
                     std::int32_t outw, const OptionT& opt) {
    MatT dst;
    dst.create(outw, 4u, opt.workspace_allocator);
    if (dst.empty()) {
        return dst;
    }
    const std::int32_t w = src.w;
    for (std::int32_t x = 0; x < outw; ++x) {
        const std::int32_t idx = x - left;
        if (type == 0) {
            dst[x] = (idx >= 0 && idx < w) ? src[idx] : 0.0F;
        } else if (type == 1) {
            dst[x] = src[idx < 0 ? 0 : (idx >= w ? w - 1 : idx)];
        } else {
            std::int32_t ridx = idx;
            if (ridx < 0) {
                ridx = -ridx;
            } else if (ridx >= w) {
                ridx = 2 * w - 2 - ridx;
            }
            dst[x] = src[ridx];
        }
    }
    return dst;
}

}  // namespace detail

// STFT magnitude / power / complex spectrogram over a 1D (w) blob.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate = false>
class Spectrogram : public LayerT {
public:
    Spectrogram();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

public:
    std::int32_t n_fft;
    std::int32_t power;
    std::int32_t hoplen;
    std::int32_t winlen;
    std::int32_t window_type;
    std::int32_t center;
    std::int32_t pad_type;
    std::int32_t normalized;
    std::int32_t onesided;
    std::vector<float> window_data;
};

#define MLAYER_SPECTROGRAM_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate>

MLAYER_SPECTROGRAM_TEMPLATE
Spectrogram<LayerT, MatT, OptionT, ParamDictT, Validate>::Spectrogram() {
    this->one_blob_only = true;
    this->support_inplace = false;
    this->n_fft = 0;
    this->power = 0;
    this->hoplen = 0;
    this->winlen = 0;
    this->window_type = 0;
    this->center = 1;
    this->pad_type = 2;
    this->normalized = 0;
    this->onesided = 1;
}

MLAYER_SPECTROGRAM_TEMPLATE
std::int32_t Spectrogram<LayerT, MatT, OptionT, ParamDictT, Validate>::load_param(const ParamDictT& pd) {
    this->n_fft = pd.get(0, 0);
    this->power = pd.get(1, 0);
    this->hoplen = pd.get(2, this->n_fft / 4);
    this->winlen = pd.get(3, this->n_fft);
    this->window_type = pd.get(4, 0);
    this->center = pd.get(5, 1);
    this->pad_type = pd.get(6, 2);
    this->normalized = pd.get(7, 0);
    this->onesided = pd.get(8, 1);
    if (this->n_fft <= 0 || this->winlen <= 0 || this->winlen > this->n_fft || this->hoplen <= 0 ||
        this->power < 0 || this->power > 2 || this->window_type < 0 || this->window_type > 2 ||
        this->normalized < 0 || this->normalized > 2 || (this->center == 1 && (this->pad_type < 0 || this->pad_type > 2))) {
        return -1;
    }
    detail::spectrogram_window(this->window_data, this->n_fft, this->winlen, this->window_type, this->normalized,
                               false);
    return 0;
}

MLAYER_SPECTROGRAM_TEMPLATE
std::int32_t Spectrogram<LayerT, MatT, OptionT, ParamDictT, Validate>::forward(const MatT& bottom_blob,
                                                                               MatT& top_blob,
                                                                               const OptionT& opt) const {
    constexpr double kTwoPi = 6.28318530717958647692;
    MatT bordered = bottom_blob;
    if (this->center == 1) {
        const std::int32_t pad = this->n_fft / 2;
        bordered = detail::spectrogram_pad<MatT, OptionT>(bottom_blob, pad, pad, this->pad_type,
                                                          bottom_blob.w + 2 * pad, opt);
        if (bordered.empty()) {
            return -100;
        }
    }

    const std::int32_t size = bordered.w;
    const std::int32_t frames = (size - this->n_fft) / this->hoplen + 1;
    if (frames <= 0) {
        return -1;
    }
    const std::int32_t freqs_onesided = this->n_fft / 2 + 1;
    const std::int32_t freqs = this->onesided != 0 ? freqs_onesided : this->n_fft;
    const std::size_t elemsize = bordered.elemsize;

    if (this->power == 0) {
        top_blob.create(2, frames, freqs, elemsize, opt.blob_allocator);
    } else {
        top_blob.create(frames, freqs, elemsize, opt.blob_allocator);
    }
    if (top_blob.empty()) {
        return -100;
    }

#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t i = 0; i < freqs_onesided; ++i) {
        float* outptr = this->power == 0 ? top_blob.channel(i).row(0) : top_blob.row(i);
        for (std::int32_t j = 0; j < frames; ++j) {
            float re = 0.0F;
            float im = 0.0F;
            const float* frame = static_cast<const float*>(bordered) + j * this->hoplen;
            for (std::int32_t k = 0; k < this->n_fft; ++k) {
                const float v = frame[k] * this->window_data[static_cast<std::size_t>(k)];
                const double angle = kTwoPi * i * k / this->n_fft;
                re += v * static_cast<float>(std::cos(angle));
                im -= v * static_cast<float>(std::sin(angle));
            }
            if (this->normalized == 1) {
                const float norm = 1.0F / std::sqrt(static_cast<float>(this->n_fft));
                re *= norm;
                im *= norm;
            } else if (this->normalized == 2) {
                const float norm = this->window_data[static_cast<std::size_t>(this->n_fft)];
                re *= norm;
                im *= norm;
            }
            if (this->power == 0) {
                outptr[0] = re;
                outptr[1] = im;
                outptr += 2;
            } else if (this->power == 1) {
                outptr[0] = std::sqrt(re * re + im * im);
                outptr += 1;
            } else {
                outptr[0] = re * re + im * im;
                outptr += 1;
            }
        }
    }

    if (this->onesided == 0) {
#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t i = freqs_onesided; i < this->n_fft; ++i) {
            if (this->power == 0) {
                const float* ptr = top_blob.channel(this->n_fft - i).row(0);
                float* outptr = top_blob.channel(i).row(0);
                for (std::int32_t j = 0; j < frames; ++j) {
                    outptr[0] = ptr[0];
                    outptr[1] = -ptr[1];
                    ptr += 2;
                    outptr += 2;
                }
            } else {
                const float* ptr = top_blob.row(this->n_fft - i);
                float* outptr = top_blob.row(i);
                for (std::int32_t j = 0; j < frames; ++j) {
                    outptr[j] = ptr[j];
                }
            }
        }
    }
    return 0;
}

// Inverse STFT: (2, frames, freqs) complex or (frames, freqs) magnitude input.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate = false>
class InverseSpectrogram : public LayerT {
public:
    InverseSpectrogram();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

public:
    std::int32_t n_fft;
    std::int32_t returns;
    std::int32_t hoplen;
    std::int32_t winlen;
    std::int32_t window_type;
    std::int32_t center;
    std::int32_t normalized;
    std::vector<float> window_data;
};

#define MLAYER_INVSPECTROGRAM_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate>

MLAYER_INVSPECTROGRAM_TEMPLATE
InverseSpectrogram<LayerT, MatT, OptionT, ParamDictT, Validate>::InverseSpectrogram() {
    this->one_blob_only = true;
    this->support_inplace = false;
    this->n_fft = 0;
    this->returns = 0;
    this->hoplen = 0;
    this->winlen = 0;
    this->window_type = 0;
    this->center = 1;
    this->normalized = 0;
}

MLAYER_INVSPECTROGRAM_TEMPLATE
std::int32_t InverseSpectrogram<LayerT, MatT, OptionT, ParamDictT, Validate>::load_param(const ParamDictT& pd) {
    this->n_fft = pd.get(0, 0);
    this->returns = pd.get(1, 0);
    this->hoplen = pd.get(2, this->n_fft / 4);
    this->winlen = pd.get(3, this->n_fft);
    this->window_type = pd.get(4, 0);
    this->center = pd.get(5, 1);
    this->normalized = pd.get(7, 0);
    if (this->n_fft <= 0 || this->winlen <= 0 || this->winlen > this->n_fft || this->hoplen <= 0 ||
        this->returns < 0 || this->returns > 2 || this->window_type < 0 || this->window_type > 2 ||
        this->normalized < 0 || this->normalized > 2) {
        return -1;
    }
    detail::spectrogram_window(this->window_data, this->n_fft, this->winlen, this->window_type, this->normalized,
                               true);
    return 0;
}

MLAYER_INVSPECTROGRAM_TEMPLATE
std::int32_t InverseSpectrogram<LayerT, MatT, OptionT, ParamDictT, Validate>::forward(const MatT& bottom_blob,
                                                                                      MatT& top_blob,
                                                                                      const OptionT& opt) const {
    constexpr double kTwoPi = 6.28318530717958647692;
    const std::int32_t frames = bottom_blob.h;
    const std::int32_t freqs = bottom_blob.c;
    const std::int32_t onesided = freqs == this->n_fft / 2 + 1 ? 1 : 0;
    const std::int32_t outsize = this->center != 0
                                     ? (frames - 1) * this->hoplen + (this->n_fft - this->n_fft / 2 * 2)
                                     : (frames - 1) * this->hoplen + this->n_fft;
    if (outsize <= 0) {
        return -1;
    }
    const std::size_t elemsize = bottom_blob.elemsize;

    if (this->returns == 0) {
        top_blob.create(2, outsize, elemsize, opt.blob_allocator);
    } else {
        top_blob.create(outsize, elemsize, opt.blob_allocator);
    }
    if (top_blob.empty()) {
        return -100;
    }

    std::vector<float> window_sumsquare(static_cast<std::size_t>(outsize + this->n_fft), 0.0F);
    top_blob.fill(0.0F);

    for (std::int32_t j = 0; j < frames; ++j) {
        std::vector<float> sp(static_cast<std::size_t>(2 * this->n_fft), 0.0F);
        if (onesided != 0) {
            for (std::int32_t k = 0; k < this->n_fft / 2 + 1; ++k) {
                sp[static_cast<std::size_t>(2 * k)] = bottom_blob.channel(k).row(j)[0];
                sp[static_cast<std::size_t>(2 * k + 1)] = bottom_blob.channel(k).row(j)[1];
            }
            for (std::int32_t k = this->n_fft / 2 + 1; k < this->n_fft; ++k) {
                sp[static_cast<std::size_t>(2 * k)] = bottom_blob.channel(this->n_fft - k).row(j)[0];
                sp[static_cast<std::size_t>(2 * k + 1)] = -bottom_blob.channel(this->n_fft - k).row(j)[1];
            }
        } else {
            for (std::int32_t k = 0; k < this->n_fft; ++k) {
                sp[static_cast<std::size_t>(2 * k)] = bottom_blob.channel(k).row(j)[0];
                sp[static_cast<std::size_t>(2 * k + 1)] = bottom_blob.channel(k).row(j)[1];
            }
        }

        if (this->normalized == 1) {
            const float norm = std::sqrt(static_cast<float>(this->n_fft));
            for (float& v : sp) {
                v *= norm;
            }
        } else if (this->normalized == 2) {
            const float norm = this->window_data[static_cast<std::size_t>(this->n_fft)];
            for (float& v : sp) {
                v *= norm;
            }
        }

#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t i = 0; i < this->n_fft; ++i) {
            float re = 0.0F;
            float im = 0.0F;
            for (std::int32_t k = 0; k < this->n_fft; ++k) {
                const double angle = kTwoPi * i * k / this->n_fft;
                const float c = static_cast<float>(std::cos(angle));
                const float s = static_cast<float>(std::sin(angle));
                re += sp[static_cast<std::size_t>(2 * k)] * c - sp[static_cast<std::size_t>(2 * k + 1)] * s;
                im += sp[static_cast<std::size_t>(2 * k)] * s + sp[static_cast<std::size_t>(2 * k + 1)] * c;
            }
            re /= static_cast<float>(this->n_fft);
            im /= static_cast<float>(this->n_fft);
            re *= this->window_data[static_cast<std::size_t>(i)];
            im *= this->window_data[static_cast<std::size_t>(i)];

            std::int32_t output_index = j * this->hoplen + i;
            if (this->center == 1) {
                output_index -= this->n_fft / 2;
            }
            if (output_index >= 0 && output_index < outsize) {
                window_sumsquare[static_cast<std::size_t>(output_index)] +=
                    this->window_data[static_cast<std::size_t>(i)] * this->window_data[static_cast<std::size_t>(i)];
                if (this->returns == 0) {
                    top_blob.row(output_index)[0] += re;
                    top_blob.row(output_index)[1] += im;
                } else if (this->returns == 1) {
                    top_blob[output_index] += re;
                } else {
                    top_blob[output_index] += im;
                }
            }
        }
    }

    for (std::int32_t i = 0; i < outsize; ++i) {
        const float norm = window_sumsquare[static_cast<std::size_t>(i)];
        if (norm == 0.0F) {
            continue;
        }
        if (this->returns == 0) {
            top_blob.row(i)[0] /= norm;
            top_blob.row(i)[1] /= norm;
        } else {
            top_blob[i] /= norm;
        }
    }
    return 0;
}

}  // namespace mlayer

#endif  // MLAYER_SPECTROGRAM_HPP
