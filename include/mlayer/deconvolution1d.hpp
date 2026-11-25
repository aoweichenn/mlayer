// Copyright 2021 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_DECONVOLUTION1D_HPP
#define MLAYER_DECONVOLUTION1D_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace mlayer {

// 1D transposed convolution (also used for DeconvolutionDepthWise1D via the
// group parameter).
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT,
         bool Validate = false, float (*Activation)(float, std::int32_t, const MatT&) = nullptr>
class Deconvolution1D : public LayerT {
public:
    Deconvolution1D();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t load_model(const ModelBinT& mb) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

public:
    std::int32_t num_output;
    std::int32_t kernel_w;
    std::int32_t dilation_w;
    std::int32_t stride_w;
    std::int32_t pad_left;
    std::int32_t pad_right;
    std::int32_t output_pad_right;
    std::int32_t output_w;
    std::int32_t bias_term;
    std::int32_t weight_data_size;
    std::int32_t group;
    std::int32_t activation_type;
    MatT activation_params;
    std::int32_t dynamic_weight;
    MatT weight_data;
    MatT bias_data;
};

#define MLAYER_DECONV1D_TEMPLATE                                                                       \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT, \
             bool Validate, float (*Activation)(float, std::int32_t, const MatT&)>

MLAYER_DECONV1D_TEMPLATE
Deconvolution1D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation>::Deconvolution1D() {
    this->one_blob_only = true;
    this->support_inplace = false;
    this->num_output = 0;
    this->kernel_w = 0;
    this->dilation_w = 1;
    this->stride_w = 1;
    this->pad_left = 0;
    this->pad_right = 0;
    this->output_pad_right = 0;
    this->output_w = 0;
    this->bias_term = 0;
    this->weight_data_size = 0;
    this->group = 1;
    this->activation_type = 0;
    this->dynamic_weight = 0;
}

MLAYER_DECONV1D_TEMPLATE
std::int32_t Deconvolution1D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation>::load_param(
    const ParamDictT& pd) {
    this->num_output = pd.get(0, 0);
    this->kernel_w = pd.get(1, 0);
    this->dilation_w = pd.get(2, 1);
    this->stride_w = pd.get(3, 1);
    this->pad_left = pd.get(4, 0);
    this->pad_right = pd.get(15, this->pad_left);
    this->output_pad_right = pd.get(18, 0);
    this->output_w = pd.get(20, 0);
    this->bias_term = pd.get(5, 0);
    this->weight_data_size = pd.get(6, 0);
    this->group = pd.get(7, 1);
    this->activation_type = pd.get(9, 0);
    this->activation_params = pd.get(10, MatT());
    this->dynamic_weight = pd.get(28, 0);
    if (this->kernel_w <= 0 || this->num_output <= 0 || this->group <= 0) {
        return -1;
    }
    return 0;
}

MLAYER_DECONV1D_TEMPLATE
std::int32_t Deconvolution1D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation>::load_model(
    const ModelBinT& mb) {
    if (this->dynamic_weight != 0) {
        return 0;
    }
    this->weight_data = mb.load(this->weight_data_size, 0);
    if (this->weight_data.empty()) {
        return -100;
    }
    if (this->bias_term != 0) {
        this->bias_data = mb.load(this->num_output, 1);
        if (this->bias_data.empty()) {
            return -100;
        }
    }
    return 0;
}

MLAYER_DECONV1D_TEMPLATE
std::int32_t Deconvolution1D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation>::forward(
    const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const {
    const std::int32_t w = bottom_blob.w;
    const std::int32_t inch = bottom_blob.h;
    const std::size_t elemsize = bottom_blob.elemsize;
    const std::int32_t kernel_extent = this->dilation_w * (this->kernel_w - 1) + 1;
    const std::int32_t outw_full = (w - 1) * this->stride_w + kernel_extent + this->output_pad_right;
    const std::int32_t outw = outw_full - this->pad_left - this->pad_right - this->output_pad_right;
    if (outw <= 0 || inch % this->group != 0 || this->num_output % this->group != 0) {
        return -1;
    }

    top_blob.create(outw, this->num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty()) {
        return -100;
    }

    const std::int32_t inch_g = inch / this->group;
    const std::int32_t outch_g = this->num_output / this->group;
    const float* weight = this->weight_data;

    // Accumulate into a zeroed full-size buffer, then cut the padding.
    MatT bordered;
    bordered.create(outw_full, this->num_output, elemsize, opt.workspace_allocator);
    if (bordered.empty()) {
        return -100;
    }
    bordered.fill(0.0F);

#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t p = 0; p < this->num_output; ++p) {
        const std::int32_t g = p / outch_g;
        const float* kptr = weight + static_cast<std::size_t>(p) * inch_g * this->kernel_w;
        float* outptr = bordered.row(p);
        for (std::int32_t q = g * inch_g; q < (g + 1) * inch_g; ++q) {
            const float* sptr = bottom_blob.row(q);
            const float* kk = kptr + static_cast<std::size_t>(q - g * inch_g) * this->kernel_w;
            for (std::int32_t i = 0; i < w; ++i) {
                const float v = sptr[i];
                float* base = outptr + i * this->stride_w;
                for (std::int32_t k = 0; k < this->kernel_w; ++k) {
                    base[k * this->dilation_w] += v * kk[k];
                }
            }
        }
    }

#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t p = 0; p < this->num_output; ++p) {
        const float* src = bordered.row(p) + this->pad_left;
        float* dst = top_blob.row(p);
        const float bias = this->bias_term != 0 ? this->bias_data[p] : 0.0F;
        for (std::int32_t i = 0; i < outw; ++i) {
            const float v = src[i] + bias;
            dst[i] = this->activation_type == 0 ? v : Activation(v, this->activation_type, this->activation_params);
        }
    }
    return 0;
}

}  // namespace mlayer

#endif  // MLAYER_DECONVOLUTION1D_HPP
