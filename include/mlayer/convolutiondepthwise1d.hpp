// Copyright 2022 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_CONVOLUTIONDEPTHWISE1D_HPP
#define MLAYER_CONVOLUTIONDEPTHWISE1D_HPP

#include <algorithm>
#include <cstdint>
#include <vector>

namespace mlayer {

// 1D depthwise / group convolution over a (w, channels) blob.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT,
         bool Validate = false, float (*Activation)(float, std::int32_t, const MatT&) = nullptr>
class ConvolutionDepthWise1D : public LayerT {
public:
    ConvolutionDepthWise1D();

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
    float pad_value;
    std::int32_t bias_term;
    std::int32_t weight_data_size;
    std::int32_t group;
    std::int32_t activation_type;
    MatT activation_params;
    std::int32_t dynamic_weight;
    MatT weight_data;
    MatT bias_data;
};

#define MLAYER_CONVDW1D_TEMPLATE                                                                       \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT, \
             bool Validate, float (*Activation)(float, std::int32_t, const MatT&)>

MLAYER_CONVDW1D_TEMPLATE
ConvolutionDepthWise1D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation>::
    ConvolutionDepthWise1D() {
    this->one_blob_only = true;
    this->support_inplace = false;
    this->num_output = 0;
    this->kernel_w = 0;
    this->dilation_w = 1;
    this->stride_w = 1;
    this->pad_left = 0;
    this->pad_right = 0;
    this->pad_value = 0.0F;
    this->bias_term = 0;
    this->weight_data_size = 0;
    this->group = 1;
    this->activation_type = 0;
    this->dynamic_weight = 0;
}

MLAYER_CONVDW1D_TEMPLATE
std::int32_t ConvolutionDepthWise1D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation>::
    load_param(const ParamDictT& pd) {
    this->num_output = pd.get(0, 0);
    this->kernel_w = pd.get(1, 0);
    this->dilation_w = pd.get(2, 1);
    this->stride_w = pd.get(3, 1);
    this->pad_left = pd.get(4, 0);
    this->pad_right = pd.get(15, this->pad_left);
    this->pad_value = pd.get(18, 0.0F);
    this->bias_term = pd.get(5, 0);
    this->weight_data_size = pd.get(6, 0);
    this->group = pd.get(7, 1);
    this->activation_type = pd.get(9, 0);
    this->activation_params = pd.get(10, MatT());
    this->dynamic_weight = pd.get(19, 0);
    if (this->kernel_w <= 0 || this->num_output <= 0 || this->group <= 0) {
        return -1;
    }
    return 0;
}

MLAYER_CONVDW1D_TEMPLATE
std::int32_t ConvolutionDepthWise1D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation>::
    load_model(const ModelBinT& mb) {
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

MLAYER_CONVDW1D_TEMPLATE
std::int32_t ConvolutionDepthWise1D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation>::
    forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const {
    const std::int32_t w = bottom_blob.w;
    const std::int32_t channels = bottom_blob.h;
    const std::size_t elemsize = bottom_blob.elemsize;

    const std::int32_t kernel_extent = this->dilation_w * (this->kernel_w - 1) + 1;
    const std::int32_t outw =
        (w + this->pad_left + this->pad_right - kernel_extent) / this->stride_w + 1;
    if (outw <= 0) {
        return -1;
    }

    top_blob.create(outw, this->num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty()) {
        return -100;
    }

    const std::int32_t inch = channels;
    const std::int32_t outch = this->num_output;
    if (inch % this->group != 0 || outch % this->group != 0) {
        return -1;
    }
    const std::int32_t inch_g = inch / this->group;
    const std::int32_t outch_g = outch / this->group;
    const float* weight = this->weight_data;

#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t p = 0; p < outch; ++p) {
        const std::int32_t g = p / outch_g;
        const std::int32_t p_in_group = p - g * outch_g;
        const float* kptr =
            weight + static_cast<std::size_t>(g) * outch_g * inch_g * this->kernel_w +
            static_cast<std::size_t>(p_in_group) * inch_g * this->kernel_w;
        float* outptr = top_blob.row(p);
        for (std::int32_t i = 0; i < outw; ++i) {
            float sum = this->bias_term != 0 ? this->bias_data[p] : 0.0F;
            for (std::int32_t q = 0; q < inch_g; ++q) {
                const float* sptr = bottom_blob.row(g * inch_g + q);
                const float* kk = kptr + static_cast<std::size_t>(q) * this->kernel_w;
                for (std::int32_t k = 0; k < this->kernel_w; ++k) {
                    const std::int32_t sx = i * this->stride_w + k * this->dilation_w - this->pad_left;
                    if (sx < 0 || sx >= w) {
                        sum += kk[k] * this->pad_value;
                    } else {
                        sum += sptr[sx] * kk[k];
                    }
                }
            }
            outptr[i] = this->activation_type == 0 ? sum : Activation(sum, this->activation_type, this->activation_params);
        }
    }
    return 0;
}

}  // namespace mlayer

#endif  // MLAYER_CONVOLUTIONDEPTHWISE1D_HPP
