// Copyright 2022 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_DECONVOLUTION3D_HPP
#define MLAYER_DECONVOLUTION3D_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace mlayer {

// 3D transposed convolution (also used for DeconvolutionDepthWise3D).
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT,
         bool Validate = false, float (*Activation)(float, std::int32_t, const MatT&) = nullptr>
class Deconvolution3D : public LayerT {
public:
    Deconvolution3D();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t load_model(const ModelBinT& mb) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

public:
    std::int32_t num_output;
    std::int32_t kernel_w;
    std::int32_t kernel_h;
    std::int32_t kernel_d;
    std::int32_t dilation_w;
    std::int32_t dilation_h;
    std::int32_t dilation_d;
    std::int32_t stride_w;
    std::int32_t stride_h;
    std::int32_t stride_d;
    std::int32_t pad_left;
    std::int32_t pad_right;
    std::int32_t pad_top;
    std::int32_t pad_bottom;
    std::int32_t pad_front;
    std::int32_t pad_behind;
    std::int32_t output_pad_right;
    std::int32_t output_pad_bottom;
    std::int32_t output_pad_behind;
    std::int32_t output_w;
    std::int32_t output_h;
    std::int32_t output_d;
    std::int32_t bias_term;
    std::int32_t weight_data_size;
    std::int32_t group;
    std::int32_t activation_type;
    MatT activation_params;
    MatT weight_data;
    MatT bias_data;
};

#define MLAYER_DECONV3D_TEMPLATE                                                                       \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT, \
             bool Validate, float (*Activation)(float, std::int32_t, const MatT&)>

MLAYER_DECONV3D_TEMPLATE
Deconvolution3D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation>::Deconvolution3D() {
    this->one_blob_only = true;
    this->support_inplace = false;
    this->num_output = 0;
    this->kernel_w = 0;
    this->kernel_h = 0;
    this->kernel_d = 0;
    this->dilation_w = 1;
    this->dilation_h = 1;
    this->dilation_d = 1;
    this->stride_w = 1;
    this->stride_h = 1;
    this->stride_d = 1;
    this->pad_left = 0;
    this->pad_right = 0;
    this->pad_top = 0;
    this->pad_bottom = 0;
    this->pad_front = 0;
    this->pad_behind = 0;
    this->output_pad_right = 0;
    this->output_pad_bottom = 0;
    this->output_pad_behind = 0;
    this->output_w = 0;
    this->output_h = 0;
    this->output_d = 0;
    this->bias_term = 0;
    this->weight_data_size = 0;
    this->group = 1;
    this->activation_type = 0;
}

MLAYER_DECONV3D_TEMPLATE
std::int32_t Deconvolution3D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation>::load_param(
    const ParamDictT& pd) {
    this->num_output = pd.get(0, 0);
    this->kernel_w = pd.get(1, 0);
    this->kernel_h = pd.get(11, this->kernel_w);
    this->kernel_d = pd.get(21, this->kernel_w);
    this->dilation_w = pd.get(2, 1);
    this->dilation_h = pd.get(12, this->dilation_w);
    this->dilation_d = pd.get(22, this->dilation_w);
    this->stride_w = pd.get(3, 1);
    this->stride_h = pd.get(13, this->stride_w);
    this->stride_d = pd.get(23, this->stride_w);
    this->pad_left = pd.get(4, 0);
    this->pad_right = pd.get(15, this->pad_left);
    this->pad_top = pd.get(14, this->pad_left);
    this->pad_bottom = pd.get(16, this->pad_top);
    this->pad_front = pd.get(24, this->pad_left);
    this->pad_behind = pd.get(17, this->pad_front);
    this->output_pad_right = pd.get(18, 0);
    this->output_pad_bottom = pd.get(19, this->output_pad_right);
    this->output_pad_behind = pd.get(20, this->output_pad_right);
    this->output_w = pd.get(25, 0);
    this->output_h = pd.get(26, this->output_w);
    this->output_d = pd.get(27, this->output_w);
    this->bias_term = pd.get(5, 0);
    this->weight_data_size = pd.get(6, 0);
    this->group = pd.get(7, 1);
    this->activation_type = pd.get(9, 0);
    this->activation_params = pd.get(10, MatT());
    if (this->kernel_w <= 0 || this->num_output <= 0 || this->group <= 0) {
        return -1;
    }
    return 0;
}

MLAYER_DECONV3D_TEMPLATE
std::int32_t Deconvolution3D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation>::load_model(
    const ModelBinT& mb) {
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

MLAYER_DECONV3D_TEMPLATE
std::int32_t Deconvolution3D<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Activation>::forward(
    const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const {
    const std::int32_t w = bottom_blob.w;
    const std::int32_t h = bottom_blob.h;
    const std::int32_t d = bottom_blob.d;
    const std::int32_t inch = bottom_blob.c;
    const std::size_t elemsize = bottom_blob.elemsize;
    const std::int32_t maxk = this->kernel_w * this->kernel_h * this->kernel_d;

    const std::int32_t extent_w = this->dilation_w * (this->kernel_w - 1) + 1;
    const std::int32_t extent_h = this->dilation_h * (this->kernel_h - 1) + 1;
    const std::int32_t extent_d = this->dilation_d * (this->kernel_d - 1) + 1;
    const std::int32_t outw_full = (w - 1) * this->stride_w + extent_w + this->output_pad_right;
    const std::int32_t outh_full = (h - 1) * this->stride_h + extent_h + this->output_pad_bottom;
    const std::int32_t outd_full = (d - 1) * this->stride_d + extent_d + this->output_pad_behind;
    const std::int32_t outw = outw_full - this->pad_left - this->pad_right - this->output_pad_right;
    const std::int32_t outh = outh_full - this->pad_top - this->pad_bottom - this->output_pad_bottom;
    const std::int32_t outd = outd_full - this->pad_front - this->pad_behind - this->output_pad_behind;
    if (outw <= 0 || outh <= 0 || outd <= 0 || inch % this->group != 0 ||
        this->num_output % this->group != 0) {
        return -1;
    }

    top_blob.create(outw, outh, outd, this->num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty()) {
        return -100;
    }

    const std::int32_t inch_g = inch / this->group;
    const std::int32_t outch_g = this->num_output / this->group;
    const float* weight = this->weight_data;

    MatT bordered;
    bordered.create(outw_full, outh_full, outd_full, this->num_output, elemsize, opt.workspace_allocator);
    if (bordered.empty()) {
        return -100;
    }
    bordered.fill(0.0F);

#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t p = 0; p < this->num_output; ++p) {
        const std::int32_t g = p / outch_g;
        const float* kptr = weight + static_cast<std::size_t>(p) * inch_g * maxk;
        MatT out = bordered.channel(p);
        for (std::int32_t q = g * inch_g; q < (g + 1) * inch_g; ++q) {
            const MatT src = bottom_blob.channel(q);
            const float* kk = kptr + static_cast<std::size_t>(q - g * inch_g) * maxk;
            for (std::int32_t iz = 0; iz < d; ++iz) {
                for (std::int32_t iy = 0; iy < h; ++iy) {
                    const float* srow = src.depth(iz).row(iy);
                    for (std::int32_t ix = 0; ix < w; ++ix) {
                        const float v = srow[ix];
                        float* base = out.depth(iz * this->stride_d).row(iy * this->stride_h) +
                                      ix * this->stride_w;
                        for (std::int32_t kz = 0; kz < this->kernel_d; ++kz) {
                            float* zptr = base + kz * this->dilation_d * outw_full * outh_full;
                            for (std::int32_t ky = 0; ky < this->kernel_h; ++ky) {
                                float* yptr = zptr + ky * this->dilation_h * outw_full;
                                const float* krow = kk + (kz * this->kernel_h + ky) * this->kernel_w;
                                for (std::int32_t kx = 0; kx < this->kernel_w; ++kx) {
                                    yptr[kx * this->dilation_w] += v * krow[kx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t p = 0; p < this->num_output; ++p) {
        const MatT src = bordered.channel(p);
        MatT dst = top_blob.channel(p);
        const float bias = this->bias_term != 0 ? this->bias_data[p] : 0.0F;
        for (std::int32_t oz = 0; oz < outd; ++oz) {
            for (std::int32_t oy = 0; oy < outh; ++oy) {
                const float* s = src.depth(oz + this->pad_front).row(oy + this->pad_top) + this->pad_left;
                float* t = dst.depth(oz).row(oy);
                for (std::int32_t ox = 0; ox < outw; ++ox) {
                    const float v = s[ox] + bias;
                    t[ox] = this->activation_type == 0
                                ? v
                                : Activation(v, this->activation_type, this->activation_params);
                }
            }
        }
    }
    return 0;
}

}  // namespace mlayer

#endif  // MLAYER_DECONVOLUTION3D_HPP
