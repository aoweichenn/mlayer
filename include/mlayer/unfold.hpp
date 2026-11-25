// Copyright 2023 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_UNFOLD_HPP
#define MLAYER_UNFOLD_HPP

#include <cstdint>

namespace mlayer {

// im2col: (w, h, c) feature map -> (outw*outh, kw*kh*c) patch matrix.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class Unfold : public LayerT {
public:
    Unfold();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

public:
    std::int32_t kernel_w;
    std::int32_t kernel_h;
    std::int32_t dilation_w;
    std::int32_t dilation_h;
    std::int32_t stride_w;
    std::int32_t stride_h;
    std::int32_t pad_left;
    std::int32_t pad_right;
    std::int32_t pad_top;
    std::int32_t pad_bottom;
    float pad_value;
};

#define MLAYER_UNFOLD_TEMPLATE template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>

MLAYER_UNFOLD_TEMPLATE
Unfold<LayerT, MatT, OptionT, ParamDictT>::Unfold() {
    this->one_blob_only = true;
    this->support_inplace = false;
    this->kernel_w = 0;
    this->kernel_h = 0;
    this->dilation_w = 1;
    this->dilation_h = 1;
    this->stride_w = 1;
    this->stride_h = 1;
    this->pad_left = 0;
    this->pad_right = 0;
    this->pad_top = 0;
    this->pad_bottom = 0;
    this->pad_value = 0.0F;
}

MLAYER_UNFOLD_TEMPLATE
std::int32_t Unfold<LayerT, MatT, OptionT, ParamDictT>::load_param(const ParamDictT& pd) {
    this->kernel_w = pd.get(1, 0);
    this->kernel_h = pd.get(11, this->kernel_w);
    this->dilation_w = pd.get(2, 1);
    this->dilation_h = pd.get(12, this->dilation_w);
    this->stride_w = pd.get(3, 1);
    this->stride_h = pd.get(13, this->stride_w);
    this->pad_left = pd.get(4, 0);
    this->pad_right = pd.get(15, this->pad_left);
    this->pad_top = pd.get(14, this->pad_left);
    this->pad_bottom = pd.get(16, this->pad_top);
    this->pad_value = pd.get(18, 0.0F);
    return this->kernel_w > 0 ? 0 : -1;
}

MLAYER_UNFOLD_TEMPLATE
std::int32_t Unfold<LayerT, MatT, OptionT, ParamDictT>::forward(const MatT& bottom_blob, MatT& top_blob,
                                                               const OptionT& opt) const {
    const std::int32_t w = bottom_blob.w;
    const std::int32_t h = bottom_blob.h;
    const std::int32_t channels = bottom_blob.c;
    const std::size_t elemsize = bottom_blob.elemsize;

    const std::int32_t pw = w + this->pad_left + this->pad_right;
    const std::int32_t ph = h + this->pad_top + this->pad_bottom;
    const std::int32_t extent_w = this->dilation_w * (this->kernel_w - 1) + 1;
    const std::int32_t extent_h = this->dilation_h * (this->kernel_h - 1) + 1;
    const std::int32_t outw = (pw - extent_w) / this->stride_w + 1;
    const std::int32_t outh = (ph - extent_h) / this->stride_h + 1;
    if (outw <= 0 || outh <= 0) {
        return -1;
    }
    const std::int32_t maxk = this->kernel_w * this->kernel_h;
    const std::int32_t size = outw * outh;

    top_blob.create(size, maxk * channels, elemsize, opt.blob_allocator);
    if (top_blob.empty()) {
        return -100;
    }

#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t p = 0; p < channels; ++p) {
        const MatT img = bottom_blob.channel(p);
        for (std::int32_t u = 0; u < this->kernel_h; ++u) {
            for (std::int32_t v = 0; v < this->kernel_w; ++v) {
                float* ptr = top_blob.row(p * maxk + u * this->kernel_w + v);
                for (std::int32_t i = 0; i < outh; ++i) {
                    for (std::int32_t j = 0; j < outw; ++j) {
                        const std::int32_t sy = i * this->stride_h + u * this->dilation_h - this->pad_top;
                        const std::int32_t sx = j * this->stride_w + v * this->dilation_w - this->pad_left;
                        if (sy < 0 || sy >= h || sx < 0 || sx >= w) {
                            ptr[j] = this->pad_value;
                        } else {
                            ptr[j] = img.row(sy)[sx];
                        }
                    }
                    ptr += outw;
                }
            }
        }
    }
    return 0;
}

// col2im: (outw*outh, kw*kh*c) patch matrix -> (output_w, output_h, c).
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT,
         void (*CopyCutBorder)(const MatT&, MatT&, std::int32_t, std::int32_t, std::int32_t, std::int32_t,
                               const OptionT&)>
class Fold : public LayerT {
public:
    Fold();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

public:
    std::int32_t kernel_w;
    std::int32_t kernel_h;
    std::int32_t dilation_w;
    std::int32_t dilation_h;
    std::int32_t stride_w;
    std::int32_t stride_h;
    std::int32_t pad_left;
    std::int32_t pad_right;
    std::int32_t pad_top;
    std::int32_t pad_bottom;
    std::int32_t output_w;
    std::int32_t output_h;
};

#define MLAYER_FOLD_TEMPLATE                                                                     \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT,              \
             void (*CopyCutBorder)(const MatT&, MatT&, std::int32_t, std::int32_t, std::int32_t, \
                                   std::int32_t, const OptionT&)>

MLAYER_FOLD_TEMPLATE
Fold<LayerT, MatT, OptionT, ParamDictT, CopyCutBorder>::Fold() {
    this->one_blob_only = true;
    this->support_inplace = false;
    this->kernel_w = 0;
    this->kernel_h = 0;
    this->dilation_w = 1;
    this->dilation_h = 1;
    this->stride_w = 1;
    this->stride_h = 1;
    this->pad_left = 0;
    this->pad_right = 0;
    this->pad_top = 0;
    this->pad_bottom = 0;
    this->output_w = 0;
    this->output_h = 0;
}

MLAYER_FOLD_TEMPLATE
std::int32_t Fold<LayerT, MatT, OptionT, ParamDictT, CopyCutBorder>::load_param(const ParamDictT& pd) {
    this->kernel_w = pd.get(1, 0);
    this->kernel_h = pd.get(11, this->kernel_w);
    this->dilation_w = pd.get(2, 1);
    this->dilation_h = pd.get(12, this->dilation_w);
    this->stride_w = pd.get(3, 1);
    this->stride_h = pd.get(13, this->stride_w);
    this->pad_left = pd.get(4, 0);
    this->pad_right = pd.get(15, this->pad_left);
    this->pad_top = pd.get(14, this->pad_left);
    this->pad_bottom = pd.get(16, this->pad_top);
    this->output_w = pd.get(20, 0);
    this->output_h = pd.get(21, this->output_w);
    return this->kernel_w > 0 ? 0 : -1;
}

MLAYER_FOLD_TEMPLATE
std::int32_t Fold<LayerT, MatT, OptionT, ParamDictT, CopyCutBorder>::forward(const MatT& bottom_blob,
                                                                            MatT& top_blob,
                                                                            const OptionT& opt) const {
    const std::int32_t max_channels = bottom_blob.h;
    const std::size_t elemsize = bottom_blob.elemsize;
    const std::int32_t extent_w = this->dilation_w * (this->kernel_w - 1) + 1;
    const std::int32_t extent_h = this->dilation_h * (this->kernel_h - 1) + 1;
    const std::int32_t outw = this->output_w + this->pad_left + this->pad_right;
    const std::int32_t outh = this->output_h + this->pad_top + this->pad_bottom;
    const std::int32_t inw = (outw - extent_w) / this->stride_w + 1;
    const std::int32_t inh = (outh - extent_h) / this->stride_h + 1;
    const std::int32_t maxk = this->kernel_w * this->kernel_h;
    const std::int32_t channels = max_channels / maxk;
    if (inw <= 0 || inh <= 0 || channels <= 0) {
        return -1;
    }

    MatT bordered;
    bordered.create(outw, outh, channels, elemsize, opt.workspace_allocator);
    if (bordered.empty()) {
        return -100;
    }

#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t p = 0; p < channels; ++p) {
        MatT outm = bordered.channel(p);
        outm.fill(0.0F);
        for (std::int32_t u = 0; u < this->kernel_h; ++u) {
            for (std::int32_t v = 0; v < this->kernel_w; ++v) {
                const float* sptr = bottom_blob.row(p * maxk + u * this->kernel_w + v);
                float* ptr = outm.row(u * this->dilation_h) + v * this->dilation_w;
                for (std::int32_t i = 0; i < inh; ++i) {
                    for (std::int32_t j = 0; j < inw; ++j) {
                        ptr[j * this->stride_w] += sptr[j];
                    }
                    sptr += inw;
                    ptr += this->stride_h * outw;
                }
            }
        }
    }

    if (this->pad_left > 0 || this->pad_right > 0 || this->pad_top > 0 || this->pad_bottom > 0) {
        CopyCutBorder(bordered, top_blob, this->pad_top, this->pad_bottom, this->pad_left, this->pad_right, opt);
        if (top_blob.empty()) {
            return -100;
        }
    } else {
        top_blob = bordered;
    }
    return 0;
}

}  // namespace mlayer

#endif  // MLAYER_UNFOLD_HPP
