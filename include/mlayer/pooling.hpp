// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_POOLING_HPP
#define MLAYER_POOLING_HPP

#include <cfloat>
#include <climits>
#include <cstddef>
#include <cstdint>

namespace mlayer {

constexpr std::int32_t kBorderConstant = 0;

#define MLAYER_POOLING_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate, \
             void (*CopyMakeBorder)(const MatT&, MatT&, std::int32_t, std::int32_t, std::int32_t, std::int32_t, \
                                    std::int32_t, float, const OptionT&)>

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate,
         void (*CopyMakeBorder)(const MatT&, MatT&, std::int32_t, std::int32_t, std::int32_t, std::int32_t, std::int32_t,
                                float, const OptionT&)>
class Pooling : public LayerT {
public:
    Pooling();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

    enum PoolMethod {
        PoolMethod_MAX = 0,
        PoolMethod_AVE = 1
    };

protected:
    void make_padding(const MatT& bottom_blob, MatT& bottom_blob_bordered, const OptionT& opt) const;

public:
    std::int32_t pooling_type;
    std::int32_t kernel_w;
    std::int32_t kernel_h;
    std::int32_t stride_w;
    std::int32_t stride_h;
    std::int32_t pad_left;
    std::int32_t pad_right;
    std::int32_t pad_top;
    std::int32_t pad_bottom;
    std::int32_t global_pooling;
    std::int32_t pad_mode;
    std::int32_t avgpool_count_include_pad;
    std::int32_t adaptive_pooling;
    std::int32_t out_w;
    std::int32_t out_h;
};

MLAYER_POOLING_TEMPLATE
Pooling<LayerT, MatT, OptionT, ParamDictT, Validate, CopyMakeBorder>::Pooling()
{
    this->one_blob_only = true;
    this->support_inplace = false;
}

MLAYER_POOLING_TEMPLATE
std::int32_t Pooling<LayerT, MatT, OptionT, ParamDictT, Validate, CopyMakeBorder>::load_param(const ParamDictT& pd)
{
    pooling_type = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    stride_w = pd.get(2, 1);
    stride_h = pd.get(12, stride_w);
    pad_left = pd.get(3, 0);
    pad_right = pd.get(14, pad_left);
    pad_top = pd.get(13, pad_left);
    pad_bottom = pd.get(15, pad_top);
    global_pooling = pd.get(4, 0);
    pad_mode = pd.get(5, 0);
    avgpool_count_include_pad = pd.get(6, 0);
    adaptive_pooling = pd.get(7, 0);
    out_w = pd.get(8, 0);
    out_h = pd.get(18, out_w);

if constexpr (Validate) {
    if (pooling_type < PoolMethod_MAX || pooling_type > PoolMethod_AVE)
        return -1;

    if (!global_pooling && adaptive_pooling)
    {
        if ((out_w <= 0 && out_w != -233) || (out_h <= 0 && out_h != -233))
            return -1;
    }

    if (!global_pooling && !adaptive_pooling)
    {
        if (pad_mode < 0 || pad_mode > 3)
            return -1;

        if (kernel_w <= 0 || stride_w <= 0 || kernel_h <= 0 || stride_h <= 0 || kernel_w > INT_MAX / kernel_h)
            return -1;
    }
}

    return 0;
}

MLAYER_POOLING_TEMPLATE
std::int32_t Pooling<LayerT, MatT, OptionT, ParamDictT, Validate, CopyMakeBorder>::forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const
{
    // max value in NxN window
    // avg value in NxN window

    std::int32_t w = bottom_blob.w;
    std::int32_t h = bottom_blob.h;
    std::int32_t channels = bottom_blob.c;
    std::size_t elemsize = bottom_blob.elemsize;

    //     NCNN_LOGE("Pooling     input %d x %d  pad = %d %d %d %d  ksize=%d %d  stride=%d %d", w, h, pad_left, pad_right, pad_top, pad_bottom, kernel_w, kernel_h, stride_w, stride_h);
    if (global_pooling)
    {
        top_blob.create(channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        std::int32_t size = w * h;

        if (pooling_type == PoolMethod_MAX)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                float max = ptr[0];
                for (std::int32_t i = 0; i < size; i++)
                {
                    max = std::max(max, ptr[i]);
                }

                top_blob[q] = max;
            }
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                float sum = 0.f;
                for (std::int32_t i = 0; i < size; i++)
                {
                    sum += ptr[i];
                }

                top_blob[q] = sum / size;
            }
        }

        return 0;
    }

    if (adaptive_pooling)
    {
        std::int32_t _out_w = out_w == -233 ? w : out_w;
        std::int32_t _out_h = out_h == -233 ? h : out_h;

        if (_out_w == w && _out_h == h)
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(_out_w, _out_h, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pooling_type == PoolMethod_MAX)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < channels; q++)
            {
                const float* inptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                for (std::int32_t i = 0; i < _out_h; i++)
                {
                    // floor div
                    const std::int32_t ih0 = h * i / _out_h;
                    // ceil div
                    const std::int32_t ih1 = (h * (i + 1) + _out_h - 1) / _out_h;
                    for (std::int32_t j = 0; j < _out_w; j++)
                    {
                        // floor div
                        const std::int32_t iw0 = w * j / _out_w;
                        // ceil div
                        const std::int32_t iw1 = (w * (j + 1) + _out_w - 1) / _out_w;

                        float max = inptr[ih0 * w + iw0];
                        for (std::int32_t ih = ih0; ih < ih1; ih++)
                        {
                            for (std::int32_t iw = iw0; iw < iw1; iw++)
                            {
                                max = std::max(max, inptr[ih * w + iw]);
                            }
                        }

                        outptr[j] = max;
                    }
                    outptr += _out_w;
                }
            }
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < channels; q++)
            {
                const float* inptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                for (std::int32_t i = 0; i < _out_h; i++)
                {
                    // floor div
                    const std::int32_t ih0 = h * i / _out_h;
                    // ceil div
                    const std::int32_t ih1 = (h * (i + 1) + _out_h - 1) / _out_h;
                    const std::int32_t hk = ih1 - ih0;
                    for (std::int32_t j = 0; j < _out_w; j++)
                    {
                        // floor div
                        const std::int32_t iw0 = w * j / _out_w;
                        // ceil div
                        const std::int32_t iw1 = (w * (j + 1) + _out_w - 1) / _out_w;
                        const std::int32_t wk = iw1 - iw0;

                        float sum = 0;
                        for (std::int32_t ih = ih0; ih < ih1; ih++)
                        {
                            for (std::int32_t iw = iw0; iw < iw1; iw++)
                            {
                                sum += inptr[ih * w + iw];
                            }
                        }

                        outptr[j] = sum / hk / wk;
                    }

                    outptr += _out_w;
                }
            }
        }

        return 0;
    }

    MatT bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    std::int32_t outw = (w - kernel_w) / stride_w + 1;
    std::int32_t outh = (h - kernel_h) / stride_h + 1;

    top_blob.create(outw, outh, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const std::int32_t maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<std::int32_t> _space_ofs(maxk);
    std::int32_t* space_ofs = &_space_ofs[0];
    {
        std::int32_t p1 = 0;
        std::int32_t p2 = 0;
        std::int32_t gap = w - kernel_w;
        for (std::int32_t i = 0; i < kernel_h; i++)
        {
            for (std::int32_t j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2++;
            }
            p2 += gap;
        }
    }

    if (pooling_type == PoolMethod_MAX)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t q = 0; q < channels; q++)
        {
            const MatT m = bottom_blob_bordered.channel(q);
            float* outptr = top_blob.channel(q);

            for (std::int32_t i = 0; i < outh; i++)
            {
                for (std::int32_t j = 0; j < outw; j++)
                {
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    float max = sptr[0];

                    for (std::int32_t k = 0; k < maxk; k++)
                    {
                        float val = sptr[space_ofs[k]];
                        max = std::max(max, val);
                    }

                    outptr[j] = max;
                }

                outptr += outw;
            }
        }
    }
    else if (pooling_type == PoolMethod_AVE)
    {
        if (avgpool_count_include_pad == 0)
        {
            std::int32_t wtailpad = 0;
            std::int32_t htailpad = 0;

            if (pad_mode == 0) // full padding
            {
                wtailpad = bottom_blob_bordered.w - bottom_blob.w - pad_left - pad_right;
                htailpad = bottom_blob_bordered.h - bottom_blob.h - pad_top - pad_bottom;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < channels; q++)
            {
                const MatT m = bottom_blob_bordered.channel(q);
                float* outptr = top_blob.channel(q);

                for (std::int32_t i = 0; i < outh; i++)
                {
                    std::int32_t sy0 = i * stride_h;

                    for (std::int32_t j = 0; j < outw; j++)
                    {
                        std::int32_t sx0 = j * stride_w;

                        float sum = 0;
                        std::int32_t area = 0;

                        for (std::int32_t ki = 0; ki < kernel_h; ki++)
                        {
                            std::int32_t sy = sy0 + ki;

                            if (sy < pad_top)
                                continue;

                            if (sy >= h - pad_bottom - htailpad)
                                break;

                            for (std::int32_t kj = 0; kj < kernel_w; kj++)
                            {
                                std::int32_t sx = sx0 + kj;

                                if (sx < pad_left)
                                    continue;

                                if (sx >= w - pad_right - wtailpad)
                                    break;

                                float val = m.row(sy)[sx];
                                sum += val;
                                area += 1;
                            }
                        }

                        outptr[j] = sum / area;
                    }

                    outptr += outw;
                }
            }
        }
        else // if (avgpool_count_include_pad == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (std::int32_t q = 0; q < channels; q++)
            {
                const MatT m = bottom_blob_bordered.channel(q);
                float* outptr = top_blob.channel(q);

                for (std::int32_t i = 0; i < outh; i++)
                {
                    for (std::int32_t j = 0; j < outw; j++)
                    {
                        const float* sptr = m.row(i * stride_h) + j * stride_w;

                        float sum = 0;

                        for (std::int32_t k = 0; k < maxk; k++)
                        {
                            float val = sptr[space_ofs[k]];
                            sum += val;
                        }

                        outptr[j] = sum / maxk;
                    }

                    outptr += outw;
                }
            }
        }
    }

    return 0;
}

MLAYER_POOLING_TEMPLATE
void Pooling<LayerT, MatT, OptionT, ParamDictT, Validate, CopyMakeBorder>::make_padding(const MatT& bottom_blob, MatT& bottom_blob_bordered, const OptionT& opt) const
{
    std::int32_t w = bottom_blob.w;
    std::int32_t h = bottom_blob.h;

    bottom_blob_bordered = bottom_blob;

    float pad_value = 0.f;
    if (pooling_type == PoolMethod_MAX)
    {
        pad_value = bottom_blob.elemsize == 1 ? -128.f : -FLT_MAX;
    }
    else if (pooling_type == PoolMethod_AVE)
    {
        pad_value = 0.f;
    }

    std::int32_t wtailpad = 0;
    std::int32_t htailpad = 0;

    if (pad_mode == 0) // full padding
    {
        std::int32_t wtail = (w + pad_left + pad_right - kernel_w) % stride_w;
        std::int32_t htail = (h + pad_top + pad_bottom - kernel_h) % stride_h;

        if (wtail != 0)
            wtailpad = stride_w - wtail;
        if (htail != 0)
            htailpad = stride_h - htail;

        OptionT opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        CopyMakeBorder(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom + htailpad, pad_left, pad_right + wtailpad, kBorderConstant, pad_value, opt_b);
    }
    else if (pad_mode == 1) // valid padding
    {
        OptionT opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        CopyMakeBorder(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right, kBorderConstant, pad_value, opt_b);
    }
    else if (pad_mode == 2) // tensorflow padding=SAME or onnx padding=SAME_UPPER
    {
        std::int32_t wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
        std::int32_t hpad = kernel_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            OptionT opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            CopyMakeBorder(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, kBorderConstant, pad_value, opt_b);
        }
    }
    else if (pad_mode == 3) // onnx padding=SAME_LOWER
    {
        std::int32_t wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
        std::int32_t hpad = kernel_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            OptionT opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            CopyMakeBorder(bottom_blob, bottom_blob_bordered, hpad - hpad / 2, hpad / 2, wpad - wpad / 2, wpad / 2, kBorderConstant, pad_value, opt_b);
        }
    }
}


#undef MLAYER_POOLING_TEMPLATE

}  // namespace mlayer

#endif  // MLAYER_POOLING_HPP
