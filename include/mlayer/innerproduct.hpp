// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_INNERPRODUCT_HPP
#define MLAYER_INNERPRODUCT_HPP

#include <cstddef>
#include <cstdint>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT, bool Validate,
         bool Int8, float (*Activation)(float, std::int32_t, const MatT&),
         void (*QuantizeInt8)(const MatT&, MatT&, const MatT&, const OptionT&)>
class InnerProduct : public LayerT {
public:
    InnerProduct();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t load_model(const ModelBinT& mb) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

protected:
    std::int32_t forward_int8(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const;

public:
    std::int32_t num_output;
    std::int32_t bias_term;
    std::int32_t weight_data_size;
    std::int32_t int8_scale_term;
    std::int32_t activation_type;
    MatT activation_params;

    MatT weight_data;
    MatT bias_data;

    MatT weight_data_int8_scales;
    MatT bottom_blob_int8_scales;
};

#define MLAYER_INNERPRODUCT_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT, bool Validate, \
             bool Int8, float (*Activation)(float, std::int32_t, const MatT&), \
             void (*QuantizeInt8)(const MatT&, MatT&, const MatT&, const OptionT&)>

MLAYER_INNERPRODUCT_TEMPLATE
InnerProduct<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8>::InnerProduct()
{
    this->one_blob_only = true;
    this->support_inplace = false;
}

MLAYER_INNERPRODUCT_TEMPLATE
std::int32_t InnerProduct<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8>::load_param(const ParamDictT& pd)
{
    num_output = pd.get(0, 0);
    bias_term = pd.get(1, 0);
    weight_data_size = pd.get(2, 0);
    int8_scale_term = pd.get(8, 0);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, MatT());

    const std::int32_t activation_params_type = pd.type(10);
if constexpr (Validate) {
    if (activation_params_type != 0 && activation_params_type != 4 && activation_params_type != 5 && activation_params_type != 6)
        return -1;

    if ((activation_params.dims != 0 || activation_params.w != 0 || activation_params.data) && (activation_params.dims != 1 || activation_params.w < 0 || activation_params.elempack != 1 || activation_params.elemsize != 4u || (activation_params.w > 0 && !activation_params.data)))
        return -1;

    if (activation_type < 0 || activation_type > 6)
        return -1;

    if ((activation_type == 2 && activation_params.w < 1) || ((activation_type == 3 || activation_type == 6) && activation_params.w < 2))
        return -1;

    if (num_output <= 0 || weight_data_size <= 0 || weight_data_size % num_output != 0)
        return -1;
}

    // convert integer text arrays without modifying the shared data
    if (activation_params_type == 5 && !activation_params.empty())
    {
        MatT converted(activation_params.w);
        if (converted.empty())
            return -100;

        const std::int32_t* p = activation_params;
        for (std::int32_t i = 0; i < activation_params.w; i++)
            converted[i] = (float)p[i];

        activation_params = converted;
    }

    if (int8_scale_term)
    {
        if constexpr (Int8)
        {
            this->support_int8_storage = true;
        }
        else
        {
            return -1;
        }
    }

    return 0;
}

MLAYER_INNERPRODUCT_TEMPLATE
std::int32_t InnerProduct<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8>::load_model(const ModelBinT& mb)
{
    weight_data = mb.load(weight_data_size, 0);
    if (weight_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
            return -100;
    }

if constexpr (Int8) {
    if (int8_scale_term)
    {
        weight_data_int8_scales = mb.load(num_output, 1);
        bottom_blob_int8_scales = mb.load(1, 1);
    }
}

if constexpr (Int8) {
    // runtime quantize the weight data
    if (weight_data.elemsize == (std::size_t)4u && int8_scale_term)
    {
        const std::int32_t num_input = weight_data_size / num_output;

        MatT weight_data_r2 = weight_data.reshape(num_input, num_output);

        MatT weight_data_int8;
        OptionT opt_q;
        opt_q.num_threads = 1;
        opt_q.use_packing_layout = false;
        QuantizeInt8(weight_data_r2, weight_data_int8, weight_data_int8_scales, opt_q);
        if (weight_data_int8.empty())
            return -100;

        weight_data = weight_data_int8.reshape(weight_data_size);
    }
}

    return 0;
}

MLAYER_INNERPRODUCT_TEMPLATE
std::int32_t InnerProduct<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8>::forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const
{
    if constexpr (Int8)
    {
        // int8 weights can only be interpreted by the int8 kernel; the
        // use_int8_inference flag is not required for them.
        if (weight_data.elemsize == (std::size_t)1u)
        {
            return forward_int8(bottom_blob, top_blob, opt);
        }
    }

    const std::int32_t num_input = weight_data_size / num_output;

    std::int32_t w = bottom_blob.w;
    std::int32_t h = bottom_blob.h;
    std::int32_t d = bottom_blob.d;
    std::int32_t channels = bottom_blob.c;
    std::size_t elemsize = bottom_blob.elemsize;
    std::int32_t size = w * h * d;

    if (bottom_blob.dims == 2 && w == num_input)
    {
        // gemm
        top_blob.create(num_output, h, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t j = 0; j < h; j++)
        {
            const float* m = bottom_blob.row(j);
            float* outptr = top_blob.row(j);

            for (std::int32_t p = 0; p < num_output; p++)
            {
                const float* kptr = (const float*)weight_data + w * p;

                float sum = 0.f;

                if (bias_term)
                    sum = bias_data[p];

                for (std::int32_t i = 0; i < w; i++)
                {
                    sum += m[i] * kptr[i];
                }

                outptr[p] = Activation(sum, activation_type, activation_params);
            }
        }

        return 0;
    }

    top_blob.create(num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t p = 0; p < num_output; p++)
    {
        float sum = 0.f;

        if (bias_term)
            sum = bias_data[p];

        // channels
        for (std::int32_t q = 0; q < channels; q++)
        {
            const float* w = (const float*)weight_data + size * channels * p + size * q;
            const float* m = bottom_blob.channel(q);

            for (std::int32_t i = 0; i < size; i++)
            {
                sum += m[i] * w[i];
            }
        }

        top_blob[p] = Activation(sum, activation_type, activation_params);
    }

    return 0;
}

MLAYER_INNERPRODUCT_TEMPLATE
std::int32_t InnerProduct<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Validate, Int8, Activation, QuantizeInt8>::forward_int8(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const
{
    const std::int32_t num_input = weight_data_size / num_output;

    std::int32_t w = bottom_blob.w;
    std::int32_t h = bottom_blob.h;
    std::int32_t d = bottom_blob.d;
    std::int32_t channels = bottom_blob.c;
    std::size_t elemsize = bottom_blob.elemsize;
    std::int32_t size = w * h * d;

    MatT bottom_blob_int8 = bottom_blob;
    if (elemsize != 1)
    {
        OptionT opt_g = opt;
        opt_g.blob_allocator = opt.workspace_allocator;
        opt_g.use_packing_layout = false;

        QuantizeInt8(bottom_blob, bottom_blob_int8, bottom_blob_int8_scales, opt_g);
        if (bottom_blob_int8.empty())
            return -100;
    }

    if (bottom_blob.dims == 2 && w == num_input)
    {
        // gemm
        top_blob.create(num_output, h, 4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t j = 0; j < h; j++)
        {
            const std::int8_t* m = bottom_blob_int8.template row<std::int8_t>(j);
            float* outptr = top_blob.row(j);

            for (std::int32_t p = 0; p < num_output; p++)
            {
                const std::int8_t* kptr = (const std::int8_t*)weight_data + w * p;
                std::int32_t sum = 0;

                for (std::int32_t i = 0; i < w; i++)
                {
                    sum += m[i] * kptr[i];
                }
                // dequantize and relu
                float scale_in;
                if (weight_data_int8_scales[p] == 0)
                    scale_in = 0;
                else
                    scale_in = 1.f / (bottom_blob_int8_scales[0] * weight_data_int8_scales[p]);

                float sumfp32 = sum * scale_in;

                if (bias_term)
                    sumfp32 += bias_data[p];

                outptr[p] = Activation(sumfp32, activation_type, activation_params);
            }
        }

        return 0;
    }

    top_blob.create(num_output, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t p = 0; p < num_output; p++)
    {
        float* outptr = top_blob;

        std::int32_t sum = 0;

        std::int32_t offset = size * channels * p;
        // channels
        for (std::int32_t q = 0; q < channels; q++)
        {
            const std::int8_t* w = (const std::int8_t*)weight_data + offset + size * q;
            const std::int8_t* m = bottom_blob_int8.channel(q);

            for (std::int32_t i = 0; i < size; i++)
            {
                sum += m[i] * w[i];
            }
        }

        // dequantize and relu
        float scale_in;
        if (weight_data_int8_scales[p] == 0)
            scale_in = 0;
        else
            scale_in = 1.f / (bottom_blob_int8_scales[0] * weight_data_int8_scales[p]);

        float sumfp32 = sum * scale_in;

        if (bias_term)
            sumfp32 += bias_data[p];

        outptr[p] = Activation(sumfp32, activation_type, activation_params);
    }

    return 0;
}


#undef MLAYER_INNERPRODUCT_TEMPLATE

}  // namespace mlayer

#endif  // MLAYER_INNERPRODUCT_HPP
