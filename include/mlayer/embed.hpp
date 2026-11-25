// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_EMBED_HPP
#define MLAYER_EMBED_HPP

#include <cstdint>
#include <cstring>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT, bool Int8>
class Embed : public LayerT {
public:
    Embed();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t load_model(const ModelBinT& mb) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

private:
    void embed(const MatT& bottom_blob, const MatT& weight_data, const MatT& bias_data, MatT& top_blob,
               std::int32_t input_dim, const OptionT& opt) const;
    void embed_int8(const MatT& bottom_blob, const MatT& weight_data, float weight_data_int8_scale,
                    const MatT& bias_data, MatT& top_blob, std::int32_t input_dim, const OptionT& opt) const;

public:
    std::int32_t num_output;
    std::int32_t input_dim;
    std::int32_t bias_term;
    std::int32_t weight_data_size;
    std::int32_t int8_scale_term;
    MatT weight_data;
    MatT bias_data;
    float weight_data_int8_scale;
};

#define MLAYER_EMBED_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, typename ModelBinT, bool Int8>

MLAYER_EMBED_TEMPLATE
Embed<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Int8>::Embed()
{
    this->one_blob_only = true;
    this->support_inplace = false;
}

MLAYER_EMBED_TEMPLATE
std::int32_t Embed<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Int8>::load_param(const ParamDictT& pd)
{
    num_output = pd.get(0, 0);
    input_dim = pd.get(1, 0);
    bias_term = pd.get(2, 0);
    weight_data_size = pd.get(3, 0);
    int8_scale_term = pd.get(18, 0);

    return 0;
}

MLAYER_EMBED_TEMPLATE
std::int32_t Embed<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Int8>::load_model(const ModelBinT& mb)
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

    if constexpr (Int8)
    {
        if (int8_scale_term)
        {
            weight_data_int8_scale = mb.load(1, 1)[0];
        }
    }

    return 0;
}

MLAYER_EMBED_TEMPLATE
void Embed<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Int8>::embed(const MatT& bottom_blob, const MatT& weight_data, const MatT& bias_data, MatT& top_blob, std::int32_t input_dim, const OptionT& opt) const
{
    const std::int32_t num_output = top_blob.w;
    const std::int32_t words = top_blob.h;

    const float* bias_ptr = bias_data;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t q = 0; q < words; q++)
    {
        float* outptr = top_blob.row(q);

        std::int32_t word_index = ((const std::int32_t*)bottom_blob)[q];

        if (word_index < 0)
            word_index = 0;
        if (word_index >= input_dim)
            word_index = input_dim - 1;

        const float* em = (const float*)weight_data + num_output * word_index;

        if (bias_ptr)
        {
            for (std::int32_t p = 0; p < num_output; p++)
            {
                outptr[p] = em[p] + bias_ptr[p];
            }
        }
        else
        {
            memcpy(outptr, em, num_output * sizeof(float));
        }
    }
}

MLAYER_EMBED_TEMPLATE
void Embed<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Int8>::embed_int8(const MatT& bottom_blob, const MatT& weight_data, float weight_data_int8_scale, const MatT& bias_data, MatT& top_blob, std::int32_t input_dim, const OptionT& opt) const
{
    const std::int32_t num_output = top_blob.w;
    const std::int32_t words = top_blob.h;

    const float* bias_ptr = bias_data;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t q = 0; q < words; q++)
    {
        float* outptr = top_blob.row(q);

        std::int32_t word_index = ((const std::int32_t*)bottom_blob)[q];

        if (word_index < 0)
            word_index = 0;
        if (word_index >= input_dim)
            word_index = input_dim - 1;

        const float descale_em = 1.f / weight_data_int8_scale;

        const std::int8_t* em = (const std::int8_t*)weight_data + num_output * word_index;

        if (bias_ptr)
        {
            for (std::int32_t p = 0; p < num_output; p++)
            {
                outptr[p] = em[p] * descale_em + bias_ptr[p];
            }
        }
        else
        {
            for (std::int32_t p = 0; p < num_output; p++)
            {
                outptr[p] = em[p] * descale_em;
            }
        }
    }
}
MLAYER_EMBED_TEMPLATE
std::int32_t Embed<LayerT, MatT, OptionT, ParamDictT, ModelBinT, Int8>::forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const
{
    const std::int32_t words = bottom_blob.w;

    top_blob.create(num_output, words, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if constexpr (Int8)
    {
        if (int8_scale_term)
        {
            embed_int8(bottom_blob, weight_data, weight_data_int8_scale, bias_data, top_blob, input_dim, opt);
            return 0;
        }
    }
    {
        embed(bottom_blob, weight_data, bias_data, top_blob, input_dim, opt);
    }

    return 0;
}


#undef MLAYER_EMBED_TEMPLATE

}  // namespace mlayer

#endif  // MLAYER_EMBED_HPP
