// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_SHUFFLECHANNEL_HPP
#define MLAYER_SHUFFLECHANNEL_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate>
class ShuffleChannel : public LayerT {
public:
    ShuffleChannel();

    std::int32_t load_param(const ParamDictT& pd) override;
    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override;

public:
    std::int32_t group;
    std::int32_t reverse;
};

#define MLAYER_SHUFFLECHANNEL_TEMPLATE \
    template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate>

MLAYER_SHUFFLECHANNEL_TEMPLATE
ShuffleChannel<LayerT, MatT, OptionT, ParamDictT, Validate>::ShuffleChannel()
{
    this->one_blob_only = true;
    this->support_inplace = false;
}

MLAYER_SHUFFLECHANNEL_TEMPLATE
std::int32_t ShuffleChannel<LayerT, MatT, OptionT, ParamDictT, Validate>::load_param(const ParamDictT& pd)
{
    group = pd.get(0, 1);
    reverse = pd.get(1, 0);

if constexpr (Validate) {
    if (group <= 0)
        return -1;
}

    return 0;
}

MLAYER_SHUFFLECHANNEL_TEMPLATE
std::int32_t ShuffleChannel<LayerT, MatT, OptionT, ParamDictT, Validate>::forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const
{
    std::int32_t w = bottom_blob.w;
    std::int32_t h = bottom_blob.h;
    std::int32_t d = bottom_blob.d;
    std::int32_t channels = bottom_blob.c;
    std::size_t elemsize = bottom_blob.elemsize;

    if (channels % group != 0)
    {
        // reject invalid group
        return -100;
    }

    std::int32_t _group = reverse ? channels / group : group;
    std::int32_t channels_per_group = channels / _group;

    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const std::size_t feature_sz = (std::size_t)w * h * d * elemsize;
    for (std::int32_t i = 0; i < _group; i++)
    {
        for (std::int32_t j = 0; j < channels_per_group; j++)
        {
            std::int32_t src_q = channels_per_group * i + j;
            std::int32_t dst_q = _group * j + i;
            memcpy(top_blob.channel(dst_q), bottom_blob.channel(src_q), feature_sz);
        }
    }

    return 0;
}


#undef MLAYER_SHUFFLECHANNEL_TEMPLATE

}  // namespace mlayer

#endif  // MLAYER_SHUFFLECHANNEL_HPP
