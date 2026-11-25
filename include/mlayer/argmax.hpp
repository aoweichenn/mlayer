// Copyright 2019 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_ARGMAX_HPP
#define MLAYER_ARGMAX_HPP

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace mlayer {

// Top-k over the flattened blob: values (optional) and indices.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class ArgMax : public LayerT {
public:
    ArgMax() {
        this->one_blob_only = true;
        this->support_inplace = false;
        this->out_max_val = 0;
        this->topk = 1;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->out_max_val = pd.get(0, 0);
        this->topk = pd.get(1, 1);
        return 0;
    }

    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override {
        const std::int32_t size = static_cast<std::int32_t>(bottom_blob.total());
        if (this->topk <= 0 || this->topk > size) {
            return -1;
        }
        top_blob.create(this->topk, this->out_max_val != 0 ? 2 : 1, 4u, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }

        const float* ptr = bottom_blob;
        std::vector<std::pair<float, std::int32_t>> vec(static_cast<std::size_t>(size));
        for (std::int32_t i = 0; i < size; ++i) {
            vec[static_cast<std::size_t>(i)] = std::make_pair(ptr[i], i);
        }
        std::partial_sort(vec.begin(), vec.begin() + this->topk, vec.end(),
                          std::greater<std::pair<float, std::int32_t>>());

        float* outptr = top_blob;
        if (this->out_max_val != 0) {
            float* valptr = outptr + this->topk;
            for (std::int32_t i = 0; i < this->topk; ++i) {
                outptr[i] = vec[static_cast<std::size_t>(i)].first;
                valptr[i] = static_cast<float>(vec[static_cast<std::size_t>(i)].second);
            }
        } else {
            for (std::int32_t i = 0; i < this->topk; ++i) {
                outptr[i] = static_cast<float>(vec[static_cast<std::size_t>(i)].second);
            }
        }
        return 0;
    }

public:
    std::int32_t out_max_val;
    std::int32_t topk;
};

}  // namespace mlayer

#endif  // MLAYER_ARGMAX_HPP
