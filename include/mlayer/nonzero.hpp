// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_NONZERO_HPP
#define MLAYER_NONZERO_HPP

#include <cstdint>
#include <vector>

namespace mlayer {

// ONNX NonZero. The result is stored as (w = number of non-zero elements,
// h = rank), i.e. ONNX [rank, count] in mnet's (w,h) convention. Row r holds
// the coordinate for ONNX axis r.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class NonZero : public LayerT {
public:
    NonZero() {
        this->one_blob_only = true;
        this->support_inplace = false;
    }

    std::int32_t load_param(const ParamDictT& /*pd*/) override { return 0; }

    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override {
        const std::int32_t dims = bottom_blob.dims;
        const std::int32_t rank = dims == 1 ? 1 : (dims == 2 ? 2 : (dims == 3 ? 3 : 4));
        std::vector<std::int32_t> coordinates;
        for (std::int32_t c = 0; c < (dims >= 3 ? bottom_blob.c : 1); ++c) {
            for (std::int32_t d = 0; d < (dims == 4 ? bottom_blob.d : 1); ++d) {
                for (std::int32_t h = 0; h < (dims >= 2 ? bottom_blob.h : 1); ++h) {
                    const float* row = dims == 1 ? static_cast<const float*>(bottom_blob)
                                                 : (dims == 2 ? bottom_blob.row(h)
                                                              : (dims == 3 ? bottom_blob.channel(c).row(h)
                                                                           : bottom_blob.channel(c).depth(d).row(h)));
                    for (std::int32_t w = 0; w < bottom_blob.w; ++w) {
                        if (row[w] == 0.0F) {
                            continue;
                        }
                        if (rank == 1) {
                            coordinates.push_back(w);
                        } else if (rank == 2) {
                            coordinates.push_back(h);
                            coordinates.push_back(w);
                        } else if (rank == 3) {
                            coordinates.push_back(c);
                            coordinates.push_back(h);
                            coordinates.push_back(w);
                        } else {
                            coordinates.push_back(c);
                            coordinates.push_back(d);
                            coordinates.push_back(h);
                            coordinates.push_back(w);
                        }
                    }
                }
            }
        }
        const auto count = static_cast<std::int32_t>(coordinates.size() / static_cast<std::size_t>(rank));
        top_blob.create(count, rank, 4u, opt.blob_allocator);
        if (top_blob.empty() && count > 0) {
            return -100;
        }
        for (std::int32_t r = 0; r < rank; ++r) {
            float* row = top_blob.row(r);
            for (std::int32_t i = 0; i < count; ++i) {
                row[i] = static_cast<float>(coordinates[static_cast<std::size_t>(i) * rank + r]);
            }
        }
        return 0;
    }
};

}  // namespace mlayer

#endif  // MLAYER_NONZERO_HPP
