// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_GATHERND_HPP
#define MLAYER_GATHERND_HPP

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mlayer {

// ONNX GatherND for the duration-expansion pattern used by VITS/Piper.
// Indices are (w = k, h = number of index tuples). Supports k == data rank
// (scalar result per tuple) and k == rank - 1 (copy the last data axis).
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class GatherND : public LayerT {
public:
    GatherND() {
        this->one_blob_only = false;
        this->support_inplace = false;
    }

    std::int32_t load_param(const ParamDictT& /*pd*/) override { return 0; }

    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override {
        if (bottom_blobs.size() < 2 || top_blobs.empty()) {
            return -100;
        }
        const MatT& data = bottom_blobs[0];
        const MatT& indices = bottom_blobs[1];
        const std::int32_t k = indices.w;
        const std::int32_t count = indices.h;
        const std::int32_t rank = data.dims == 1 ? 1 : (data.dims == 2 ? 2 : (data.dims == 3 ? 3 : 4));
        if (k != rank && k != rank - 1) {
            return -1;
        }

        MatT& top_blob = top_blobs[0];
        const bool copy_tail = k == rank - 1;
        if (copy_tail) {
            top_blob.create(data.w, count, 4u, opt.blob_allocator);
        } else {
            top_blob.create(count, 4u, opt.blob_allocator);
        }
        if (top_blob.empty() && count > 0) {
            return -100;
        }

        for (std::int32_t i = 0; i < count; ++i) {
            const float* index_row = indices.row(i);
            std::vector<std::int32_t> coordinate(static_cast<std::size_t>(k));
            for (std::int32_t j = 0; j < k; ++j) {
                coordinate[static_cast<std::size_t>(j)] = static_cast<std::int32_t>(std::lround(index_row[j]));
            }
            std::int32_t c = 0;
            std::int32_t d = 0;
            std::int32_t h = 0;
            std::int32_t w = 0;
            for (std::int32_t axis = 0; axis < k; ++axis) {
                const auto value = coordinate[static_cast<std::size_t>(axis)];
                if (rank == 1) {
                    w = value;
                } else if (rank == 2) {
                    if (axis == 0) {
                        h = value;
                    } else {
                        w = value;
                    }
                } else if (rank == 3) {
                    if (axis == 0) {
                        c = value;
                    } else if (axis == 1) {
                        h = value;
                    } else {
                        w = value;
                    }
                } else {
                    if (axis == 0) {
                        c = value;
                    } else if (axis == 1) {
                        d = value;
                    } else if (axis == 2) {
                        h = value;
                    } else {
                        w = value;
                    }
                }
            }
            const float* source = nullptr;
            if (rank == 1) {
                source = static_cast<const float*>(data) + w;
            } else if (rank == 2) {
                source = data.row(h) + w;
            } else if (rank == 3) {
                source = data.channel(c).row(h) + w;
            } else {
                source = data.channel(c).depth(d).row(h) + w;
            }
            if (copy_tail) {
                float* destination = top_blob.row(i);
                for (std::int32_t x = 0; x < data.w; ++x) {
                    destination[x] = source[x];
                }
            } else {
                top_blob[i] = source[0];
            }
        }
        return 0;
    }
};

}  // namespace mlayer

#endif  // MLAYER_GATHERND_HPP
