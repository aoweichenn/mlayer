// Copyright 2022 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_EINSUM_HPP
#define MLAYER_EINSUM_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mlayer {

template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT, bool Validate = false>
class Einsum : public LayerT {
public:
    Einsum() {
        this->one_blob_only = false;
        this->support_inplace = false;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        const MatT equation_mat = pd.get(0, MatT());

        if constexpr (Validate) {
            if (equation_mat.empty()) {
                return -1;
            }
        }

        const auto* p = static_cast<const std::int32_t*>(equation_mat);
        std::string equation;
        equation.resize(static_cast<std::size_t>(equation_mat.w));
        for (std::int32_t i = 0; i < equation_mat.w; ++i) {
            if constexpr (Validate) {
                if ((p[i] < 'i' || p[i] > 'x') && p[i] != ',' && p[i] != '-' && p[i] != '>') {
                    return -1;
                }
            }
            equation[static_cast<std::size_t>(i)] = static_cast<char>(p[i]);
        }

        if (equation == "ii") {
            this->lhs_tokens_.clear();
            this->rhs_token_ = "ii";
            return 0;
        }

        std::vector<std::string> tokens;
        std::string token;
        std::int32_t arrow = -1;
        for (std::int32_t i = 0; i < equation_mat.w; ++i) {
            const char ch = equation[static_cast<std::size_t>(i)];
            if (ch == ',' || ch == '-') {
                if constexpr (Validate) {
                    if (token.empty() || token.size() > 4) {
                        return -1;
                    }
                }
                tokens.push_back(token);
                token.clear();
                if (ch == '-') {
                    if constexpr (Validate) {
                        if (i + 1 >= equation_mat.w || equation[static_cast<std::size_t>(i + 1)] != '>') {
                            return -1;
                        }
                    }
                    arrow = i;
                    break;
                }
            } else {
                if constexpr (Validate) {
                    if (ch < 'i' || ch > 'x') {
                        return -1;
                    }
                }
                token.push_back(ch);
            }
        }

        if constexpr (Validate) {
            if (arrow < 0) {
                return -1;
            }
        }

        const std::int32_t output_dims = equation_mat.w - arrow - 2;
        if constexpr (Validate) {
            if (output_dims < 1 || output_dims > 4) {
                return -1;
            }
        }

        std::string output;
        output.resize(static_cast<std::size_t>(output_dims));
        for (std::int32_t i = 0; i < output_dims; ++i) {
            if constexpr (Validate) {
                if (equation[static_cast<std::size_t>(arrow + 2 + i)] != static_cast<char>('i' + i)) {
                    return -1;
                }
            }
            output[static_cast<std::size_t>(i)] = equation[static_cast<std::size_t>(arrow + 2 + i)];
        }

        this->lhs_tokens_ = tokens;
        this->rhs_token_ = output;
        return 0;
    }

    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override {
        const std::size_t elemsize = bottom_blobs[0].elemsize;

        if (this->lhs_tokens_.empty() && this->rhs_token_ == "ii") {
            MatT& top_blob = top_blobs[0];
            top_blob.create(1, elemsize, opt.blob_allocator);
            if (top_blob.empty()) {
                return -100;
            }

            const MatT& bottom_blob = bottom_blobs[0];
            float sum = 0.0F;
            for (std::int32_t i = 0; i < bottom_blob.h; ++i) {
                sum += bottom_blob.row(i)[i];
            }
            top_blob[0] = sum;
            return 0;
        }

        std::vector<std::int32_t> dim_sizes(16, 1);
        std::int32_t dim_sizes_count = 0;

        for (std::size_t b = 0; b < bottom_blobs.size(); ++b) {
            const std::string& lhs_token = this->lhs_tokens_[b];
            const MatT& bottom_blob = bottom_blobs[b];
            const std::int32_t in_dims = bottom_blob.dims;

            for (std::int32_t s = 0; s < in_dims; ++s) {
                std::int32_t dim_size = 1;
                if (in_dims == 1) {
                    dim_size = bottom_blob.w;
                }
                if (in_dims == 2 && s == 0) {
                    dim_size = bottom_blob.h;
                }
                if (in_dims == 2 && s == 1) {
                    dim_size = bottom_blob.w;
                }
                if (in_dims == 3 && s == 0) {
                    dim_size = bottom_blob.c;
                }
                if (in_dims == 3 && s == 1) {
                    dim_size = bottom_blob.h;
                }
                if (in_dims == 3 && s == 2) {
                    dim_size = bottom_blob.w;
                }
                if (in_dims == 4 && s == 0) {
                    dim_size = bottom_blob.c;
                }
                if (in_dims == 4 && s == 1) {
                    dim_size = bottom_blob.d;
                }
                if (in_dims == 4 && s == 2) {
                    dim_size = bottom_blob.h;
                }
                if (in_dims == 4 && s == 3) {
                    dim_size = bottom_blob.w;
                }

                const std::int32_t dim_sizes_index = lhs_token[static_cast<std::size_t>(s)] - 'i';
                dim_sizes[static_cast<std::size_t>(dim_sizes_index)] = dim_size;
                dim_sizes_count = std::max(dim_sizes_count, dim_sizes_index + 1);
            }
        }

        dim_sizes.resize(static_cast<std::size_t>(dim_sizes_count));

        const std::int32_t out_dims = static_cast<std::int32_t>(this->rhs_token_.size());
        std::vector<std::int32_t> indexes(static_cast<std::size_t>(dim_sizes_count), 0);

        if (out_dims == 1) {
            MatT& top_blob = top_blobs[0];
            top_blob.create(dim_sizes[0], elemsize, opt.blob_allocator);
            if (top_blob.empty()) {
                return -100;
            }
            for (std::int32_t i = 0; i < top_blob.w; ++i) {
                indexes[0] = i;
                top_blob[static_cast<std::size_t>(i)] = this->sum_dim(dim_sizes, 1, bottom_blobs, indexes);
            }
        }
        if (out_dims == 2) {
            MatT& top_blob = top_blobs[0];
            top_blob.create(dim_sizes[1], dim_sizes[0], elemsize, opt.blob_allocator);
            if (top_blob.empty()) {
                return -100;
            }
            for (std::int32_t i = 0; i < top_blob.h; ++i) {
                indexes[0] = i;
                for (std::int32_t j = 0; j < top_blob.w; ++j) {
                    indexes[1] = j;
                    top_blob.row(i)[j] = this->sum_dim(dim_sizes, 2, bottom_blobs, indexes);
                }
            }
        }
        if (out_dims == 3) {
            MatT& top_blob = top_blobs[0];
            top_blob.create(dim_sizes[2], dim_sizes[1], dim_sizes[0], elemsize, opt.blob_allocator);
            if (top_blob.empty()) {
                return -100;
            }
            for (std::int32_t i = 0; i < top_blob.c; ++i) {
                indexes[0] = i;
                for (std::int32_t j = 0; j < top_blob.h; ++j) {
                    indexes[1] = j;
                    for (std::int32_t k = 0; k < top_blob.w; ++k) {
                        indexes[2] = k;
                        top_blob.channel(i).row(j)[k] = this->sum_dim(dim_sizes, 3, bottom_blobs, indexes);
                    }
                }
            }
        }
        if (out_dims == 4) {
            MatT& top_blob = top_blobs[0];
            top_blob.create(dim_sizes[3], dim_sizes[2], dim_sizes[1], dim_sizes[0], elemsize, opt.blob_allocator);
            if (top_blob.empty()) {
                return -100;
            }
            for (std::int32_t i = 0; i < top_blob.c; ++i) {
                indexes[0] = i;
                for (std::int32_t j = 0; j < top_blob.d; ++j) {
                    indexes[1] = j;
                    for (std::int32_t k = 0; k < top_blob.h; ++k) {
                        indexes[2] = k;
                        for (std::int32_t l = 0; l < top_blob.w; ++l) {
                            indexes[3] = l;
                            top_blob.channel(i).depth(j).row(k)[l] =
                                this->sum_dim(dim_sizes, 4, bottom_blobs, indexes);
                        }
                    }
                }
            }
        }

        return 0;
    }

private:
    static float get_indexed_value(const MatT& m, const std::string& token,
                                   const std::vector<std::int32_t>& indexes) {
        const std::int32_t dims = m.dims;

        if (dims == 1) {
            const std::int32_t x = indexes[static_cast<std::size_t>(token[0] - 'i')];
            return m[static_cast<std::size_t>(x)];
        }
        if (dims == 2) {
            const std::int32_t y = indexes[static_cast<std::size_t>(token[0] - 'i')];
            const std::int32_t x = indexes[static_cast<std::size_t>(token[1] - 'i')];
            return m.row(y)[x];
        }
        if (dims == 3) {
            const std::int32_t c = indexes[static_cast<std::size_t>(token[0] - 'i')];
            const std::int32_t y = indexes[static_cast<std::size_t>(token[1] - 'i')];
            const std::int32_t x = indexes[static_cast<std::size_t>(token[2] - 'i')];
            return m.channel(c).row(y)[x];
        }
        if (dims == 4) {
            const std::int32_t c = indexes[static_cast<std::size_t>(token[0] - 'i')];
            const std::int32_t z = indexes[static_cast<std::size_t>(token[1] - 'i')];
            const std::int32_t y = indexes[static_cast<std::size_t>(token[2] - 'i')];
            const std::int32_t x = indexes[static_cast<std::size_t>(token[3] - 'i')];
            return m.channel(c).depth(z).row(y)[x];
        }
        return 0.0F;
    }

    float sum_dim(const std::vector<std::int32_t>& dim_sizes, std::int32_t d,
                  const std::vector<MatT>& bottom_blobs, std::vector<std::int32_t>& indexes) const {
        if (d == static_cast<std::int32_t>(dim_sizes.size())) {
            float v = 1.0F;
            for (std::size_t b = 0; b < bottom_blobs.size(); ++b) {
                v *= get_indexed_value(bottom_blobs[b], this->lhs_tokens_[b], indexes);
            }
            return v;
        }

        float sum = 0.0F;
        for (std::int32_t i = 0; i < dim_sizes[static_cast<std::size_t>(d)]; ++i) {
            indexes[static_cast<std::size_t>(d)] = i;
            sum += this->sum_dim(dim_sizes, d + 1, bottom_blobs, indexes);
        }
        return sum;
    }

    std::vector<std::string> lhs_tokens_;
    std::string rhs_token_;
};

}  // namespace mlayer

#endif  // MLAYER_EINSUM_HPP
