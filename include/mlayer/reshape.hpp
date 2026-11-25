// Copyright 2017 Tencent
// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MLAYER_RESHAPE_HPP
#define MLAYER_RESHAPE_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace mlayer {

// Reshape with numeric params or a simple comma-separated shape expression.
// Supported expression tokens: integer literals, "<index><w|h|d|c>" references
// such as "1w", -1 for the remaining dimension and 0 for the same input axis.
template<typename LayerT, typename MatT, typename OptionT, typename ParamDictT>
class Reshape : public LayerT {
public:
    Reshape() {
        this->one_blob_only = true;
        this->support_inplace = false;
        this->w = -233;
        this->h = -233;
        this->d = -233;
        this->c = -233;
        this->ndim = 0;
    }

    std::int32_t load_param(const ParamDictT& pd) override {
        this->w = pd.get(0, -233);
        this->h = pd.get(1, -233);
        this->d = pd.get(11, -233);
        this->c = pd.get(2, -233);
        this->shape_expr = pd.get(6, std::string());

        this->ndim = 0;
        const std::int32_t values[4] = {this->w, this->h, this->d, this->c};
        for (std::int32_t v : values) {
            if (v != -233) {
                ++this->ndim;
            }
        }
        if (!this->shape_expr.empty()) {
            this->one_blob_only = shape_expr_blob_count(this->shape_expr) <= 1;
        }
        return 0;
    }

    std::int32_t forward(const MatT& bottom_blob, MatT& top_blob, const OptionT& opt) const override {
        std::vector<MatT> bottoms(1);
        bottoms[0] = bottom_blob;
        std::vector<MatT> tops(1);
        const std::int32_t ret = this->forward(bottoms, tops, opt);
        top_blob = tops[0];
        return ret;
    }

    std::int32_t forward(const std::vector<MatT>& bottom_blobs, std::vector<MatT>& top_blobs,
                         const OptionT& opt) const override {
        if (bottom_blobs.empty()) {
            return -1;
        }
        const MatT& bottom = bottom_blobs[0];
        MatT& top = top_blobs[0];

        std::int32_t in[4] = {1, 1, 1, 1};
        if (bottom.dims == 1) {
            in[0] = bottom.w;
        } else if (bottom.dims == 2) {
            in[0] = bottom.w;
            in[1] = bottom.h;
        } else if (bottom.dims == 3) {
            in[0] = bottom.w;
            in[1] = bottom.h;
            in[3] = bottom.c;
        } else {
            in[0] = bottom.w;
            in[1] = bottom.h;
            in[2] = bottom.d;
            in[3] = bottom.c;
        }

        std::vector<std::int32_t> shape;
        if (!this->shape_expr.empty()) {
            if (parse_expr(this->shape_expr, bottom_blobs, shape) != 0) {
                return -1;
            }
        } else {
            const std::int32_t values[4] = {this->w, this->h, this->d, this->c};
            for (std::int32_t v : values) {
                if (v != -233) {
                    shape.push_back(v);
                }
            }
        }
        if (shape.empty() || shape.size() > 4) {
            return -1;
        }

        std::int32_t total = 1;
        for (std::int32_t v : in) {
            total *= v;
        }

        std::int32_t known = 1;
        std::int32_t remaining_index = -1;
        for (std::size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] == 0) {
                shape[i] = in[i];
            }
            if (shape[i] == -1) {
                if (remaining_index >= 0) {
                    return -1;
                }
                remaining_index = static_cast<std::int32_t>(i);
                continue;
            }
            if (shape[i] <= 0) {
                return -1;
            }
            known *= shape[i];
        }
        if (remaining_index >= 0) {
            if (known == 0 || total % known != 0) {
                return -1;
            }
            shape[static_cast<std::size_t>(remaining_index)] = total / known;
        } else if (known != total) {
            return -1;
        }

        if (shape.size() == 1) {
            top = bottom.reshape(shape[0], opt.blob_allocator);
        } else if (shape.size() == 2) {
            top = bottom.reshape(shape[0], shape[1], opt.blob_allocator);
        } else if (shape.size() == 3) {
            top = bottom.reshape(shape[0], shape[1], shape[2], opt.blob_allocator);
        } else {
            top = bottom.reshape(shape[0], shape[1], shape[2], shape[3], opt.blob_allocator);
        }
        return top.empty() ? -100 : 0;
    }

public:
    std::int32_t w;
    std::int32_t h;
    std::int32_t d;
    std::int32_t c;
    std::int32_t ndim;
    std::string shape_expr;

private:
    static std::int32_t shape_expr_blob_count(const std::string& expr) {
        std::int32_t count = 0;
        for (std::size_t i = 0; i + 1 < expr.size(); ++i) {
            if (expr[i] >= '0' && expr[i] <= '9' && is_dim(expr[i + 1])) {
                const std::int32_t index = expr[i] - '0';
                if (index + 1 > count) {
                    count = index + 1;
                }
            }
        }
        return count;
    }

    static bool is_dim(char c) {
        return c == 'w' || c == 'h' || c == 'd' || c == 'c';
    }

    static std::int32_t parse_expr(const std::string& expr, const std::vector<MatT>& blobs,
                                   std::vector<std::int32_t>& shape) {
        std::size_t start = 0;
        while (start <= expr.size()) {
            const std::size_t comma = expr.find(',', start);
            const std::string token = expr.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
            if (token.empty()) {
                return -1;
            }

            std::int32_t value = 0;
            bool parsed = false;
            if (token.size() >= 2 && token[0] >= '0' && token[0] <= '9' && is_dim(token[1])) {
                const std::int32_t index = token[0] - '0';
                if (index < 0 || static_cast<std::size_t>(index) >= blobs.size()) {
                    return -1;
                }
                const MatT& b = blobs[static_cast<std::size_t>(index)];
                const char dim = token[1];
                if (dim == 'w') {
                    value = b.w;
                } else if (dim == 'h') {
                    value = b.h;
                } else if (dim == 'd') {
                    value = b.d;
                } else {
                    value = b.c;
                }
                parsed = true;
            } else {
                std::size_t pos = 0;
                bool negative = false;
                if (token[pos] == '-') {
                    negative = true;
                    ++pos;
                }
                if (pos >= token.size()) {
                    return -1;
                }
                std::int32_t magnitude = 0;
                for (; pos < token.size(); ++pos) {
                    if (token[pos] < '0' || token[pos] > '9') {
                        return -1;
                    }
                    magnitude = magnitude * 10 + (token[pos] - '0');
                }
                value = negative ? -magnitude : magnitude;
                parsed = true;
            }
            if (!parsed) {
                return -1;
            }
            shape.push_back(value);

            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
        }
        return 0;
    }
};

}  // namespace mlayer

#endif  // MLAYER_RESHAPE_HPP
