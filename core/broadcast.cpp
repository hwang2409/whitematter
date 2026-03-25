#include "broadcast.h"
#include <algorithm>
#include <cassert>

std::vector<size_t> broadcast_shape(const std::vector<size_t>& a,
                                    const std::vector<size_t>& b) {
    size_t ndim = std::max(a.size(), b.size());
    std::vector<size_t> result(ndim);

    for (size_t i = 0; i < ndim; i++) {
        size_t dim_a = (i < ndim - a.size()) ? 1 : a[i - (ndim - a.size())];
        size_t dim_b = (i < ndim - b.size()) ? 1 : b[i - (ndim - b.size())];

        if (dim_a == dim_b) {
            result[i] = dim_a;
        } else if (dim_a == 1) {
            result[i] = dim_b;
        } else if (dim_b == 1) {
            result[i] = dim_a;
        } else {
            assert(false && "Shapes are not broadcastable");
        }
    }
    return result;
}

bool is_broadcastable(const std::vector<size_t>& a,
                      const std::vector<size_t>& b) {
    size_t ndim = std::max(a.size(), b.size());
    for (size_t i = 0; i < ndim; i++) {
        size_t dim_a = (i < ndim - a.size()) ? 1 : a[i - (ndim - a.size())];
        size_t dim_b = (i < ndim - b.size()) ? 1 : b[i - (ndim - b.size())];
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return false;
        }
    }
    return true;
}

size_t broadcast_index(const std::vector<size_t>& idx,
                       const std::vector<size_t>& src_shape,
                       const std::vector<size_t>& src_strides,
                       size_t ndim) {
    size_t src_ndim = src_shape.size();
    size_t linear = 0;
    for (size_t i = 0; i < src_ndim; i++) {
        size_t broadcast_dim = ndim - src_ndim + i;
        size_t src_idx = (src_shape[i] == 1) ? 0 : idx[broadcast_dim];
        linear += src_idx * src_strides[i];
    }
    return linear;
}

std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
    if (shape.empty()) return {};
    std::vector<size_t> strides(shape.size());
    strides[shape.size() - 1] = 1;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}
