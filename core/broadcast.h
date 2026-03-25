#ifndef BROADCAST_H
#define BROADCAST_H

#include <vector>
#include <cstddef>

std::vector<size_t> broadcast_shape(const std::vector<size_t>& a,
                                    const std::vector<size_t>& b);

bool is_broadcastable(const std::vector<size_t>& a,
                      const std::vector<size_t>& b);

size_t broadcast_index(const std::vector<size_t>& idx,
                       const std::vector<size_t>& src_shape,
                       const std::vector<size_t>& src_strides,
                       size_t ndim);

std::vector<size_t> compute_strides(const std::vector<size_t>& shape);

#endif
