#ifndef BROADCAST_H
#define BROADCAST_H

#include <vector>
#include <cstddef>

// Compute broadcast-compatible output shape from two input shapes.
std::vector<size_t> broadcast_shape(const std::vector<size_t>& a,
                                    const std::vector<size_t>& b);

// Check if two shapes are broadcastable.
bool is_broadcastable(const std::vector<size_t>& a,
                      const std::vector<size_t>& b);

// Compute linear index in source tensor given multi-dim index in broadcast result.
size_t broadcast_index(const std::vector<size_t>& idx,
                       const std::vector<size_t>& src_shape,
                       const std::vector<size_t>& src_strides,
                       size_t ndim);

// Compute row-major strides for a shape.
std::vector<size_t> compute_strides(const std::vector<size_t>& shape);

#endif
