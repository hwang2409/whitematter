#include "tensor.h"
#include "memory_pool.h"
#include "broadcast.h"
#include "ops/simd_ops.h"
#include "ops/matmul_cpu.h"
#include "ops/fp16.h"
#if defined(WHITEMATTER_METAL) && defined(__APPLE__)
#include "metal/metal_backend.h"
#endif
#if defined(WHITEMATTER_CUDA)
#include "cuda/cuda_backend.h"
#include "cuda/cuda_memory.h"
#include "cuda/cuda_tensor_ops.h"
#endif
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cstring>
#include <limits>

static thread_local std::mt19937 rng(42);

Tensor::Tensor() : requires_grad(false), data_size_(0), grad_size_(0) {}

Tensor::~Tensor() {}

Tensor::Tensor(const std::vector<size_t>& shape, bool requires_grad)
    : shape(shape), requires_grad(requires_grad) {
    size_t total = 1;
    for (auto s : shape) total *= s;
    data_size_ = total;
    data_storage_ = MemoryPool::instance().acquire_shared(total);
    if (!data_storage_) throw std::bad_alloc();
    std::fill(data(), data() + total, 0.0f);
    if (requires_grad) {
        grad_size_ = total;
        grad_storage_ = MemoryPool::instance().acquire_shared(total);
        if (!grad_storage_) throw std::bad_alloc();
        std::fill(grad(), grad() + total, 0.0f);
    } else {
        grad_size_ = 0;
    }
}

Tensor::Tensor(const std::vector<float>& data, const std::vector<size_t>& shape, bool requires_grad)
    : shape(shape), requires_grad(requires_grad) {
    data_size_ = data.size();
    data_storage_ = MemoryPool::instance().acquire_shared(data_size_);
    if (!data_storage_) throw std::bad_alloc();
    std::memcpy(data_storage_.get(), data.data(), data_size_ * sizeof(float));
    if (requires_grad) {
        grad_size_ = data_size_;
        grad_storage_ = MemoryPool::instance().acquire_shared(grad_size_);
        if (!grad_storage_) throw std::bad_alloc();
        std::fill(this->grad(), this->grad() + grad_size_, 0.0f);
    } else {
        grad_size_ = 0;
    }
}

Tensor::Tensor(std::shared_ptr<float> data_storage, size_t data_size,
               const std::vector<size_t>& shape, bool requires_grad)
    : shape(shape), requires_grad(requires_grad),
      data_storage_(std::move(data_storage)), data_size_(data_size) {
    if (requires_grad) {
        grad_size_ = data_size_;
        grad_storage_ = MemoryPool::instance().acquire_shared(grad_size_);
        if (!grad_storage_) throw std::bad_alloc();
        std::fill(grad(), grad() + grad_size_, 0.0f);
    } else {
        grad_size_ = 0;
    }
}

TensorPtr Tensor::create(const std::vector<size_t>& shape, bool requires_grad) {
    return std::make_shared<Tensor>(shape, requires_grad);
}

TensorPtr Tensor::create(const std::vector<float>& data, const std::vector<size_t>& shape, bool requires_grad) {
    return std::make_shared<Tensor>(data, shape, requires_grad);
}

TensorPtr Tensor::create_on_device(const std::vector<size_t>& shape, bool requires_grad,
                                     whitematter::DeviceType dev) {
    if (dev == whitematter::DeviceType::CPU) return create(shape, requires_grad);
#if defined(WHITEMATTER_CUDA)
    if (dev == whitematter::DeviceType::CUDA) {
        // Allocate shape-only on CPU, then swap storage to GPU
        auto t = std::make_shared<Tensor>(shape, false);
        auto& pool = whitematter::CUDAMemoryPool::instance();
        t->data_storage_ = pool.acquire_shared(t->data_size_);
        whitematter::CUDABackend::instance().memset_zero(t->data(), t->data_size_);
        t->device = dev;
        t->requires_grad = requires_grad;
        if (requires_grad) {
            t->grad_size_ = t->data_size_;
            t->grad_storage_ = pool.acquire_shared(t->grad_size_);
            whitematter::CUDABackend::instance().memset_zero(t->grad(), t->grad_size_);
        }
        return t;
    }
#endif
    return create(shape, requires_grad);  // fallback
}

TensorPtr Tensor::zeros(const std::vector<size_t>& shape, bool requires_grad) {
    auto t = create(shape, requires_grad);
    std::fill(t->data(), t->data() + t->size(), 0.0f);
    return t;
}

TensorPtr Tensor::ones(const std::vector<size_t>& shape, bool requires_grad) {
    auto t = create(shape, requires_grad);
    std::fill(t->data(), t->data() + t->size(), 1.0f);
    return t;
}

TensorPtr Tensor::randn(const std::vector<size_t>& shape, bool requires_grad) {
    auto t = create(shape, requires_grad);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < t->size(); i++) t->data()[i] = dist(rng);
    return t;
}

TensorPtr Tensor::to(whitematter::DeviceType d) const {
    if (d == device) return const_cast<Tensor*>(this)->shared_from_this();

#if defined(WHITEMATTER_CUDA)
    // GPU -> CPU
    if (device == whitematter::DeviceType::CUDA && d == whitematter::DeviceType::CPU) {
        auto out = create(shape, requires_grad);
        out->device = d;
        whitematter::CUDABackend::instance().memcpy_d2h(out->data(), data(), size());
        if (!grad_empty() && out->grad()) {
            whitematter::CUDABackend::instance().memcpy_d2h(out->grad(), grad(), grad_size_);
        }
        return out;
    }
    // CPU -> GPU
    if (device == whitematter::DeviceType::CPU && d == whitematter::DeviceType::CUDA) {
        auto out = create_on_device(shape, requires_grad, d);
        whitematter::CUDABackend::instance().memcpy_h2d(out->data(), data(), size());
        if (!grad_empty() && out->grad()) {
            whitematter::CUDABackend::instance().memcpy_h2d(out->grad(), grad(), grad_size_);
        }
        return out;
    }
#endif

    // CPU -> CPU (or unknown device types)
    auto out = create(shape, requires_grad);
    out->device = d;
    std::memcpy(out->data(), data(), size() * sizeof(float));
    if (!grad_empty() && out->grad()) {
        std::memcpy(out->grad(), grad(), grad_size_ * sizeof(float));
    }
    return out;
}

void Tensor::to_inplace(whitematter::DeviceType d) {
    if (d == device) return;
#if defined(WHITEMATTER_CUDA)
    if (device == whitematter::DeviceType::CPU && d == whitematter::DeviceType::CUDA) {
        auto& pool = whitematter::CUDAMemoryPool::instance();
        auto gpu_data = pool.acquire_shared(data_size_);
        whitematter::CUDABackend::instance().memcpy_h2d(gpu_data.get(), data(), data_size_);
        data_storage_ = gpu_data;
        if (grad_storage_) {
            auto gpu_grad = pool.acquire_shared(grad_size_);
            whitematter::CUDABackend::instance().memcpy_h2d(gpu_grad.get(), grad(), grad_size_);
            grad_storage_ = gpu_grad;
        }
        device = d;
        return;
    }
    if (device == whitematter::DeviceType::CUDA && d == whitematter::DeviceType::CPU) {
        auto cpu_data = MemoryPool::instance().acquire_shared(data_size_);
        whitematter::CUDABackend::instance().memcpy_d2h(cpu_data.get(), data(), data_size_);
        data_storage_ = cpu_data;
        if (grad_storage_) {
            auto cpu_grad = MemoryPool::instance().acquire_shared(grad_size_);
            whitematter::CUDABackend::instance().memcpy_d2h(cpu_grad.get(), grad(), grad_size_);
            grad_storage_ = cpu_grad;
        }
        device = d;
        return;
    }
#else
    (void)d;
#endif
}

TensorPtr Tensor::xavier(size_t fan_in, size_t fan_out, bool requires_grad) {
    auto t = create({fan_in, fan_out}, requires_grad);
    float std_val = std::sqrt(2.0f / (fan_in + fan_out));
    std::normal_distribution<float> dist(0.0f, std_val);
    for (size_t i = 0; i < t->size(); i++) t->data()[i] = dist(rng);
    return t;
}

TensorPtr Tensor::kaiming_normal(const std::vector<size_t>& shape, size_t fan_in, bool requires_grad) {
    auto t = create(shape, requires_grad);
    float std_val = std::sqrt(2.0f / static_cast<float>(fan_in));
    std::normal_distribution<float> dist(0.0f, std_val);
    for (size_t i = 0; i < t->size(); i++) t->data()[i] = dist(rng);
    return t;
}

TensorPtr Tensor::kaiming_uniform(const std::vector<size_t>& shape, size_t fan_in, bool requires_grad) {
    auto t = create(shape, requires_grad);
    float bound = std::sqrt(6.0f / static_cast<float>(fan_in));
    std::uniform_real_distribution<float> dist(-bound, bound);
    for (size_t i = 0; i < t->size(); i++) t->data()[i] = dist(rng);
    return t;
}

TensorPtr Tensor::xavier_normal(const std::vector<size_t>& shape, size_t fan_in, size_t fan_out, bool requires_grad) {
    auto t = create(shape, requires_grad);
    float std_val = std::sqrt(2.0f / static_cast<float>(fan_in + fan_out));
    std::normal_distribution<float> dist(0.0f, std_val);
    for (size_t i = 0; i < t->size(); i++) t->data()[i] = dist(rng);
    return t;
}

TensorPtr Tensor::xavier_uniform(const std::vector<size_t>& shape, size_t fan_in, size_t fan_out, bool requires_grad) {
    auto t = create(shape, requires_grad);
    float bound = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    std::uniform_real_distribution<float> dist(-bound, bound);
    for (size_t i = 0; i < t->size(); i++) t->data()[i] = dist(rng);
    return t;
}

TensorPtr Tensor::uniform(const std::vector<size_t>& shape, float low, float high, bool requires_grad) {
    auto t = create(shape, requires_grad);
    std::uniform_real_distribution<float> dist(low, high);
    for (size_t i = 0; i < t->size(); i++) t->data()[i] = dist(rng);
    return t;
}

TensorPtr Tensor::concat(const std::vector<TensorPtr>& tensors, int dim) {
    assert(!tensors.empty() && "concat requires at least one tensor");

    int ndim = static_cast<int>(tensors[0]->shape.size());
    if (dim < 0) dim += ndim;
    assert(dim >= 0 && dim < ndim && "Invalid dimension for concat");

    std::vector<size_t> result_shape = tensors[0]->shape;
    size_t concat_size = tensors[0]->shape[dim];

    for (size_t i = 1; i < tensors.size(); i++) {
        assert(tensors[i]->shape.size() == tensors[0]->shape.size() &&
               "All tensors must have same number of dimensions");
        for (int d = 0; d < ndim; d++) {
            if (d == dim) {
                concat_size += tensors[i]->shape[d];
            } else {
                assert(tensors[i]->shape[d] == tensors[0]->shape[d] &&
                       "Tensor shapes must match except in concat dimension");
            }
        }
    }
    result_shape[dim] = concat_size;

    bool track = false;
    for (const auto& t : tensors) {
        if (t->requires_grad) track = true;
    }
    track = track && GradMode::is_enabled();

    auto result = create(result_shape, track);

    std::vector<size_t> strides(ndim);
    strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; d--) {
        strides[d] = strides[d + 1] * result_shape[d + 1];
    }

    size_t offset_in_dim = 0;
    for (size_t t_idx = 0; t_idx < tensors.size(); t_idx++) {
        const auto& t = tensors[t_idx];
        size_t t_dim_size = t->shape[dim];

        std::vector<size_t> t_strides(ndim);
        t_strides[ndim - 1] = 1;
        for (int d = ndim - 2; d >= 0; d--) {
            t_strides[d] = t_strides[d + 1] * t->shape[d + 1];
        }

        for (size_t i = 0; i < t->size(); i++) {
            std::vector<size_t> idx(ndim);
            size_t tmp = i;
            for (int d = 0; d < ndim; d++) {
                idx[d] = tmp / t_strides[d];
                tmp %= t_strides[d];
            }

            idx[dim] += offset_in_dim;

            size_t result_idx = 0;
            for (int d = 0; d < ndim; d++) {
                result_idx += idx[d] * strides[d];
            }

            result->data()[result_idx] = t->data()[i];
        }

        offset_in_dim += t_dim_size;
    }

    if (track) {
        result->parents = tensors;
        std::vector<size_t> dim_offsets;
        size_t off = 0;
        for (const auto& t : tensors) {
            dim_offsets.push_back(off);
            off += t->shape[dim];
        }

        result->grad_fn = [tensors, result, dim, ndim, strides, dim_offsets]() {
            for (size_t t_idx = 0; t_idx < tensors.size(); t_idx++) {
                const auto& t = tensors[t_idx];
                if (!t->requires_grad) continue;

                size_t offset_in_dim = dim_offsets[t_idx];

                std::vector<size_t> t_strides(ndim);
                t_strides[ndim - 1] = 1;
                for (int d = ndim - 2; d >= 0; d--) {
                    t_strides[d] = t_strides[d + 1] * t->shape[d + 1];
                }

                for (size_t i = 0; i < t->size(); i++) {
                    std::vector<size_t> idx(ndim);
                    size_t tmp = i;
                    for (int d = 0; d < ndim; d++) {
                        idx[d] = tmp / t_strides[d];
                        tmp %= t_strides[d];
                    }

                    idx[dim] += offset_in_dim;

                    size_t result_idx = 0;
                    for (int d = 0; d < ndim; d++) {
                        result_idx += idx[d] * strides[d];
                    }

                    t->grad()[i] += result->grad()[result_idx];
                }
            }
        };
    }

    return result;
}

TensorPtr Tensor::stack(const std::vector<TensorPtr>& tensors, int dim) {
    assert(!tensors.empty() && "stack requires at least one tensor");

    int ndim = static_cast<int>(tensors[0]->shape.size());
    if (dim < 0) dim += ndim + 1;
    assert(dim >= 0 && dim <= ndim && "Invalid dimension for stack");

    for (size_t i = 1; i < tensors.size(); i++) {
        assert(tensors[i]->shape == tensors[0]->shape &&
               "All tensors must have the same shape for stack");
    }

    std::vector<size_t> result_shape;
    for (int d = 0; d < ndim; d++) {
        if (d == dim) {
            result_shape.push_back(tensors.size());
        }
        result_shape.push_back(tensors[0]->shape[d]);
    }
    if (dim == ndim) {
        result_shape.push_back(tensors.size());
    }

    bool track = false;
    for (const auto& t : tensors) {
        if (t->requires_grad) track = true;
    }
    track = track && GradMode::is_enabled();

    auto result = create(result_shape, track);

    int new_ndim = ndim + 1;

    std::vector<size_t> strides(new_ndim);
    strides[new_ndim - 1] = 1;
    for (int d = new_ndim - 2; d >= 0; d--) {
        strides[d] = strides[d + 1] * result_shape[d + 1];
    }

    std::vector<size_t> t_strides(ndim);
    if (ndim > 0) {
        t_strides[ndim - 1] = 1;
        for (int d = ndim - 2; d >= 0; d--) {
            t_strides[d] = t_strides[d + 1] * tensors[0]->shape[d + 1];
        }
    }

    for (size_t t_idx = 0; t_idx < tensors.size(); t_idx++) {
        const auto& t = tensors[t_idx];

        for (size_t i = 0; i < t->size(); i++) {
            std::vector<size_t> t_idx_vec(ndim);
            size_t tmp = i;
            for (int d = 0; d < ndim; d++) {
                t_idx_vec[d] = tmp / t_strides[d];
                tmp %= t_strides[d];
            }

            std::vector<size_t> result_idx_vec(new_ndim);
            int t_d = 0;
            for (int d = 0; d < new_ndim; d++) {
                if (d == dim) {
                    result_idx_vec[d] = t_idx;
                } else {
                    result_idx_vec[d] = t_idx_vec[t_d++];
                }
            }

            size_t result_idx = 0;
            for (int d = 0; d < new_ndim; d++) {
                result_idx += result_idx_vec[d] * strides[d];
            }

            result->data()[result_idx] = t->data()[i];
        }
    }

    if (track) {
        result->parents = tensors;

        result->grad_fn = [tensors, result, dim, ndim, new_ndim, strides, t_strides]() {
            for (size_t t_idx = 0; t_idx < tensors.size(); t_idx++) {
                const auto& t = tensors[t_idx];
                if (!t->requires_grad) continue;

                for (size_t i = 0; i < t->size(); i++) {
                    std::vector<size_t> t_idx_vec(ndim);
                    size_t tmp = i;
                    for (int d = 0; d < ndim; d++) {
                        t_idx_vec[d] = tmp / t_strides[d];
                        tmp %= t_strides[d];
                    }

                    std::vector<size_t> result_idx_vec(new_ndim);
                    int t_d = 0;
                    for (int d = 0; d < new_ndim; d++) {
                        if (d == dim) {
                            result_idx_vec[d] = t_idx;
                        } else {
                            result_idx_vec[d] = t_idx_vec[t_d++];
                        }
                    }

                    size_t result_idx = 0;
                    for (int d = 0; d < new_ndim; d++) {
                        result_idx += result_idx_vec[d] * strides[d];
                    }

                    t->grad()[i] += result->grad()[result_idx];
                }
            }
        };
    }

    return result;
}

size_t Tensor::ndim() const { return shape.size(); }

float Tensor::item() const {
    assert(size() == 1);
    return data()[0];
}

void Tensor::zero_grad() {
    if (!grad_empty()) std::fill(grad(), grad() + grad_size(), 0.0f);
}

float& Tensor::operator[](size_t idx) { return data()[idx]; }
float Tensor::operator[](size_t idx) const { return data()[idx]; }

float& Tensor::at(const std::vector<size_t>& indices) {
    size_t idx = 0, stride = 1;
    for (int i = shape.size() - 1; i >= 0; i--) {
        idx += indices[i] * stride;
        stride *= shape[i];
    }
    return data()[idx];
}

float Tensor::at(const std::vector<size_t>& indices) const {
    size_t idx = 0, stride = 1;
    for (int i = shape.size() - 1; i >= 0; i--) {
        idx += indices[i] * stride;
        stride *= shape[i];
    }
    return data()[idx];
}

bool Tensor::should_track_grad() const {
    return requires_grad && GradMode::is_enabled();
}

void Tensor::build_topo(std::vector<Tensor*>& topo, std::vector<Tensor*>& visited) {
    if (std::find(visited.begin(), visited.end(), this) != visited.end()) return;
    visited.push_back(this);
    for (auto& p : parents) {
        p->build_topo(topo, visited);
    }
    topo.push_back(this);
}

void Tensor::backward() {
    assert(size() == 1);
    if (grad_empty()) {
        grad_storage_ = MemoryPool::instance().acquire_shared(1);
        if (!grad_storage_) throw std::bad_alloc();
        grad_size_ = 1;
    }
    grad()[0] = 1.0f;

    std::vector<Tensor*> topo, visited;
    build_topo(topo, visited);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->grad_fn) (*it)->grad_fn();
    }

    for (auto* t : topo) {
        t->grad_fn = nullptr;
        t->parents.clear();
    }
}

TensorPtr Tensor::matmul(const TensorPtr& other) const {
    assert(shape.size() == 2 && other->shape.size() == 2);
    assert(shape[1] == other->shape[0]);

    size_t m = shape[0], k = shape[1], n = other->shape[1];
    bool track = (requires_grad || other->requires_grad) && GradMode::is_enabled();
    auto result = create({m, n}, track);
    result->device = device;

#if defined(WHITEMATTER_METAL) && defined(__APPLE__)
    if (device == whitematter::DeviceType::METAL && other->device == whitematter::DeviceType::METAL &&
        whitematter::metal_backend_available()) {
        whitematter::MetalBackend::instance().matmul(data(), other->data(), result->data(),
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            auto other_ptr = other;
            result->parents = {self_ptr, other_ptr};
            result->grad_fn = [self_ptr, other_ptr, result, m, k, n]() {
                if (self_ptr->requires_grad) {
                    for (size_t i = 0; i < m; i++) {
                        for (size_t l = 0; l < k; l++) {
                            self_ptr->grad()[i * k + l] += simd_dot(
                                &result->grad()[i * n], &other_ptr->data()[l * n], n);
                        }
                    }
                }
                if (other_ptr->requires_grad) {
                    for (size_t l = 0; l < k; l++) {
                        for (size_t j = 0; j < n; j++) {
                            float sum = 0.0f;
                            for (size_t i = 0; i < m; i++) {
                                sum += self_ptr->data()[i * k + l] * result->grad()[i * n + j];
                            }
                            other_ptr->grad()[l * n + j] += sum;
                        }
                    }
                }
            };
        }
        return result;
    }
#endif

#if defined(WHITEMATTER_CUDA)
    if (device == whitematter::DeviceType::CUDA && other->device == whitematter::DeviceType::CUDA &&
        whitematter::cuda_backend_available()) {
        whitematter::CUDABackend::instance().matmul(data(), other->data(), result->data(),
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            auto other_ptr = other;
            result->parents = {self_ptr, other_ptr};
            result->grad_fn = [self_ptr, other_ptr, result, m, k, n]() {
                if (self_ptr->requires_grad) {
                    for (size_t i = 0; i < m; i++) {
                        for (size_t l = 0; l < k; l++) {
                            self_ptr->grad()[i * k + l] += simd_dot(
                                &result->grad()[i * n], &other_ptr->data()[l * n], n);
                        }
                    }
                }
                if (other_ptr->requires_grad) {
                    for (size_t l = 0; l < k; l++) {
                        for (size_t j = 0; j < n; j++) {
                            float sum = 0.0f;
                            for (size_t i = 0; i < m; i++) {
                                sum += self_ptr->data()[i * k + l] * result->grad()[i * n + j];
                            }
                            other_ptr->grad()[l * n + j] += sum;
                        }
                    }
                }
            };
        }
        return result;
    }
#endif

    matmul_blocked(result->data(), data(), other->data(), m, k, n);

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        auto other_ptr = other;
        result->parents = {self_ptr, other_ptr};
        result->grad_fn = [self_ptr, other_ptr, result, m, k, n]() {
            if (self_ptr->requires_grad) {
                for (size_t i = 0; i < m; i++) {
                    for (size_t l = 0; l < k; l++) {
                        self_ptr->grad()[i * k + l] += simd_dot(
                            &result->grad()[i * n], &other_ptr->data()[l * n], n);
                    }
                }
            }
            if (other_ptr->requires_grad) {
                for (size_t l = 0; l < k; l++) {
                    for (size_t j = 0; j < n; j++) {
                        float sum = 0.0f;
                        for (size_t i = 0; i < m; i++) {
                            sum += self_ptr->data()[i * k + l] * result->grad()[i * n + j];
                        }
                        other_ptr->grad()[l * n + j] += sum;
                    }
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::bmm(const TensorPtr& other) const {
    assert(shape.size() == 3 && other->shape.size() == 3 && "bmm requires 3D tensors");
    assert(shape[0] == other->shape[0] && "Batch sizes must match");
    assert(shape[2] == other->shape[1] && "Inner dimensions must match");

    size_t batch = shape[0];
    size_t m = shape[1];
    size_t k = shape[2];
    size_t n = other->shape[2];

    bool track = (requires_grad || other->requires_grad) && GradMode::is_enabled();
    auto result = create({batch, m, n}, track);
    result->device = device;

    size_t a_batch_stride = m * k;
    size_t b_batch_stride = k * n;
    size_t c_batch_stride = m * n;

#if defined(WHITEMATTER_CUDA)
    if (device == whitematter::DeviceType::CUDA && other->device == whitematter::DeviceType::CUDA &&
        whitematter::cuda_backend_available()) {
        whitematter::CUDABackend::instance().bmm(data(), other->data(), result->data(),
            static_cast<int>(batch), static_cast<int>(m), static_cast<int>(k), static_cast<int>(n));
        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            auto other_ptr = other;
            result->parents = {self_ptr, other_ptr};
            result->grad_fn = [self_ptr, other_ptr, result, batch, m, k, n,
                              a_batch_stride, b_batch_stride, c_batch_stride]() {
                for (size_t b_idx = 0; b_idx < batch; b_idx++) {
                    if (self_ptr->requires_grad) {
                        float* a_grad = &self_ptr->grad()[b_idx * a_batch_stride];
                        const float* c_grad = &result->grad()[b_idx * c_batch_stride];
                        const float* b_data = &other_ptr->data()[b_idx * b_batch_stride];
                        for (size_t i = 0; i < m; i++) {
                            for (size_t l = 0; l < k; l++) {
                                float sum = 0.0f;
                                for (size_t j = 0; j < n; j++) {
                                    sum += c_grad[i * n + j] * b_data[l * n + j];
                                }
                                a_grad[i * k + l] += sum;
                            }
                        }
                    }
                    if (other_ptr->requires_grad) {
                        const float* a_data = &self_ptr->data()[b_idx * a_batch_stride];
                        const float* c_grad = &result->grad()[b_idx * c_batch_stride];
                        float* b_grad = &other_ptr->grad()[b_idx * b_batch_stride];
                        for (size_t l = 0; l < k; l++) {
                            for (size_t j = 0; j < n; j++) {
                                float sum = 0.0f;
                                for (size_t i = 0; i < m; i++) {
                                    sum += a_data[i * k + l] * c_grad[i * n + j];
                                }
                                b_grad[l * n + j] += sum;
                            }
                        }
                    }
                }
            };
        }
        return result;
    }
#endif

    for (size_t b_idx = 0; b_idx < batch; b_idx++) {
        const float* a_ptr = &data()[b_idx * a_batch_stride];
        const float* b_ptr = &other->data()[b_idx * b_batch_stride];
        float* c_ptr = &result->data()[b_idx * c_batch_stride];
        matmul_blocked(c_ptr, a_ptr, b_ptr, m, k, n);
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        auto other_ptr = other;
        result->parents = {self_ptr, other_ptr};
        result->grad_fn = [self_ptr, other_ptr, result, batch, m, k, n,
                          a_batch_stride, b_batch_stride, c_batch_stride]() {
            for (size_t b_idx = 0; b_idx < batch; b_idx++) {
                if (self_ptr->requires_grad) {
                    float* a_grad = &self_ptr->grad()[b_idx * a_batch_stride];
                    const float* c_grad = &result->grad()[b_idx * c_batch_stride];
                    const float* b_data = &other_ptr->data()[b_idx * b_batch_stride];

                    for (size_t i = 0; i < m; i++) {
                        for (size_t l = 0; l < k; l++) {
                            float sum = 0.0f;
                            for (size_t j = 0; j < n; j++) {
                                sum += c_grad[i * n + j] * b_data[l * n + j];
                            }
                            a_grad[i * k + l] += sum;
                        }
                    }
                }
                if (other_ptr->requires_grad) {
                    const float* a_data = &self_ptr->data()[b_idx * a_batch_stride];
                    const float* c_grad = &result->grad()[b_idx * c_batch_stride];
                    float* b_grad = &other_ptr->grad()[b_idx * b_batch_stride];

                    for (size_t l = 0; l < k; l++) {
                        for (size_t j = 0; j < n; j++) {
                            float sum = 0.0f;
                            for (size_t i = 0; i < m; i++) {
                                sum += a_data[i * k + l] * c_grad[i * n + j];
                            }
                            b_grad[l * n + j] += sum;
                        }
                    }
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::add(const TensorPtr& other) const {
    assert(is_broadcastable(shape, other->shape) && "Shapes are not broadcastable");

#if defined(WHITEMATTER_CUDA)
    if (device == whitematter::DeviceType::CUDA && other->device == whitematter::DeviceType::CUDA &&
        whitematter::cuda_backend_available() && shape == other->shape) {
        return cuda_ops::add(this, other);
    }
#endif

    bool track = (requires_grad || other->requires_grad) && GradMode::is_enabled();

    if (shape == other->shape) {
        auto result = create(shape, track);
#if defined(WHITEMATTER_METAL) && defined(__APPLE__)
        if (device == whitematter::DeviceType::METAL && other->device == whitematter::DeviceType::METAL &&
            whitematter::metal_backend_available()) {
            whitematter::MetalBackend::instance().elementwise_add(data(), other->data(), result->data(), size());
        } else
#endif
        simd_add(result->data(), data(), other->data(), size());

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            auto other_ptr = other;
            result->parents = {self_ptr, other_ptr};
            result->grad_fn = [self_ptr, other_ptr, result]() {
                if (self_ptr->requires_grad) {
                    simd_add(self_ptr->grad(), self_ptr->grad(),
                             result->grad(), self_ptr->size());
                }
                if (other_ptr->requires_grad) {
                    simd_add(other_ptr->grad(), other_ptr->grad(),
                             result->grad(), other_ptr->size());
                }
            };
        }
        return result;
    }

    auto out_shape = broadcast_shape(shape, other->shape);
    auto result = create(out_shape, track);

    auto a_strides = compute_strides(shape);
    auto b_strides = compute_strides(other->shape);
    auto out_strides = compute_strides(out_shape);
    size_t ndim = out_shape.size();

    std::vector<size_t> idx(ndim, 0);
    for (size_t i = 0; i < result->size(); i++) {
        size_t a_idx = broadcast_index(idx, shape, a_strides, ndim);
        size_t b_idx = broadcast_index(idx, other->shape, b_strides, ndim);
        result->data()[i] = data()[a_idx] + other->data()[b_idx];

        for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
            idx[d]++;
            if (idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        auto other_ptr = other;
        auto self_shape = shape;
        auto other_shape = other->shape;

        result->parents = {self_ptr, other_ptr};
        result->grad_fn = [self_ptr, other_ptr, result, self_shape, other_shape,
                          out_shape, a_strides, b_strides, out_strides, ndim]() {
            std::vector<size_t> idx(ndim, 0);
            for (size_t i = 0; i < result->size(); i++) {
                if (self_ptr->requires_grad) {
                    size_t a_idx = broadcast_index(idx, self_shape, a_strides, ndim);
                    self_ptr->grad()[a_idx] += result->grad()[i];
                }
                if (other_ptr->requires_grad) {
                    size_t b_idx = broadcast_index(idx, other_shape, b_strides, ndim);
                    other_ptr->grad()[b_idx] += result->grad()[i];
                }

                for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
                    idx[d]++;
                    if (idx[d] < out_shape[d]) break;
                    idx[d] = 0;
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::sub(const TensorPtr& other) const {
    assert(is_broadcastable(shape, other->shape) && "Shapes are not broadcastable");

#if defined(WHITEMATTER_CUDA)
    if (device == whitematter::DeviceType::CUDA && other->device == whitematter::DeviceType::CUDA &&
        whitematter::cuda_backend_available() && shape == other->shape) {
        return cuda_ops::sub(this, other);
    }
#endif

    bool track = (requires_grad || other->requires_grad) && GradMode::is_enabled();

    if (shape == other->shape) {
        auto result = create(shape, track);
#if defined(WHITEMATTER_METAL) && defined(__APPLE__)
        if (device == whitematter::DeviceType::METAL && other->device == whitematter::DeviceType::METAL &&
            whitematter::metal_backend_available()) {
            whitematter::MetalBackend::instance().elementwise_sub(data(), other->data(), result->data(), size());
        } else
#endif
        simd_sub(result->data(), data(), other->data(), size());

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            auto other_ptr = other;
            result->parents = {self_ptr, other_ptr};
            result->grad_fn = [self_ptr, other_ptr, result]() {
                if (self_ptr->requires_grad) {
                    simd_add(self_ptr->grad(), self_ptr->grad(),
                             result->grad(), self_ptr->size());
                }
                if (other_ptr->requires_grad) {
                    for (size_t i = 0; i < other_ptr->size(); i++) {
                        other_ptr->grad()[i] -= result->grad()[i];
                    }
                }
            };
        }
        return result;
    }

    auto out_shape = broadcast_shape(shape, other->shape);
    auto result = create(out_shape, track);

    auto a_strides = compute_strides(shape);
    auto b_strides = compute_strides(other->shape);
    auto out_strides = compute_strides(out_shape);
    size_t ndim = out_shape.size();

    std::vector<size_t> idx(ndim, 0);
    for (size_t i = 0; i < result->size(); i++) {
        size_t a_idx = broadcast_index(idx, shape, a_strides, ndim);
        size_t b_idx = broadcast_index(idx, other->shape, b_strides, ndim);
        result->data()[i] = data()[a_idx] - other->data()[b_idx];

        for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
            idx[d]++;
            if (idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        auto other_ptr = other;
        auto self_shape = shape;
        auto other_shape = other->shape;

        result->parents = {self_ptr, other_ptr};
        result->grad_fn = [self_ptr, other_ptr, result, self_shape, other_shape,
                          out_shape, a_strides, b_strides, out_strides, ndim]() {
            std::vector<size_t> idx(ndim, 0);
            for (size_t i = 0; i < result->size(); i++) {
                if (self_ptr->requires_grad) {
                    size_t a_idx = broadcast_index(idx, self_shape, a_strides, ndim);
                    self_ptr->grad()[a_idx] += result->grad()[i];
                }
                if (other_ptr->requires_grad) {
                    size_t b_idx = broadcast_index(idx, other_shape, b_strides, ndim);
                    other_ptr->grad()[b_idx] -= result->grad()[i];
                }

                for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
                    idx[d]++;
                    if (idx[d] < out_shape[d]) break;
                    idx[d] = 0;
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::mul(const TensorPtr& other) const {
    assert(is_broadcastable(shape, other->shape) && "Shapes are not broadcastable");

#if defined(WHITEMATTER_CUDA)
    if (device == whitematter::DeviceType::CUDA && other->device == whitematter::DeviceType::CUDA &&
        whitematter::cuda_backend_available() && shape == other->shape) {
        return cuda_ops::mul(this, other);
    }
#endif

    bool track = (requires_grad || other->requires_grad) && GradMode::is_enabled();

    if (shape == other->shape) {
        auto result = create(shape, track);
#if defined(WHITEMATTER_METAL) && defined(__APPLE__)
        if (device == whitematter::DeviceType::METAL && other->device == whitematter::DeviceType::METAL &&
            whitematter::metal_backend_available()) {
            whitematter::MetalBackend::instance().elementwise_mul(data(), other->data(), result->data(), size());
        } else
#endif
        simd_mul(result->data(), data(), other->data(), size());

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            auto other_ptr = other;
            result->parents = {self_ptr, other_ptr};
            result->grad_fn = [self_ptr, other_ptr, result]() {
                if (self_ptr->requires_grad) {
                    for (size_t i = 0; i < self_ptr->size(); i++) {
                        self_ptr->grad()[i] += result->grad()[i] * other_ptr->data()[i];
                    }
                }
                if (other_ptr->requires_grad) {
                    for (size_t i = 0; i < other_ptr->size(); i++) {
                        other_ptr->grad()[i] += result->grad()[i] * self_ptr->data()[i];
                    }
                }
            };
        }
        return result;
    }

    auto out_shape = broadcast_shape(shape, other->shape);
    auto result = create(out_shape, track);

    auto a_strides = compute_strides(shape);
    auto b_strides = compute_strides(other->shape);
    auto out_strides = compute_strides(out_shape);
    size_t ndim = out_shape.size();

    std::vector<size_t> idx(ndim, 0);
    for (size_t i = 0; i < result->size(); i++) {
        size_t a_idx = broadcast_index(idx, shape, a_strides, ndim);
        size_t b_idx = broadcast_index(idx, other->shape, b_strides, ndim);
        result->data()[i] = data()[a_idx] * other->data()[b_idx];

        for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
            idx[d]++;
            if (idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        auto other_ptr = other;
        auto self_shape = shape;
        auto other_shape = other->shape;

        result->parents = {self_ptr, other_ptr};
        result->grad_fn = [self_ptr, other_ptr, result, self_shape, other_shape,
                          out_shape, a_strides, b_strides, out_strides, ndim]() {
            std::vector<size_t> idx(ndim, 0);
            for (size_t i = 0; i < result->size(); i++) {
                size_t a_idx = broadcast_index(idx, self_shape, a_strides, ndim);
                size_t b_idx = broadcast_index(idx, other_shape, b_strides, ndim);

                if (self_ptr->requires_grad) {
                    self_ptr->grad()[a_idx] += result->grad()[i] * other_ptr->data()[b_idx];
                }
                if (other_ptr->requires_grad) {
                    other_ptr->grad()[b_idx] += result->grad()[i] * self_ptr->data()[a_idx];
                }

                for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
                    idx[d]++;
                    if (idx[d] < out_shape[d]) break;
                    idx[d] = 0;
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::div(const TensorPtr& other) const {
    assert(is_broadcastable(shape, other->shape) && "Shapes are not broadcastable");

#if defined(WHITEMATTER_CUDA)
    if (device == whitematter::DeviceType::CUDA && other->device == whitematter::DeviceType::CUDA &&
        whitematter::cuda_backend_available() && shape == other->shape) {
        return cuda_ops::div(this, other);
    }
#endif

    bool track = (requires_grad || other->requires_grad) && GradMode::is_enabled();

    if (shape == other->shape) {
        auto result = create(shape, track);
        for (size_t i = 0; i < size(); i++) {
            result->data()[i] = data()[i] / other->data()[i];
        }

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            auto other_ptr = other;
            result->parents = {self_ptr, other_ptr};
            result->grad_fn = [self_ptr, other_ptr, result]() {
                for (size_t i = 0; i < self_ptr->size(); i++) {
                    if (self_ptr->requires_grad) {
                        self_ptr->grad()[i] += result->grad()[i] / other_ptr->data()[i];
                    }
                    if (other_ptr->requires_grad) {
                        other_ptr->grad()[i] -= result->grad()[i] * self_ptr->data()[i] /
                                              (other_ptr->data()[i] * other_ptr->data()[i]);
                    }
                }
            };
        }
        return result;
    }

    auto out_shape = broadcast_shape(shape, other->shape);
    auto result = create(out_shape, track);

    auto a_strides = compute_strides(shape);
    auto b_strides = compute_strides(other->shape);
    auto out_strides = compute_strides(out_shape);
    size_t ndim = out_shape.size();

    std::vector<size_t> idx(ndim, 0);
    for (size_t i = 0; i < result->size(); i++) {
        size_t a_idx = broadcast_index(idx, shape, a_strides, ndim);
        size_t b_idx = broadcast_index(idx, other->shape, b_strides, ndim);
        result->data()[i] = data()[a_idx] / other->data()[b_idx];

        for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
            idx[d]++;
            if (idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        auto other_ptr = other;
        auto self_shape = shape;
        auto other_shape = other->shape;

        result->parents = {self_ptr, other_ptr};
        result->grad_fn = [self_ptr, other_ptr, result, self_shape, other_shape,
                          out_shape, a_strides, b_strides, out_strides, ndim]() {
            std::vector<size_t> idx(ndim, 0);
            for (size_t i = 0; i < result->size(); i++) {
                size_t a_idx = broadcast_index(idx, self_shape, a_strides, ndim);
                size_t b_idx = broadcast_index(idx, other_shape, b_strides, ndim);

                if (self_ptr->requires_grad) {
                    self_ptr->grad()[a_idx] += result->grad()[i] / other_ptr->data()[b_idx];
                }
                if (other_ptr->requires_grad) {
                    other_ptr->grad()[b_idx] -= result->grad()[i] * self_ptr->data()[a_idx] /
                                              (other_ptr->data()[b_idx] * other_ptr->data()[b_idx]);
                }

                for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
                    idx[d]++;
                    if (idx[d] < out_shape[d]) break;
                    idx[d] = 0;
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::mul(float scalar) const {
#if defined(WHITEMATTER_CUDA)
    if (device == whitematter::DeviceType::CUDA && whitematter::cuda_backend_available()) {
        return cuda_ops::scalar_mul(this, scalar);
    }
#endif

    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);

    simd_scale(result->data(), data(), scalar, size());

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result, scalar]() {
            for (size_t i = 0; i < self_ptr->size(); i++) {
                self_ptr->grad()[i] += result->grad()[i] * scalar;
            }
        };
    }
    return result;
}

TensorPtr Tensor::div(float scalar) const {
    return mul(1.0f / scalar);
}

TensorPtr Tensor::neg() const {
    return mul(-1.0f);
}

TensorPtr Tensor::relu() const {
#if defined(WHITEMATTER_CUDA)
    if (device == whitematter::DeviceType::CUDA && whitematter::cuda_backend_available()) {
        return cuda_ops::relu(this);
    }
#endif

    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);

#if defined(WHITEMATTER_METAL) && defined(__APPLE__)
    if (device == whitematter::DeviceType::METAL && whitematter::metal_backend_available()) {
        whitematter::MetalBackend::instance().relu(data(), result->data(), size());
    } else
#endif
    simd_relu(result->data(), data(), size());

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result]() {
            for (size_t i = 0; i < self_ptr->size(); i++) {
                self_ptr->grad()[i] += result->grad()[i] * (self_ptr->data()[i] > 0 ? 1.0f : 0.0f);
            }
        };
    }
    return result;
}

TensorPtr Tensor::sigmoid() const {
#if defined(WHITEMATTER_CUDA)
    if (device == whitematter::DeviceType::CUDA && whitematter::cuda_backend_available()) {
        return cuda_ops::sigmoid(this);
    }
#endif

    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);

#if defined(WHITEMATTER_METAL) && defined(__APPLE__)
    if (device == whitematter::DeviceType::METAL && whitematter::metal_backend_available()) {
        whitematter::MetalBackend::instance().sigmoid(data(), result->data(), size());
    } else
#endif
    for (size_t i = 0; i < size(); i++) {
        result->data()[i] = 1.0f / (1.0f + std::exp(-data()[i]));
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result]() {
            for (size_t i = 0; i < self_ptr->size(); i++) {
                float s = result->data()[i];
                self_ptr->grad()[i] += result->grad()[i] * s * (1.0f - s);
            }
        };
    }
    return result;
}

TensorPtr Tensor::tanh_() const {
#if defined(WHITEMATTER_CUDA)
    if (device == whitematter::DeviceType::CUDA && whitematter::cuda_backend_available()) {
        return cuda_ops::tanh_op(this);
    }
#endif

    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);

#if defined(WHITEMATTER_METAL) && defined(__APPLE__)
    if (device == whitematter::DeviceType::METAL && whitematter::metal_backend_available()) {
        whitematter::MetalBackend::instance().tanh_op(data(), result->data(), size());
    } else
#endif
    for (size_t i = 0; i < size(); i++) {
        result->data()[i] = std::tanh(data()[i]);
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result]() {
            for (size_t i = 0; i < self_ptr->size(); i++) {
                float t = result->data()[i];
                self_ptr->grad()[i] += result->grad()[i] * (1.0f - t * t);
            }
        };
    }
    return result;
}

TensorPtr Tensor::silu() const {
#if defined(WHITEMATTER_CUDA)
    if (device == whitematter::DeviceType::CUDA && whitematter::cuda_backend_available()) {
        return cuda_ops::silu(this);
    }
#endif

    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);

    for (size_t i = 0; i < size(); i++) {
        float x = data()[i];
        float s = 1.0f / (1.0f + std::exp(-x));
        result->data()[i] = x * s;
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result]() {
            for (size_t i = 0; i < self_ptr->size(); i++) {
                float x = self_ptr->data()[i];
                float s = 1.0f / (1.0f + std::exp(-x));
                // d/dx[x * sig(x)] = sig(x) * (1 + x * (1 - sig(x)))
                float grad = s * (1.0f + x * (1.0f - s));
                self_ptr->grad()[i] += result->grad()[i] * grad;
            }
        };
    }
    return result;
}

TensorPtr Tensor::gelu() const {
#if defined(WHITEMATTER_CUDA)
    if (device == whitematter::DeviceType::CUDA && whitematter::cuda_backend_available()) {
        return cuda_ops::gelu(this);
    }
#endif

    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);

    static const float sqrt2 = std::sqrt(2.0f);

    for (size_t i = 0; i < size(); i++) {
        float x = data()[i];
        result->data()[i] = x * 0.5f * (1.0f + std::erf(x / sqrt2));
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result]() {
            static const float sqrt2_l = std::sqrt(2.0f);
            static const float sqrt2pi_l = std::sqrt(2.0f * static_cast<float>(M_PI));
            for (size_t i = 0; i < self_ptr->size(); i++) {
                float x = self_ptr->data()[i];
                // d/dx = 0.5 * (1 + erf(x/sqrt(2))) + x * exp(-x^2/2) / sqrt(2*pi)
                float cdf = 0.5f * (1.0f + std::erf(x / sqrt2_l));
                float pdf = std::exp(-0.5f * x * x) / sqrt2pi_l;
                self_ptr->grad()[i] += result->grad()[i] * (cdf + x * pdf);
            }
        };
    }
    return result;
}

TensorPtr Tensor::mish() const {
#if defined(WHITEMATTER_CUDA)
    if (device == whitematter::DeviceType::CUDA && whitematter::cuda_backend_available()) {
        auto r = cuda_ops::mish(this);
        if (r) return r;
        // No CUDA mish kernel; fall through to CPU path
    }
#endif

    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);

    for (size_t i = 0; i < size(); i++) {
        float x = data()[i];
        // softplus(x) = ln(1 + exp(x)), numerically stable
        float sp = (x > 20.0f) ? x : std::log1p(std::exp(x));
        result->data()[i] = x * std::tanh(sp);
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result]() {
            for (size_t i = 0; i < self_ptr->size(); i++) {
                float x = self_ptr->data()[i];
                float sp = (x > 20.0f) ? x : std::log1p(std::exp(x));
                float tanh_sp = std::tanh(sp);
                float sig = 1.0f / (1.0f + std::exp(-x));
                // d/dx = tanh(sp) + x * (1 - tanh(sp)^2) * sigmoid(x)
                float grad = tanh_sp + x * (1.0f - tanh_sp * tanh_sp) * sig;
                self_ptr->grad()[i] += result->grad()[i] * grad;
            }
        };
    }
    return result;
}

TensorPtr Tensor::log_() const {
    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);

    for (size_t i = 0; i < size(); i++) {
        result->data()[i] = std::log(data()[i] + 1e-8f);
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result]() {
            for (size_t i = 0; i < self_ptr->size(); i++) {
                self_ptr->grad()[i] += result->grad()[i] / (self_ptr->data()[i] + 1e-8f);
            }
        };
    }
    return result;
}

TensorPtr Tensor::exp_() const {
    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);

    for (size_t i = 0; i < size(); i++) {
        result->data()[i] = std::exp(data()[i]);
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result]() {
            for (size_t i = 0; i < self_ptr->size(); i++) {
                self_ptr->grad()[i] += result->grad()[i] * result->data()[i];
            }
        };
    }
    return result;
}

TensorPtr Tensor::pow(float exponent) const {
    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);

    for (size_t i = 0; i < size(); i++) {
        result->data()[i] = std::pow(data()[i], exponent);
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result, exponent]() {
            for (size_t i = 0; i < self_ptr->size(); i++) {
                self_ptr->grad()[i] += result->grad()[i] * exponent *
                                     std::pow(self_ptr->data()[i], exponent - 1.0f);
            }
        };
    }
    return result;
}

TensorPtr Tensor::pow(const TensorPtr& exponent) const {
    assert(is_broadcastable(shape, exponent->shape) && "Shapes are not broadcastable");

    bool track = (requires_grad || exponent->requires_grad) && GradMode::is_enabled();

    if (shape == exponent->shape) {
        auto result = create(shape, track);
        for (size_t i = 0; i < size(); i++) {
            result->data()[i] = std::pow(data()[i], exponent->data()[i]);
        }

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            auto exp_ptr = exponent;
            result->parents = {self_ptr, exp_ptr};
            result->grad_fn = [self_ptr, exp_ptr, result]() {
                for (size_t i = 0; i < self_ptr->size(); i++) {
                    if (self_ptr->requires_grad) {
                        self_ptr->grad()[i] += result->grad()[i] * exp_ptr->data()[i] *
                                             std::pow(self_ptr->data()[i], exp_ptr->data()[i] - 1.0f);
                    }
                    if (exp_ptr->requires_grad) {
                        exp_ptr->grad()[i] += result->grad()[i] * result->data()[i] *
                                            std::log(self_ptr->data()[i]);
                    }
                }
            };
        }
        return result;
    }

    auto out_shape = broadcast_shape(shape, exponent->shape);
    auto result = create(out_shape, track);

    auto a_strides = compute_strides(shape);
    auto b_strides = compute_strides(exponent->shape);
    auto out_strides = compute_strides(out_shape);
    size_t ndim = out_shape.size();

    std::vector<size_t> idx(ndim, 0);
    for (size_t i = 0; i < result->size(); i++) {
        size_t a_idx = broadcast_index(idx, shape, a_strides, ndim);
        size_t b_idx = broadcast_index(idx, exponent->shape, b_strides, ndim);
        result->data()[i] = std::pow(data()[a_idx], exponent->data()[b_idx]);

        for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
            idx[d]++;
            if (idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        auto exp_ptr = exponent;
        auto self_shape = shape;
        auto exp_shape = exponent->shape;

        result->parents = {self_ptr, exp_ptr};
        result->grad_fn = [self_ptr, exp_ptr, result, self_shape, exp_shape,
                          out_shape, a_strides, b_strides, out_strides, ndim]() {
            std::vector<size_t> idx(ndim, 0);
            for (size_t i = 0; i < result->size(); i++) {
                size_t a_idx = broadcast_index(idx, self_shape, a_strides, ndim);
                size_t b_idx = broadcast_index(idx, exp_shape, b_strides, ndim);

                if (self_ptr->requires_grad) {
                    self_ptr->grad()[a_idx] += result->grad()[i] * exp_ptr->data()[b_idx] *
                                             std::pow(self_ptr->data()[a_idx], exp_ptr->data()[b_idx] - 1.0f);
                }
                if (exp_ptr->requires_grad) {
                    exp_ptr->grad()[b_idx] += result->grad()[i] * result->data()[i] *
                                            std::log(self_ptr->data()[a_idx]);
                }

                for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
                    idx[d]++;
                    if (idx[d] < out_shape[d]) break;
                    idx[d] = 0;
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::sqrt() const {
    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);

    for (size_t i = 0; i < size(); i++) {
        result->data()[i] = std::sqrt(data()[i]);
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result]() {
            for (size_t i = 0; i < self_ptr->size(); i++) {
                self_ptr->grad()[i] += result->grad()[i] / (2.0f * result->data()[i]);
            }
        };
    }
    return result;
}

TensorPtr Tensor::abs() const {
    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);

    for (size_t i = 0; i < size(); i++) {
        result->data()[i] = std::abs(data()[i]);
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result]() {
            for (size_t i = 0; i < self_ptr->size(); i++) {
                if (self_ptr->data()[i] > 0) {
                    self_ptr->grad()[i] += result->grad()[i];
                } else if (self_ptr->data()[i] < 0) {
                    self_ptr->grad()[i] -= result->grad()[i];
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::clamp(float min_val, float max_val) const {
    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);

    for (size_t i = 0; i < size(); i++) {
        result->data()[i] = std::max(min_val, std::min(max_val, data()[i]));
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result, min_val, max_val]() {
            for (size_t i = 0; i < self_ptr->size(); i++) {
                if (self_ptr->data()[i] > min_val && self_ptr->data()[i] < max_val) {
                    self_ptr->grad()[i] += result->grad()[i];
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::softmax(int dim) const {
#if defined(WHITEMATTER_CUDA)
    if (device == whitematter::DeviceType::CUDA && whitematter::cuda_backend_available()) {
        return cuda_ops::softmax(this, dim);
    }
#endif

    assert(shape.size() == 2);
    if (dim < 0) dim = shape.size() + dim;

    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);
    size_t rows = shape[0], cols = shape[1];

    for (size_t i = 0; i < rows; i++) {
        float max_val = data()[i * cols];
        for (size_t j = 1; j < cols; j++) {
            max_val = std::max(max_val, data()[i * cols + j]);
        }

        float sum = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            result->data()[i * cols + j] = std::exp(data()[i * cols + j] - max_val);
            sum += result->data()[i * cols + j];
        }

        float inv_sum = 1.0f / sum;
        for (size_t j = 0; j < cols; j++) {
            result->data()[i * cols + j] *= inv_sum;
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result, rows, cols]() {
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    float sj = result->data()[i * cols + j];
                    for (size_t k = 0; k < cols; k++) {
                        float sk = result->data()[i * cols + k];
                        float grad = (j == k) ? sj * (1.0f - sj) : -sj * sk;
                        self_ptr->grad()[i * cols + j] += result->grad()[i * cols + k] * grad;
                    }
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::log_softmax(int dim) const {
#if defined(WHITEMATTER_CUDA)
    if (device == whitematter::DeviceType::CUDA && whitematter::cuda_backend_available()) {
        return cuda_ops::log_softmax(this, dim);
    }
#endif

    assert(shape.size() == 2);
    if (dim < 0) dim = shape.size() + dim;

    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);
    size_t rows = shape[0], cols = shape[1];

    for (size_t i = 0; i < rows; i++) {
        float max_val = data()[i * cols];
        for (size_t j = 1; j < cols; j++) {
            max_val = std::max(max_val, data()[i * cols + j]);
        }

        float sum = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            sum += std::exp(data()[i * cols + j] - max_val);
        }
        float log_sum = max_val + std::log(sum);

        for (size_t j = 0; j < cols; j++) {
            result->data()[i * cols + j] = data()[i * cols + j] - log_sum;
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result, rows, cols]() {
            for (size_t i = 0; i < rows; i++) {
                float grad_sum = 0.0f;
                for (size_t j = 0; j < cols; j++) {
                    grad_sum += result->grad()[i * cols + j];
                }
                for (size_t j = 0; j < cols; j++) {
                    float softmax_j = std::exp(result->data()[i * cols + j]);
                    self_ptr->grad()[i * cols + j] += result->grad()[i * cols + j] - softmax_j * grad_sum;
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::sum(int dim, bool keepdim) const {
    if (dim == -1) {
#if defined(WHITEMATTER_CUDA)
        if (device == whitematter::DeviceType::CUDA && whitematter::cuda_backend_available()) {
            return cuda_ops::sum(this);
        }
#endif

        bool track = requires_grad && GradMode::is_enabled();
        auto result = create({1}, track);
        result->data()[0] = 0.0f;
        for (size_t i = 0; i < size(); i++) {
            result->data()[0] += data()[i];
        }

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            result->parents = {self_ptr};
            result->grad_fn = [self_ptr, result]() {
                for (size_t i = 0; i < self_ptr->size(); i++) {
                    self_ptr->grad()[i] += result->grad()[0];
                }
            };
        }
        return result;
    }

    assert(shape.size() == 2);
    size_t rows = shape[0], cols = shape[1];
    bool track = requires_grad && GradMode::is_enabled();

    if (dim == 0) {
        auto result = keepdim ? create({1, cols}, track) : create({cols}, track);
        for (size_t j = 0; j < cols; j++) {
            float sum = 0.0f;
            for (size_t i = 0; i < rows; i++) {
                sum += data()[i * cols + j];
            }
            result->data()[j] = sum;
        }

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            result->parents = {self_ptr};
            result->grad_fn = [self_ptr, result, rows, cols]() {
                for (size_t i = 0; i < rows; i++) {
                    for (size_t j = 0; j < cols; j++) {
                        self_ptr->grad()[i * cols + j] += result->grad()[j];
                    }
                }
            };
        }
        return result;
    } else {
        auto result = keepdim ? create({rows, 1}, track) : create({rows}, track);
        for (size_t i = 0; i < rows; i++) {
            float sum = 0.0f;
            for (size_t j = 0; j < cols; j++) {
                sum += data()[i * cols + j];
            }
            result->data()[i] = sum;
        }

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            result->parents = {self_ptr};
            result->grad_fn = [self_ptr, result, rows, cols]() {
                for (size_t i = 0; i < rows; i++) {
                    for (size_t j = 0; j < cols; j++) {
                        self_ptr->grad()[i * cols + j] += result->grad()[i];
                    }
                }
            };
        }
        return result;
    }
}

TensorPtr Tensor::mean(int dim, bool keepdim) const {
    if (dim == -1) {
        return sum(-1, keepdim)->mul(1.0f / static_cast<float>(size()));
    }

    float n = static_cast<float>(dim == 0 ? shape[0] : shape[1]);
    return sum(dim, keepdim)->mul(1.0f / n);
}

TensorPtr Tensor::max(int dim, bool keepdim) const {
    bool track = requires_grad && GradMode::is_enabled();

    if (dim == -1) {
        auto result = keepdim ? create(std::vector<size_t>(shape.size(), 1), track)
                              : create({1}, track);
        size_t max_idx = 0;
        float max_val = data()[0];
        for (size_t i = 1; i < size(); i++) {
            if (data()[i] > max_val) {
                max_val = data()[i];
                max_idx = i;
            }
        }
        result->data()[0] = max_val;

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            result->parents = {self_ptr};
            result->grad_fn = [self_ptr, result, max_idx]() {
                self_ptr->grad()[max_idx] += result->grad()[0];
            };
        }
        return result;
    }

    assert(shape.size() == 2);
    size_t rows = shape[0], cols = shape[1];

    if (dim == 0) {
        auto result = keepdim ? create({1, cols}, track) : create({cols}, track);
        std::vector<size_t> max_indices(cols);

        for (size_t j = 0; j < cols; j++) {
            size_t max_i = 0;
            float max_val = data()[j];
            for (size_t i = 1; i < rows; i++) {
                if (data()[i * cols + j] > max_val) {
                    max_val = data()[i * cols + j];
                    max_i = i;
                }
            }
            result->data()[j] = max_val;
            max_indices[j] = max_i;
        }

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            result->parents = {self_ptr};
            result->grad_fn = [self_ptr, result, cols, max_indices]() {
                for (size_t j = 0; j < cols; j++) {
                    self_ptr->grad()[max_indices[j] * cols + j] += result->grad()[j];
                }
            };
        }
        return result;
    } else {
        auto result = keepdim ? create({rows, 1}, track) : create({rows}, track);
        std::vector<size_t> max_indices(rows);

        for (size_t i = 0; i < rows; i++) {
            size_t max_j = 0;
            float max_val = data()[i * cols];
            for (size_t j = 1; j < cols; j++) {
                if (data()[i * cols + j] > max_val) {
                    max_val = data()[i * cols + j];
                    max_j = j;
                }
            }
            result->data()[i] = max_val;
            max_indices[i] = max_j;
        }

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            result->parents = {self_ptr};
            result->grad_fn = [self_ptr, result, rows, cols, max_indices]() {
                for (size_t i = 0; i < rows; i++) {
                    self_ptr->grad()[i * cols + max_indices[i]] += result->grad()[i];
                }
            };
        }
        return result;
    }
}

TensorPtr Tensor::min(int dim, bool keepdim) const {
    bool track = requires_grad && GradMode::is_enabled();

    if (dim == -1) {
        auto result = keepdim ? create(std::vector<size_t>(shape.size(), 1), track)
                              : create({1}, track);
        size_t min_idx = 0;
        float min_val = data()[0];
        for (size_t i = 1; i < size(); i++) {
            if (data()[i] < min_val) {
                min_val = data()[i];
                min_idx = i;
            }
        }
        result->data()[0] = min_val;

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            result->parents = {self_ptr};
            result->grad_fn = [self_ptr, result, min_idx]() {
                self_ptr->grad()[min_idx] += result->grad()[0];
            };
        }
        return result;
    }

    assert(shape.size() == 2);
    size_t rows = shape[0], cols = shape[1];

    if (dim == 0) {
        auto result = keepdim ? create({1, cols}, track) : create({cols}, track);
        std::vector<size_t> min_indices(cols);

        for (size_t j = 0; j < cols; j++) {
            size_t min_i = 0;
            float min_val = data()[j];
            for (size_t i = 1; i < rows; i++) {
                if (data()[i * cols + j] < min_val) {
                    min_val = data()[i * cols + j];
                    min_i = i;
                }
            }
            result->data()[j] = min_val;
            min_indices[j] = min_i;
        }

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            result->parents = {self_ptr};
            result->grad_fn = [self_ptr, result, cols, min_indices]() {
                for (size_t j = 0; j < cols; j++) {
                    self_ptr->grad()[min_indices[j] * cols + j] += result->grad()[j];
                }
            };
        }
        return result;
    } else {
        auto result = keepdim ? create({rows, 1}, track) : create({rows}, track);
        std::vector<size_t> min_indices(rows);

        for (size_t i = 0; i < rows; i++) {
            size_t min_j = 0;
            float min_val = data()[i * cols];
            for (size_t j = 1; j < cols; j++) {
                if (data()[i * cols + j] < min_val) {
                    min_val = data()[i * cols + j];
                    min_j = j;
                }
            }
            result->data()[i] = min_val;
            min_indices[i] = min_j;
        }

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            result->parents = {self_ptr};
            result->grad_fn = [self_ptr, result, rows, cols, min_indices]() {
                for (size_t i = 0; i < rows; i++) {
                    self_ptr->grad()[i * cols + min_indices[i]] += result->grad()[i];
                }
            };
        }
        return result;
    }
}

TensorPtr Tensor::max(const TensorPtr& other) const {
    assert(is_broadcastable(shape, other->shape) && "Shapes are not broadcastable");

    bool track = (requires_grad || other->requires_grad) && GradMode::is_enabled();

    if (shape == other->shape) {
        auto result = create(shape, track);
        for (size_t i = 0; i < size(); i++) {
            result->data()[i] = std::max(data()[i], other->data()[i]);
        }

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            auto other_ptr = other;
            result->parents = {self_ptr, other_ptr};
            result->grad_fn = [self_ptr, other_ptr, result]() {
                for (size_t i = 0; i < self_ptr->size(); i++) {
                    if (self_ptr->data()[i] >= other_ptr->data()[i]) {
                        if (self_ptr->requires_grad)
                            self_ptr->grad()[i] += result->grad()[i];
                    } else {
                        if (other_ptr->requires_grad)
                            other_ptr->grad()[i] += result->grad()[i];
                    }
                }
            };
        }
        return result;
    }

    auto out_shape = broadcast_shape(shape, other->shape);
    auto result = create(out_shape, track);

    auto a_strides = compute_strides(shape);
    auto b_strides = compute_strides(other->shape);
    auto out_strides = compute_strides(out_shape);
    size_t ndim = out_shape.size();

    std::vector<size_t> idx(ndim, 0);
    for (size_t i = 0; i < result->size(); i++) {
        size_t a_idx = broadcast_index(idx, shape, a_strides, ndim);
        size_t b_idx = broadcast_index(idx, other->shape, b_strides, ndim);
        result->data()[i] = std::max(data()[a_idx], other->data()[b_idx]);

        for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
            idx[d]++;
            if (idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        auto other_ptr = other;
        auto self_shape = shape;
        auto other_shape = other->shape;

        result->parents = {self_ptr, other_ptr};
        result->grad_fn = [self_ptr, other_ptr, result, self_shape, other_shape,
                          out_shape, a_strides, b_strides, out_strides, ndim]() {
            std::vector<size_t> idx(ndim, 0);
            for (size_t i = 0; i < result->size(); i++) {
                size_t a_idx = broadcast_index(idx, self_shape, a_strides, ndim);
                size_t b_idx = broadcast_index(idx, other_shape, b_strides, ndim);

                if (self_ptr->data()[a_idx] >= other_ptr->data()[b_idx]) {
                    if (self_ptr->requires_grad)
                        self_ptr->grad()[a_idx] += result->grad()[i];
                } else {
                    if (other_ptr->requires_grad)
                        other_ptr->grad()[b_idx] += result->grad()[i];
                }

                for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
                    idx[d]++;
                    if (idx[d] < out_shape[d]) break;
                    idx[d] = 0;
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::min(const TensorPtr& other) const {
    assert(is_broadcastable(shape, other->shape) && "Shapes are not broadcastable");

    bool track = (requires_grad || other->requires_grad) && GradMode::is_enabled();

    if (shape == other->shape) {
        auto result = create(shape, track);
        for (size_t i = 0; i < size(); i++) {
            result->data()[i] = std::min(data()[i], other->data()[i]);
        }

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            auto other_ptr = other;
            result->parents = {self_ptr, other_ptr};
            result->grad_fn = [self_ptr, other_ptr, result]() {
                for (size_t i = 0; i < self_ptr->size(); i++) {
                    if (self_ptr->data()[i] <= other_ptr->data()[i]) {
                        if (self_ptr->requires_grad)
                            self_ptr->grad()[i] += result->grad()[i];
                    } else {
                        if (other_ptr->requires_grad)
                            other_ptr->grad()[i] += result->grad()[i];
                    }
                }
            };
        }
        return result;
    }

    auto out_shape = broadcast_shape(shape, other->shape);
    auto result = create(out_shape, track);

    auto a_strides = compute_strides(shape);
    auto b_strides = compute_strides(other->shape);
    auto out_strides = compute_strides(out_shape);
    size_t ndim = out_shape.size();

    std::vector<size_t> idx(ndim, 0);
    for (size_t i = 0; i < result->size(); i++) {
        size_t a_idx = broadcast_index(idx, shape, a_strides, ndim);
        size_t b_idx = broadcast_index(idx, other->shape, b_strides, ndim);
        result->data()[i] = std::min(data()[a_idx], other->data()[b_idx]);

        for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
            idx[d]++;
            if (idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        auto other_ptr = other;
        auto self_shape = shape;
        auto other_shape = other->shape;

        result->parents = {self_ptr, other_ptr};
        result->grad_fn = [self_ptr, other_ptr, result, self_shape, other_shape,
                          out_shape, a_strides, b_strides, out_strides, ndim]() {
            std::vector<size_t> idx(ndim, 0);
            for (size_t i = 0; i < result->size(); i++) {
                size_t a_idx = broadcast_index(idx, self_shape, a_strides, ndim);
                size_t b_idx = broadcast_index(idx, other_shape, b_strides, ndim);

                if (self_ptr->data()[a_idx] <= other_ptr->data()[b_idx]) {
                    if (self_ptr->requires_grad)
                        self_ptr->grad()[a_idx] += result->grad()[i];
                } else {
                    if (other_ptr->requires_grad)
                        other_ptr->grad()[b_idx] += result->grad()[i];
                }

                for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
                    idx[d]++;
                    if (idx[d] < out_shape[d]) break;
                    idx[d] = 0;
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::argmax(int dim, bool keepdim) const {
    if (dim < 0) dim = static_cast<int>(shape.size()) + dim;
    assert(dim >= 0 && dim < static_cast<int>(shape.size()));

    size_t ndims = shape.size();

    std::vector<size_t> out_shape;
    for (size_t i = 0; i < ndims; i++) {
        if (static_cast<int>(i) == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape[i]);
        }
    }
    if (out_shape.empty()) out_shape.push_back(1);

    auto result = create(out_shape, false);

    std::vector<size_t> strides(ndims);
    strides[ndims - 1] = 1;
    for (int i = static_cast<int>(ndims) - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    size_t dim_size = shape[dim];
    size_t dim_stride = strides[dim];

    size_t num_slices = 1;
    for (size_t i = 0; i < ndims; i++) {
        if (static_cast<int>(i) != dim) num_slices *= shape[i];
    }

    size_t out_idx = 0;
    std::vector<size_t> idx(ndims, 0);

    for (size_t slice = 0; slice < num_slices; slice++) {
        size_t base_idx = 0;
        for (size_t i = 0; i < ndims; i++) {
            if (static_cast<int>(i) != dim) {
                base_idx += idx[i] * strides[i];
            }
        }

        float max_val = data()[base_idx];
        size_t max_idx = 0;
        for (size_t d = 1; d < dim_size; d++) {
            float val = data()[base_idx + d * dim_stride];
            if (val > max_val) {
                max_val = val;
                max_idx = d;
            }
        }
        result->data()[out_idx++] = static_cast<float>(max_idx);

        for (int i = static_cast<int>(ndims) - 1; i >= 0; i--) {
            if (i == dim) continue;
            idx[i]++;
            if (idx[i] < shape[i]) break;
            idx[i] = 0;
        }
    }

    return result;
}

TensorPtr Tensor::argmin(int dim, bool keepdim) const {
    if (dim < 0) dim = static_cast<int>(shape.size()) + dim;
    assert(dim >= 0 && dim < static_cast<int>(shape.size()));

    size_t ndims = shape.size();

    std::vector<size_t> out_shape;
    for (size_t i = 0; i < ndims; i++) {
        if (static_cast<int>(i) == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape[i]);
        }
    }
    if (out_shape.empty()) out_shape.push_back(1);

    auto result = create(out_shape, false);

    std::vector<size_t> strides(ndims);
    strides[ndims - 1] = 1;
    for (int i = static_cast<int>(ndims) - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    size_t dim_size = shape[dim];
    size_t dim_stride = strides[dim];

    size_t num_slices = 1;
    for (size_t i = 0; i < ndims; i++) {
        if (static_cast<int>(i) != dim) num_slices *= shape[i];
    }

    size_t out_idx = 0;
    std::vector<size_t> idx(ndims, 0);

    for (size_t slice = 0; slice < num_slices; slice++) {
        size_t base_idx = 0;
        for (size_t i = 0; i < ndims; i++) {
            if (static_cast<int>(i) != dim) {
                base_idx += idx[i] * strides[i];
            }
        }

        float min_val = data()[base_idx];
        size_t min_idx = 0;
        for (size_t d = 1; d < dim_size; d++) {
            float val = data()[base_idx + d * dim_stride];
            if (val < min_val) {
                min_val = val;
                min_idx = d;
            }
        }
        result->data()[out_idx++] = static_cast<float>(min_idx);

        for (int i = static_cast<int>(ndims) - 1; i >= 0; i--) {
            if (i == dim) continue;
            idx[i]++;
            if (idx[i] < shape[i]) break;
            idx[i] = 0;
        }
    }

    return result;
}

TensorPtr Tensor::transpose() const {
    assert(shape.size() == 2);
    size_t rows = shape[0], cols = shape[1];

    bool track = requires_grad && GradMode::is_enabled();
    auto result = create({cols, rows}, track);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result->data()[j * rows + i] = data()[i * cols + j];
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result, rows, cols]() {
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    self_ptr->grad()[i * cols + j] += result->grad()[j * rows + i];
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::reshape(const std::vector<size_t>& new_shape) const {
    size_t total = 1;
    for (auto s : new_shape) total *= s;
    assert(total == size());

    bool track = requires_grad && GradMode::is_enabled();
    auto result = std::make_shared<Tensor>(data_storage_, data_size_, new_shape, track);
    if (track && !grad_empty()) std::memcpy(result->grad(), grad(), size() * sizeof(float));

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result]() {
            simd_add(self_ptr->grad(), self_ptr->grad(),
                     result->grad(), self_ptr->size());
        };
    }
    return result;
}

TensorPtr Tensor::slice(size_t start, size_t end, int dim) const {
    assert(shape.size() == 2);
    assert(dim == 0);

    size_t cols = shape[1];
    size_t num_rows = end - start;

    bool track = requires_grad && GradMode::is_enabled();
    auto result = create({num_rows, cols}, track);

    std::memcpy(result->data(), &data()[start * cols], num_rows * cols * sizeof(float));

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result, start, cols, num_rows]() {
            for (size_t i = 0; i < num_rows * cols; i++) {
                self_ptr->grad()[start * cols + i] += result->grad()[i];
            }
        };
    }
    return result;
}

void Tensor::print(const char* name) const {
    if (name) printf("%s: ", name);
    printf("Tensor(shape=[");
    for (size_t i = 0; i < shape.size(); i++) {
        printf("%zu%s", shape[i], i < shape.size() - 1 ? ", " : "");
    }
    printf("]");

    if (size() <= 10) {
        printf(", data=[");
        for (size_t i = 0; i < size(); i++) {
            printf("%.4f%s", data()[i], i < size() - 1 ? ", " : "");
        }
        printf("]");
    }
    printf(")\n");
}

TensorPtr Tensor::squeeze(int dim) const {
    std::vector<size_t> new_shape;
    auto self = const_cast<Tensor*>(this)->shared_from_this();

    if (dim == -1) {
        for (size_t i = 0; i < shape.size(); i++) {
            if (shape[i] != 1) {
                new_shape.push_back(shape[i]);
            }
        }
        if (new_shape.empty()) {
            new_shape.push_back(1);
        }
    } else {
        int ndims = static_cast<int>(shape.size());
        if (dim < 0) dim += ndims;
        assert(dim >= 0 && dim < ndims && "Dimension out of range");

        for (int i = 0; i < ndims; i++) {
            if (i == dim) {
                if (shape[i] != 1) {
                    new_shape.push_back(shape[i]);
                }
            } else {
                new_shape.push_back(shape[i]);
            }
        }
    }

    if (new_shape == shape) {
        return self;
    }

    bool track = should_track_grad();
    auto result = std::make_shared<Tensor>(data_storage_, data_size_, new_shape, track);

    if (track) {
        result->parents = {self};
        result->grad_fn = [self, result]() {
            for (size_t i = 0; i < self->size(); i++) {
                self->grad()[i] += result->grad()[i];
            }
        };
    }

    return result;
}

TensorPtr Tensor::unsqueeze(int dim) const {
    auto self = const_cast<Tensor*>(this)->shared_from_this();
    int ndims = static_cast<int>(shape.size());

    if (dim < 0) dim += ndims + 1;
    assert(dim >= 0 && dim <= ndims && "Dimension out of range");

    std::vector<size_t> new_shape;
    for (int i = 0; i < ndims + 1; i++) {
        if (i == dim) {
            new_shape.push_back(1);
        }
        if (i < ndims) {
            new_shape.push_back(shape[i]);
        }
    }

    bool track = should_track_grad();
    auto result = std::make_shared<Tensor>(data_storage_, data_size_, new_shape, track);

    if (track) {
        result->parents = {self};
        result->grad_fn = [self, result]() {
            for (size_t i = 0; i < self->size(); i++) {
                self->grad()[i] += result->grad()[i];
            }
        };
    }

    return result;
}

TensorPtr Tensor::permute(const std::vector<int>& dims) const {
    auto self = const_cast<Tensor*>(this)->shared_from_this();
    int ndims = static_cast<int>(shape.size());

    assert(static_cast<int>(dims.size()) == ndims && "Permute dims must match tensor ndim");

    std::vector<int> perm(ndims);
    std::vector<bool> seen(ndims, false);
    for (int i = 0; i < ndims; i++) {
        int d = dims[i];
        if (d < 0) d += ndims;
        assert(d >= 0 && d < ndims && "Permute dimension out of range");
        assert(!seen[d] && "Duplicate dimension in permute");
        seen[d] = true;
        perm[i] = d;
    }

    std::vector<size_t> new_shape(ndims);
    for (int i = 0; i < ndims; i++) {
        new_shape[i] = shape[perm[i]];
    }

    std::vector<size_t> old_strides(ndims);
    old_strides[ndims - 1] = 1;
    for (int i = ndims - 2; i >= 0; i--) {
        old_strides[i] = old_strides[i + 1] * shape[i + 1];
    }

    std::vector<size_t> new_strides(ndims);
    new_strides[ndims - 1] = 1;
    for (int i = ndims - 2; i >= 0; i--) {
        new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
    }

    auto result = Tensor::create(new_shape, requires_grad);

    for (size_t i = 0; i < size(); i++) {
        std::vector<size_t> new_idx(ndims);
        size_t tmp = i;
        for (int d = 0; d < ndims; d++) {
            new_idx[d] = tmp / new_strides[d];
            tmp %= new_strides[d];
        }

        size_t old_linear = 0;
        for (int d = 0; d < ndims; d++) {
            old_linear += new_idx[d] * old_strides[perm[d]];
        }

        result->data()[i] = data()[old_linear];
    }

    if (should_track_grad()) {
        result->parents = {self};

        std::vector<int> inv_perm(ndims);
        for (int i = 0; i < ndims; i++) {
            inv_perm[perm[i]] = i;
        }

        result->grad_fn = [self, result, perm, inv_perm, old_strides, new_strides, ndims]() {
            for (size_t i = 0; i < result->size(); i++) {
                std::vector<size_t> new_idx(ndims);
                size_t tmp = i;
                for (int d = 0; d < ndims; d++) {
                    new_idx[d] = tmp / new_strides[d];
                    tmp %= new_strides[d];
                }

                size_t old_linear = 0;
                for (int d = 0; d < ndims; d++) {
                    old_linear += new_idx[d] * old_strides[perm[d]];
                }

                self->grad()[old_linear] += result->grad()[i];
            }
        };
    }

    return result;
}

TensorPtr operator*(float scalar, const TensorPtr& t) {
    return t->mul(scalar);
}

// ---------------------------------------------------------------------------
// fp16 conversion
// ---------------------------------------------------------------------------

TensorPtr Tensor::half() const {
    if (dtype == DType::Float16) return const_cast<Tensor*>(this)->shared_from_this();

    auto result = std::make_shared<Tensor>();
    result->shape = shape;
    result->dtype = DType::Float16;
    result->requires_grad = false;  // fp16 tensors don't track gradients
    result->data_size_ = data_size_;

    // Allocate fp16 storage
    result->half_storage_ = std::shared_ptr<uint16_t>(
        new uint16_t[data_size_], [](uint16_t* p) { delete[] p; });

    float_to_half(data(), result->half_storage_.get(), data_size_);
    return result;
}

TensorPtr Tensor::to_float() const {
    if (dtype == DType::Float32) return const_cast<Tensor*>(this)->shared_from_this();

    auto result = create(shape, false);
    half_to_float(half_storage_.get(), result->data(), data_size_);
    return result;
}

TensorPtr Tensor::to(DType target_dtype) const {
    if (target_dtype == DType::Float16) return half();
    return to_float();
}
