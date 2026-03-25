#ifndef TENSOR_H
#define TENSOR_H

#include "device.h"
#include "autograd.h"
#include <vector>
#include <memory>
#include <functional>
#include <cmath>
#include <cassert>
#include <cstdio>
#include <random>

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    std::vector<size_t> shape;
    bool requires_grad;
    whitematter::DeviceType device{whitematter::DeviceType::CPU};

    std::function<void()> grad_fn;
    std::vector<TensorPtr> parents;

    // Pool-backed storage accessors (shared_ptr; views share data_storage_)
    float* data() { return data_storage_ ? data_storage_.get() : nullptr; }
    const float* data() const { return data_storage_ ? data_storage_.get() : nullptr; }
    size_t size() const { return data_size_; }
    float* grad() { return grad_storage_ ? grad_storage_.get() : nullptr; }
    const float* grad() const { return grad_storage_ ? grad_storage_.get() : nullptr; }
    size_t grad_size() const { return grad_size_; }
    bool grad_empty() const { return !grad_storage_; }

    Tensor();
    ~Tensor();
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(const std::vector<size_t>& shape, bool requires_grad = false);
    Tensor(const std::vector<float>& data, const std::vector<size_t>& shape, bool requires_grad = false);

    // View constructor: shares data_storage_, own grad. Used by reshape/squeeze/unsqueeze.
    Tensor(std::shared_ptr<float> data_storage, size_t data_size,
          const std::vector<size_t>& shape, bool requires_grad);

    static TensorPtr create(const std::vector<size_t>& shape, bool requires_grad = false);
    static TensorPtr create(const std::vector<float>& data, const std::vector<size_t>& shape, bool requires_grad = false);
    static TensorPtr zeros(const std::vector<size_t>& shape, bool requires_grad = false);
    static TensorPtr ones(const std::vector<size_t>& shape, bool requires_grad = false);
    static TensorPtr randn(const std::vector<size_t>& shape, bool requires_grad = false);
    static TensorPtr xavier(size_t fan_in, size_t fan_out, bool requires_grad = true);

    // Concatenate tensors along an existing dimension
    static TensorPtr concat(const std::vector<TensorPtr>& tensors, int dim = 0);

    // Stack tensors along a new dimension
    static TensorPtr stack(const std::vector<TensorPtr>& tensors, int dim = 0);

    size_t ndim() const;
    float item() const;
    void zero_grad();
    void backward();

    float& operator[](size_t idx);
    float operator[](size_t idx) const;
    float& at(const std::vector<size_t>& indices);
    float at(const std::vector<size_t>& indices) const;

    TensorPtr matmul(const TensorPtr& other) const;
    TensorPtr bmm(const TensorPtr& other) const;  // Batch matrix multiplication
    TensorPtr add(const TensorPtr& other) const;
    TensorPtr sub(const TensorPtr& other) const;
    TensorPtr mul(const TensorPtr& other) const;
    TensorPtr div(const TensorPtr& other) const;
    TensorPtr mul(float scalar) const;
    TensorPtr div(float scalar) const;
    TensorPtr neg() const;
    TensorPtr relu() const;
    TensorPtr sigmoid() const;
    TensorPtr tanh_() const;
    TensorPtr log_() const;
    TensorPtr exp_() const;
    TensorPtr pow(float exponent) const;
    TensorPtr pow(const TensorPtr& exponent) const;  // Element-wise power with broadcasting
    TensorPtr sqrt() const;
    TensorPtr abs() const;
    TensorPtr clamp(float min_val, float max_val) const;
    TensorPtr softmax(int dim = -1) const;
    TensorPtr log_softmax(int dim = -1) const;
    TensorPtr sum(int dim = -1, bool keepdim = false) const;
    TensorPtr mean(int dim = -1, bool keepdim = false) const;
    TensorPtr max(int dim, bool keepdim = false) const;   // Reduce along dim
    TensorPtr min(int dim, bool keepdim = false) const;   // Reduce along dim
    TensorPtr max(const TensorPtr& other) const;          // Element-wise max with broadcasting
    TensorPtr min(const TensorPtr& other) const;          // Element-wise min with broadcasting
    TensorPtr argmax(int dim = -1, bool keepdim = false) const;  // Index of max along dim
    TensorPtr argmin(int dim = -1, bool keepdim = false) const;  // Index of min along dim
    TensorPtr transpose() const;
    TensorPtr reshape(const std::vector<size_t>& new_shape) const;
    TensorPtr slice(size_t start, size_t end, int dim = 0) const;
    TensorPtr conv2d(const TensorPtr& weight, const TensorPtr& bias,
                     size_t stride = 1, size_t padding = 0) const;
    TensorPtr conv_transpose2d(const TensorPtr& weight, const TensorPtr& bias,
                               size_t stride = 1, size_t padding = 0,
                               size_t output_padding = 0) const;
    TensorPtr maxpool2d(size_t kernel_size, size_t stride = 0) const;
    TensorPtr avgpool2d(size_t kernel_size, size_t stride = 0) const;
    TensorPtr flatten(size_t start_dim = 1) const;
    TensorPtr squeeze(int dim = -1) const;      // Remove dimension(s) of size 1
    TensorPtr unsqueeze(int dim) const;         // Add dimension of size 1
    TensorPtr permute(const std::vector<int>& dims) const;  // Reorder dimensions

    // Device placement (CPU, METAL, or CUDA). to(METAL)/to(CUDA) use GPU for supported ops when available.
    TensorPtr to(whitematter::DeviceType d) const;

    // Data augmentation (for images: [N,C,H,W] or [C,H,W])
    TensorPtr flip_horizontal() const;                           // Flip left-right
    TensorPtr random_flip_horizontal(float p = 0.5f) const;      // Random flip with probability p
    TensorPtr pad2d(size_t padding) const;                       // Zero-pad height and width
    TensorPtr crop(size_t top, size_t left, size_t height, size_t width) const;  // Crop region
    TensorPtr random_crop(size_t height, size_t width) const;    // Random crop (use after padding)

    TensorPtr operator+(const TensorPtr& other) const { return add(other); }
    TensorPtr operator-(const TensorPtr& other) const { return sub(other); }
    TensorPtr operator*(const TensorPtr& other) const { return mul(other); }
    TensorPtr operator/(const TensorPtr& other) const { return div(other); }
    TensorPtr operator*(float scalar) const { return mul(scalar); }
    TensorPtr operator/(float scalar) const { return div(scalar); }
    TensorPtr operator-() const { return neg(); }

    void print(const char* name = nullptr) const;

private:
    void build_topo(std::vector<Tensor*>& topo, std::vector<Tensor*>& visited);
    bool should_track_grad() const;

    std::shared_ptr<float> data_storage_;
    size_t data_size_;
    std::shared_ptr<float> grad_storage_;
    size_t grad_size_;
};

TensorPtr operator*(float scalar, const TensorPtr& t);

#endif
