#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <memory>
#include <functional>
#include <cmath>
#include <cassert>
#include <cstdio>
#include <random>

// Global gradient computation flag
class GradMode {
public:
    static bool is_enabled();
    static void set_enabled(bool enabled);
private:
    static bool enabled_;
};

// RAII guard for disabling gradients (like PyTorch's torch.no_grad())
class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();
private:
    bool prev_mode_;
};

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<size_t> shape;
    bool requires_grad;

    std::function<void()> grad_fn;
    std::vector<TensorPtr> parents;

    Tensor();
    Tensor(const std::vector<size_t>& shape, bool requires_grad = false);
    Tensor(const std::vector<float>& data, const std::vector<size_t>& shape, bool requires_grad = false);

    static TensorPtr create(const std::vector<size_t>& shape, bool requires_grad = false);
    static TensorPtr create(const std::vector<float>& data, const std::vector<size_t>& shape, bool requires_grad = false);
    static TensorPtr zeros(const std::vector<size_t>& shape, bool requires_grad = false);
    static TensorPtr ones(const std::vector<size_t>& shape, bool requires_grad = false);
    static TensorPtr randn(const std::vector<size_t>& shape, bool requires_grad = false);
    static TensorPtr xavier(size_t fan_in, size_t fan_out, bool requires_grad = true);

    size_t size() const;
    size_t ndim() const;
    float item() const;
    void zero_grad();
    void backward();

    float& operator[](size_t idx);
    float operator[](size_t idx) const;
    float& at(const std::vector<size_t>& indices);
    float at(const std::vector<size_t>& indices) const;

    TensorPtr matmul(const TensorPtr& other) const;
    TensorPtr add(const TensorPtr& other) const;
    TensorPtr sub(const TensorPtr& other) const;
    TensorPtr mul(const TensorPtr& other) const;
    TensorPtr mul(float scalar) const;
    TensorPtr div(float scalar) const;
    TensorPtr neg() const;
    TensorPtr relu() const;
    TensorPtr sigmoid() const;
    TensorPtr tanh_() const;
    TensorPtr log_() const;
    TensorPtr exp_() const;
    TensorPtr softmax(int dim = -1) const;
    TensorPtr log_softmax(int dim = -1) const;
    TensorPtr sum(int dim = -1, bool keepdim = false) const;
    TensorPtr mean(int dim = -1, bool keepdim = false) const;
    TensorPtr transpose() const;
    TensorPtr reshape(const std::vector<size_t>& new_shape) const;
    TensorPtr slice(size_t start, size_t end, int dim = 0) const;

    TensorPtr operator+(const TensorPtr& other) const { return add(other); }
    TensorPtr operator-(const TensorPtr& other) const { return sub(other); }
    TensorPtr operator*(const TensorPtr& other) const { return mul(other); }
    TensorPtr operator*(float scalar) const { return mul(scalar); }
    TensorPtr operator/(float scalar) const { return div(scalar); }
    TensorPtr operator-() const { return neg(); }

    void print(const char* name = nullptr) const;

private:
    void build_topo(std::vector<Tensor*>& topo, std::vector<Tensor*>& visited);
    bool should_track_grad() const;
};

TensorPtr operator*(float scalar, const TensorPtr& t);

#endif
