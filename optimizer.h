#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"
#include <vector>

class Optimizer {
public:
    std::vector<TensorPtr> params;
    float lr;

    Optimizer(const std::vector<TensorPtr>& params, float lr);
    virtual ~Optimizer() = default;

    virtual void step() = 0;
    void zero_grad();
};

class SGD : public Optimizer {
public:
    float momentum;
    std::vector<std::vector<float>> velocity;

    SGD(const std::vector<TensorPtr>& params, float lr, float momentum = 0.0f);
    void step() override;
};

class Adam : public Optimizer {
public:
    float beta1, beta2, eps;
    int t;
    std::vector<std::vector<float>> m;
    std::vector<std::vector<float>> v;

    Adam(const std::vector<TensorPtr>& params, float lr = 0.001f,
         float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);
    void step() override;
};

#endif
