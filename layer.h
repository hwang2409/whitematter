#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"
#include <vector>
#include <initializer_list>

class Module {
public:
    virtual ~Module() = default;
    virtual TensorPtr forward(const TensorPtr& input) = 0;
    virtual std::vector<TensorPtr> parameters() { return {}; }

    TensorPtr operator()(const TensorPtr& input) {
        return forward(input);
    }
};

class Linear : public Module {
public:
    TensorPtr weight;
    TensorPtr bias;
    size_t in_features, out_features;

    Linear(size_t in_features, size_t out_features);

    TensorPtr forward(const TensorPtr& input) override;
    std::vector<TensorPtr> parameters() override;
};

class ReLU : public Module {
public:
    TensorPtr forward(const TensorPtr& input) override;
};

class Sigmoid : public Module {
public:
    TensorPtr forward(const TensorPtr& input) override;
};

class Tanh : public Module {
public:
    TensorPtr forward(const TensorPtr& input) override;
};

class Softmax : public Module {
public:
    int dim;
    Softmax(int dim = -1);
    TensorPtr forward(const TensorPtr& input) override;
};

class LogSoftmax : public Module {
public:
    int dim;
    LogSoftmax(int dim = -1);
    TensorPtr forward(const TensorPtr& input) override;
};

class Dropout : public Module {
public:
    float p;
    bool training;

    Dropout(float p = 0.5f);
    TensorPtr forward(const TensorPtr& input) override;
    void train() { training = true; }
    void eval() { training = false; }
};

class Sequential : public Module {
public:
    std::vector<Module*> layers;

    Sequential() = default;
    Sequential(std::initializer_list<Module*> modules);
    ~Sequential();

    void add(Module* module);
    TensorPtr forward(const TensorPtr& input) override;
    std::vector<TensorPtr> parameters() override;

    void train();
    void eval();
};

#endif
