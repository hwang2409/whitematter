#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

class LossFunction {
public:
    virtual ~LossFunction() = default;
    virtual TensorPtr forward(const TensorPtr& prediction, const TensorPtr& target) = 0;

    TensorPtr operator()(const TensorPtr& prediction, const TensorPtr& target) {
        return forward(prediction, target);
    }
};

class MSELoss : public LossFunction {
public:
    TensorPtr forward(const TensorPtr& prediction, const TensorPtr& target) override;
};

class CrossEntropyLoss : public LossFunction {
public:
    TensorPtr forward(const TensorPtr& prediction, const TensorPtr& target) override;
};

class NLLLoss : public LossFunction {
public:
    TensorPtr forward(const TensorPtr& prediction, const TensorPtr& target) override;
};

#endif
