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

class L1Loss : public LossFunction {
public:
    TensorPtr forward(const TensorPtr& prediction, const TensorPtr& target) override;
};

class SmoothL1Loss : public LossFunction {
public:
    float beta;  // Threshold for switching between L1 and L2
    SmoothL1Loss(float beta = 1.0f) : beta(beta) {}
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

class BCELoss : public LossFunction {
public:
    TensorPtr forward(const TensorPtr& prediction, const TensorPtr& target) override;
};

class BCEWithLogitsLoss : public LossFunction {
public:
    TensorPtr forward(const TensorPtr& prediction, const TensorPtr& target) override;
};

class KLDivLoss : public LossFunction {
public:
    bool log_target;  // If true, target is already in log space
    KLDivLoss(bool log_target = false) : log_target(log_target) {}
    TensorPtr forward(const TensorPtr& prediction, const TensorPtr& target) override;
};

class FocalLoss : public LossFunction {
public:
    float alpha;  // Class weight (use -1 for no weighting)
    float gamma;  // Focusing parameter (higher = more focus on hard examples)
    FocalLoss(float gamma = 2.0f, float alpha = -1.0f) : alpha(alpha), gamma(gamma) {}
    TensorPtr forward(const TensorPtr& prediction, const TensorPtr& target) override;
};

class BinaryFocalLoss : public LossFunction {
public:
    float alpha;  // Weight for positive class (use -1 for no weighting)
    float gamma;  // Focusing parameter
    BinaryFocalLoss(float gamma = 2.0f, float alpha = 0.25f) : alpha(alpha), gamma(gamma) {}
    TensorPtr forward(const TensorPtr& prediction, const TensorPtr& target) override;
};

#endif
