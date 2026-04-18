"""Loss functions."""

from __future__ import annotations

import numpy as np
from ..tensor import Tensor, _acc
from .module import Module


class MSELoss(Module):
    """Mean Squared Error loss."""

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred.sub(target)
        return diff.mul(diff).mean()


class L1Loss(Module):
    """Mean Absolute Error loss."""

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return pred.sub(target).abs().mean()


class SmoothL1Loss(Module):
    """Smooth L1 (Huber) loss."""

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred.sub(target)
        abs_diff = diff.abs()
        beta = self.beta
        # Piecewise: 0.5*x^2/beta if |x|<beta, else |x|-0.5*beta
        d = diff.data
        ad = np.abs(d)
        loss_data = np.where(ad < beta, 0.5 * d ** 2 / beta, ad - 0.5 * beta)
        result = pred._make_result(np.atleast_1d(loss_data.mean()), [pred, target])
        if result.requires_grad:
            n = float(pred.data.size)
            def _backward(grad):
                scale = grad.flat[0] / n
                g = np.where(ad < beta, d / beta, np.sign(d)) * scale
                if pred.requires_grad:
                    pred.grad = _acc(pred.grad, g.astype(np.float32))
                if target.requires_grad:
                    target.grad = _acc(target.grad, (-g).astype(np.float32))
            result._grad_fn = _backward
        return result


class CrossEntropyLoss(Module):
    """Cross-entropy loss (expects raw logits and integer labels)."""

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        # logits: (N, C), targets: (N,) integer labels
        x = logits.data
        t = targets.data.astype(np.int64)
        N = x.shape[0]
        # Numerically stable log-softmax
        x_max = x.max(axis=1, keepdims=True)
        log_sum_exp = x_max + np.log(np.exp(x - x_max).sum(axis=1, keepdims=True))
        log_probs = x - log_sum_exp
        loss_val = -log_probs[np.arange(N), t].mean()

        result = logits._make_result(np.atleast_1d(loss_val), [logits])
        if result.requires_grad:
            probs = np.exp(log_probs)
            def _backward(grad):
                if logits.requires_grad:
                    g = probs.copy()
                    g[np.arange(N), t] -= 1.0
                    g *= grad.flat[0] / N
                    logits.grad = _acc(logits.grad, g.astype(np.float32))
            result._grad_fn = _backward
        return result


class NLLLoss(Module):
    """Negative log-likelihood loss (expects log-probabilities)."""

    def forward(self, log_probs: Tensor, targets: Tensor) -> Tensor:
        lp = log_probs.data
        t = targets.data.astype(np.int64)
        N = lp.shape[0]
        loss_val = -lp[np.arange(N), t].mean()

        result = log_probs._make_result(np.atleast_1d(loss_val), [log_probs])
        if result.requires_grad:
            def _backward(grad):
                if log_probs.requires_grad:
                    g = np.zeros_like(lp)
                    g[np.arange(N), t] = -1.0 / N
                    log_probs.grad = _acc(log_probs.grad, (g * grad.flat[0]).astype(np.float32))
            result._grad_fn = _backward
        return result


class BCELoss(Module):
    """Binary cross-entropy loss (expects probabilities)."""

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        p = np.clip(pred.data, 1e-7, 1 - 1e-7)
        y = target.data
        loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        result = pred._make_result(np.atleast_1d(loss.mean()), [pred, target])
        if result.requires_grad:
            n = float(pred.data.size)
            def _backward(grad):
                scale = grad.flat[0] / n
                g = (-y / p + (1 - y) / (1 - p)) * scale
                if pred.requires_grad:
                    pred.grad = _acc(pred.grad, g.astype(np.float32))
            result._grad_fn = _backward
        return result


class BCEWithLogitsLoss(Module):
    """Binary cross-entropy with logits (numerically stable)."""

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        x = logits.data
        y = target.data
        # max(x,0) - x*y + log(1+exp(-|x|))
        loss = np.maximum(x, 0) - x * y + np.log1p(np.exp(-np.abs(x)))
        result = logits._make_result(np.atleast_1d(loss.mean()), [logits, target])
        if result.requires_grad:
            sig = 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))
            n = float(logits.data.size)
            def _backward(grad):
                scale = grad.flat[0] / n
                g = (sig - y) * scale
                if logits.requires_grad:
                    logits.grad = _acc(logits.grad, g.astype(np.float32))
            result._grad_fn = _backward
        return result


class KLDivLoss(Module):
    """KL divergence loss."""

    def __init__(self, log_target: bool = False):
        super().__init__()
        self.log_target = log_target

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # pred = log(Q), target = P (or log(P))
        p = pred.data
        if self.log_target:
            t_exp = np.exp(target.data)
            loss = (t_exp * (target.data - p)).sum() / p.shape[0]
        else:
            t = target.data
            loss = (t * (np.log(t + 1e-8) - p)).sum() / p.shape[0]

        result = pred._make_result(np.atleast_1d(loss), [pred])
        if result.requires_grad:
            N = p.shape[0]
            def _backward(grad):
                if pred.requires_grad:
                    if self.log_target:
                        g = -np.exp(target.data) / N
                    else:
                        g = -target.data / N
                    pred.grad = _acc(pred.grad, (g * grad.flat[0]).astype(np.float32))
            result._grad_fn = _backward
        return result


class FocalLoss(Module):
    """Focal loss for class-imbalanced classification."""

    def __init__(self, gamma: float = 2.0, alpha: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        x = logits.data
        t = targets.data.astype(np.int64)
        N = x.shape[0]
        gamma, alpha = self.gamma, self.alpha

        x_max = x.max(axis=1, keepdims=True)
        e = np.exp(x - x_max)
        probs = e / e.sum(axis=1, keepdims=True)

        p_t = probs[np.arange(N), t]
        focal_weight = (1 - p_t) ** gamma
        loss = -alpha * focal_weight * np.log(p_t + 1e-8)
        result = logits._make_result(np.atleast_1d(loss.mean()), [logits])
        if result.requires_grad:
            def _backward(grad):
                if logits.requires_grad:
                    g2 = probs.copy()
                    for i in range(N):
                        pt_i = p_t[i]
                        log_pt_i = np.log(pt_i + 1e-8)
                        fw = (1 - pt_i) ** gamma
                        dfw = -gamma * (1 - pt_i) ** (gamma - 1)
                        for c in range(x.shape[1]):
                            # dsoftmax/dlogit: p_c * (delta_ct - p_t)
                            ds = probs[i, c] * ((1.0 if c == t[i] else 0.0) - probs[i, t[i]])
                            # d(loss_i)/dlogit_c = -alpha * (dfw * ds * log_pt + fw * ds / pt)
                            g2[i, c] = -alpha * (dfw * ds * log_pt_i + fw * ds / (pt_i + 1e-8))
                    logits.grad = _acc(logits.grad, (g2 * grad.flat[0] / N).astype(np.float32))
            result._grad_fn = _backward
        return result


class BinaryFocalLoss(Module):
    """Binary focal loss with sigmoid."""

    def __init__(self, gamma: float = 2.0, alpha: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        x = logits.data
        y = target.data
        gamma, alpha = self.gamma, self.alpha

        sig = 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))
        p_t = y * sig + (1 - y) * (1 - sig)
        focal_weight = (1 - p_t) ** gamma
        bce = np.maximum(x, 0) - x * y + np.log1p(np.exp(-np.abs(x)))
        loss = alpha * focal_weight * bce

        result = logits._make_result(np.atleast_1d(loss.mean()), [logits])
        if result.requires_grad:
            n = float(logits.data.size)
            def _backward(grad):
                if logits.requires_grad:
                    # d/dx = alpha * [dfw * bce + fw * dbce]
                    dp_t = y * sig * (1 - sig) + (1 - y) * (-sig * (1 - sig))
                    dfw = -gamma * (1 - p_t) ** (gamma - 1) * dp_t
                    dbce = sig - y
                    g = alpha * (dfw * bce + focal_weight * dbce)
                    logits.grad = _acc(logits.grad, (g * grad.flat[0] / n).astype(np.float32))
            result._grad_fn = _backward
        return result
