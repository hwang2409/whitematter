#!/usr/bin/env python3
"""Minimal training loop using whitematter Python bindings.

Run from repo root:
  pip install -e .   # or: PYTHONPATH=. python examples/train_python.py
  python examples/train_python.py

Requires: numpy, whitematter (build with: python setup.py build_ext --inplace).
"""
import os
import sys

# Prefer repo root so whitematter.cpython-*.so is found when run without installing
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np

try:
    import whitematter as wm
except ImportError as e:
    raise SystemExit(f"Import whitematter failed: {e}. From repo root run: pip install -e .") from e

def main():
    # Model: 2 -> 4 -> 2 (binary-ish)
    model = wm.Sequential()
    model.add(wm.Linear(2, 4))
    model.add(wm.ReLU())
    model.add(wm.Linear(4, 2))

    opt = wm.Adam(model.parameters(), lr=0.1)
    criterion = wm.CrossEntropyLoss()

    # Dummy data: 32 samples, 2 features; label = (x0 + x1 > 0)
    np.random.seed(42)
    x = np.random.randn(32, 2).astype(np.float32) * 0.5
    y = (x[:, 0] + x[:, 1] > 0).astype(np.int64)

    model.train()
    for step in range(50):
        t_x = wm.Tensor.from_numpy(x, requires_grad=True)
        logits = model.forward(t_x)
        loss = wm.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            print(f"step {step} loss {loss.item():.4f}")

    print("Training loop OK. Save/load:")
    wm.save_model(model, "/tmp/wm_train_python.bin")
    model2 = wm.Sequential()
    model2.add(wm.Linear(2, 4))
    model2.add(wm.ReLU())
    model2.add(wm.Linear(4, 2))
    wm.load_model(model2, "/tmp/wm_train_python.bin")
    print("Save/load OK.")

if __name__ == "__main__":
    main()
