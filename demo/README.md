# whitematter CIFAR-10 Demo

Browser-based image classifier using a ResNet-18 trained from scratch.

## Setup

1. Train the model and export to ONNX:
   ```bash
   # Train (or use existing checkpoint)
   ./build/resnet18_cifar10_cuda data 64

   # Export weights
   ./build/resnet18_export checkpoints/resnet18_best.ckpt resnet18_weights.bin

   # Convert to ONNX
   pip install onnx numpy
   python examples/resnet18_to_onnx.py resnet18_weights.bin demo/resnet18.onnx
   ```

2. Serve locally:
   ```bash
   cd demo
   python -m http.server 8080
   ```

3. Open http://localhost:8080

## How it works

- Model runs entirely in the browser via ONNX Runtime Web (WASM)
- Zero server cost -- no GPU/API needed for inference
- Image is center-cropped to square, resized to 32x32, and normalized with CIFAR-10 mean/std
- Top-5 predictions shown with confidence bars

## Normalization

The demo applies the same normalization as the C++ training data loader:

```
normalized = (pixel / 255 - mean) / std
```

Per-channel values:
- Mean: [0.4914, 0.4822, 0.4465]
- Std:  [0.2470, 0.2435, 0.2616]
