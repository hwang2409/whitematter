#!/usr/bin/env python3
"""
ResNet-18 ONNX Converter

Reads the binary weights file produced by resnet18_export and builds a
complete ONNX model with the full ResNet-18 graph, including skip connections.

Usage:
    python resnet18_to_onnx.py <weights.bin> [output.onnx]

Requirements:
    pip install numpy onnx
"""

import struct
import sys
import os
import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError:
    print("Error: 'onnx' package is required. Install with: pip install onnx")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Binary file reader
# ---------------------------------------------------------------------------
MAGIC = 0x524E4554  # "RNET"


def read_weights(path):
    """Read the binary weights file and return a dict of name -> numpy array."""
    params = {}
    with open(path, "rb") as f:
        magic, = struct.unpack("<I", f.read(4))
        if magic != MAGIC:
            raise ValueError(f"Bad magic: 0x{magic:08X} (expected 0x{MAGIC:08X})")

        num_params, = struct.unpack("<I", f.read(4))

        for _ in range(num_params):
            name_len, = struct.unpack("<I", f.read(4))
            name = f.read(name_len).decode("ascii")

            ndim, = struct.unpack("<I", f.read(4))
            shape = struct.unpack(f"<{ndim}I", f.read(4 * ndim))

            numel = 1
            for d in shape:
                numel *= d
            data = np.frombuffer(f.read(numel * 4), dtype=np.float32).copy()
            data = data.reshape(shape)
            params[name] = data

    print(f"Loaded {len(params)} tensors from {path}")
    return params


# ---------------------------------------------------------------------------
# ONNX graph builder helpers
# ---------------------------------------------------------------------------
class GraphBuilder:
    """Incrementally builds an ONNX graph with named intermediates."""

    def __init__(self):
        self.nodes = []
        self.initializers = []
        self.counter = 0

    def _name(self, prefix="t"):
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def add_initializer(self, name, data):
        self.initializers.append(numpy_helper.from_array(data, name=name))

    def conv(self, x, weight_name, bias_name, kernel_shape, strides, pads,
             name_prefix="conv"):
        y = self._name(name_prefix)
        self.nodes.append(helper.make_node(
            "Conv", [x, weight_name, bias_name], [y],
            kernel_shape=kernel_shape, strides=strides, pads=pads,
        ))
        return y

    def batchnorm(self, x, scale_name, bias_name, mean_name, var_name,
                  epsilon=1e-5, name_prefix="bn"):
        y = self._name(name_prefix)
        self.nodes.append(helper.make_node(
            "BatchNormalization",
            [x, scale_name, bias_name, mean_name, var_name], [y],
            epsilon=epsilon,
        ))
        return y

    def relu(self, x):
        y = self._name("relu")
        self.nodes.append(helper.make_node("Relu", [x], [y]))
        return y

    def add(self, a, b):
        y = self._name("add")
        self.nodes.append(helper.make_node("Add", [a, b], [y]))
        return y

    def global_avg_pool(self, x):
        y = self._name("gap")
        self.nodes.append(helper.make_node("GlobalAveragePool", [x], [y]))
        return y

    def flatten(self, x, axis=1):
        y = self._name("flat")
        self.nodes.append(helper.make_node("Flatten", [x], [y], axis=axis))
        return y

    def gemm(self, x, weight_name, bias_name):
        y = self._name("fc")
        self.nodes.append(helper.make_node(
            "Gemm", [x, weight_name, bias_name], [y],
            alpha=1.0, beta=1.0, transB=1,
        ))
        return y


# ---------------------------------------------------------------------------
# ResNet-18 graph construction
# ---------------------------------------------------------------------------
def add_conv_bn_relu(gb, params, x, conv_prefix, bn_prefix,
                     kernel_shape, strides, pads):
    """Conv -> BatchNorm -> ReLU.  Returns output name."""
    cw = conv_prefix + ".weight"
    cb = conv_prefix + ".bias"
    bw = bn_prefix + ".weight"
    bb = bn_prefix + ".bias"
    bm = bn_prefix + ".running_mean"
    bv = bn_prefix + ".running_var"

    gb.add_initializer(cw, params[cw])
    gb.add_initializer(cb, params[cb])
    gb.add_initializer(bw, params[bw])
    gb.add_initializer(bb, params[bb])
    gb.add_initializer(bm, params[bm])
    gb.add_initializer(bv, params[bv])

    y = gb.conv(x, cw, cb, kernel_shape, strides, pads, conv_prefix)
    y = gb.batchnorm(y, bw, bb, bm, bv, name_prefix=bn_prefix)
    y = gb.relu(y)
    return y


def add_conv_bn_only(gb, params, x, conv_prefix, bn_prefix,
                     kernel_shape, strides, pads):
    """Conv -> BatchNorm (no ReLU).  Returns output name."""
    cw = conv_prefix + ".weight"
    cb = conv_prefix + ".bias"
    bw = bn_prefix + ".weight"
    bb = bn_prefix + ".bias"
    bm = bn_prefix + ".running_mean"
    bv = bn_prefix + ".running_var"

    gb.add_initializer(cw, params[cw])
    gb.add_initializer(cb, params[cb])
    gb.add_initializer(bw, params[bw])
    gb.add_initializer(bb, params[bb])
    gb.add_initializer(bm, params[bm])
    gb.add_initializer(bv, params[bv])

    y = gb.conv(x, cw, cb, kernel_shape, strides, pads, conv_prefix)
    y = gb.batchnorm(y, bw, bb, bm, bv, name_prefix=bn_prefix)
    return y


def add_basic_block(gb, params, x, prefix, stride, in_ch, out_ch):
    """
    BasicBlock with skip connection.
      Main path:  conv3x3(stride) -> BN -> ReLU -> conv3x3 -> BN
      Skip path:  identity or conv1x1(stride) -> BN
      Output:     ReLU(main + skip)
    """
    has_downsample = (stride != 1 or in_ch != out_ch)

    # Main path
    main = add_conv_bn_relu(gb, params, x,
                            f"{prefix}.conv1", f"{prefix}.bn1",
                            kernel_shape=[3, 3],
                            strides=[stride, stride],
                            pads=[1, 1, 1, 1])
    main = add_conv_bn_only(gb, params, main,
                            f"{prefix}.conv2", f"{prefix}.bn2",
                            kernel_shape=[3, 3],
                            strides=[1, 1],
                            pads=[1, 1, 1, 1])

    # Skip path
    if has_downsample:
        skip = add_conv_bn_only(gb, params, x,
                                f"{prefix}.downsample.conv",
                                f"{prefix}.downsample.bn",
                                kernel_shape=[1, 1],
                                strides=[stride, stride],
                                pads=[0, 0, 0, 0])
    else:
        skip = x

    # Add + ReLU
    y = gb.add(main, skip)
    y = gb.relu(y)
    return y


def build_resnet18_onnx(params, output_path, num_classes=10):
    """Build the complete ResNet-18 ONNX graph."""
    gb = GraphBuilder()

    # Input: [batch, 3, 32, 32] for CIFAR-10 variant
    x = "input"

    # Stem: conv1(3,64,3,s1,p1) -> bn1 -> relu
    x = add_conv_bn_relu(gb, params, x, "conv1", "bn1",
                         kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])

    # Layer 1: two blocks, 64 channels, no downsampling
    x = add_basic_block(gb, params, x, "layer1.0", stride=1, in_ch=64, out_ch=64)
    x = add_basic_block(gb, params, x, "layer1.1", stride=1, in_ch=64, out_ch=64)

    # Layer 2: first block downsamples (stride=2), 64->128
    x = add_basic_block(gb, params, x, "layer2.0", stride=2, in_ch=64, out_ch=128)
    x = add_basic_block(gb, params, x, "layer2.1", stride=1, in_ch=128, out_ch=128)

    # Layer 3: first block downsamples, 128->256
    x = add_basic_block(gb, params, x, "layer3.0", stride=2, in_ch=128, out_ch=256)
    x = add_basic_block(gb, params, x, "layer3.1", stride=1, in_ch=256, out_ch=256)

    # Layer 4: first block downsamples, 256->512
    x = add_basic_block(gb, params, x, "layer4.0", stride=2, in_ch=256, out_ch=512)
    x = add_basic_block(gb, params, x, "layer4.1", stride=1, in_ch=512, out_ch=512)

    # Global average pool -> flatten -> FC
    x = gb.global_avg_pool(x)
    x = gb.flatten(x)

    # FC layer: Linear(512, num_classes)
    # The library stores weight as [in, out], ONNX Gemm with transB=1 expects [out, in]
    fc_w = params["fc.weight"]  # shape: [512, 10]
    fc_b = params["fc.bias"]    # shape: [10]
    # Transpose for Gemm transB=1: Gemm computes Y = alpha * A * B^T + beta * C
    # A is [batch, 512], B is [10, 512] (transposed), so we need [out, in]
    fc_w_t = fc_w.T.copy()
    gb.add_initializer("fc.weight", fc_w_t)
    gb.add_initializer("fc.bias", fc_b)
    output = gb.gemm(x, "fc.weight", "fc.bias")

    # Build the ONNX graph
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])
    output_info = helper.make_tensor_value_info(output, TensorProto.FLOAT, [1, num_classes])

    graph = helper.make_graph(
        gb.nodes,
        "resnet18_cifar10",
        [input_info],
        [output_info],
        initializer=gb.initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.producer_name = "whitematter"
    model.producer_version = "1.0"
    model.ir_version = 8

    # Validate
    onnx.checker.check_model(model)

    # Save
    onnx.save(model, output_path)

    # Stats
    file_size = os.path.getsize(output_path)
    print(f"\nONNX model saved to: {output_path}")
    print(f"  File size: {file_size / (1024 * 1024):.2f} MB")
    print(f"  Nodes: {len(gb.nodes)}")
    print(f"  Initializers: {len(gb.initializers)}")
    print(f"  Input:  input [1, 3, 32, 32]")
    print(f"  Output: {output} [1, {num_classes}]")
    print(f"\nValidation: PASSED")

    return model


# ---------------------------------------------------------------------------
# Optional: verify with ONNX Runtime
# ---------------------------------------------------------------------------
def verify_model(onnx_path):
    """Run a dummy input through the model to check it executes."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("\nSkipping runtime verification (install onnxruntime for this)")
        return

    sess = ort.InferenceSession(onnx_path)
    dummy = np.random.randn(1, 3, 32, 32).astype(np.float32)
    outputs = sess.run(None, {"input": dummy})
    print(f"\nRuntime verification:")
    print(f"  Input shape:  {dummy.shape}")
    print(f"  Output shape: {outputs[0].shape}")
    print(f"  Output range: [{outputs[0].min():.4f}, {outputs[0].max():.4f}]")
    print(f"  Runtime verification: PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h"):
        print("Usage: python resnet18_to_onnx.py <weights.bin> [output.onnx]")
        print()
        print("Converts ResNet-18 weights (from resnet18_export) to ONNX format.")
        print()
        print("Requirements: pip install numpy onnx")
        print("Optional:     pip install onnxruntime  (for verification)")
        sys.exit(0 if sys.argv[1:] else 1)

    weights_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "resnet18.onnx"

    print(f"ResNet-18 ONNX Converter")
    print(f"========================\n")

    params = read_weights(weights_path)
    model = build_resnet18_onnx(params, output_path)
    verify_model(output_path)


if __name__ == "__main__":
    main()
