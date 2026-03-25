def generate_makefile() -> str:
    return '''# Auto-generated Makefile
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -ffast-math -funroll-loops
LDFLAGS =

# Detect macOS and add OpenMP flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CXXFLAGS += -mcpu=apple-m1 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
    LDFLAGS += -L/opt/homebrew/opt/libomp/lib -lomp
endif

# Path to whitematter source (project root is two levels up from generated/{job_id}/)
PROJECT_ROOT = ../..
CORE_DIR = $(PROJECT_ROOT)/core
BUILD_DIR = $(PROJECT_ROOT)/build
WHITEMATTER_LIB = $(BUILD_DIR)/libwhitematter.a

all: train infer

train: train.cpp $(WHITEMATTER_LIB)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -o $@ $< -L$(BUILD_DIR) -lwhitematter $(LDFLAGS)

infer: infer.cpp $(WHITEMATTER_LIB)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -o $@ $< -L$(BUILD_DIR) -lwhitematter $(LDFLAGS)

clean:
\trm -f train infer

.PHONY: clean
'''


def generate_byoc_makefile() -> str:
    return '''# Auto-generated standalone Makefile for BYOC (Linux x86_64, EC2 Deep Learning AMI)
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -fopenmp
LDFLAGS = -fopenmp

# SIMD: AVX/FMA with fallback if not supported
SIMD_CHECK := $(shell $(CXX) -mavx -mfma -xc++ -E - < /dev/null 2>/dev/null && echo 1 || echo 0)
ifeq ($(SIMD_CHECK),1)
    CXXFLAGS += -mavx -mfma
endif

CORE_DIR = core
LAYERS_DIR = $(CORE_DIR)/layers
SERIAL_DIR = $(CORE_DIR)/serialization
OPS_DIR = $(CORE_DIR)/ops
BUILD_DIR = build

WHITEMATTER_LIB = $(BUILD_DIR)/libwhitematter.a

CORE_SRCS = $(CORE_DIR)/memory_pool.cpp $(CORE_DIR)/autograd.cpp $(CORE_DIR)/broadcast.cpp \\
            $(CORE_DIR)/tensor.cpp $(CORE_DIR)/loss.cpp \\
            $(CORE_DIR)/optimizer.cpp $(SERIAL_DIR)/serialize.cpp $(CORE_DIR)/dataloader.cpp \\
            $(CORE_DIR)/model_zoo.cpp $(SERIAL_DIR)/onnx_export.cpp $(SERIAL_DIR)/onnx_import.cpp $(CORE_DIR)/device.cpp \\
            $(LAYERS_DIR)/linear.cpp $(LAYERS_DIR)/activations.cpp $(LAYERS_DIR)/conv.cpp \\
            $(LAYERS_DIR)/normalization.cpp $(LAYERS_DIR)/embedding.cpp $(LAYERS_DIR)/recurrent.cpp \\
            $(LAYERS_DIR)/attention.cpp $(LAYERS_DIR)/sequential.cpp \\
            $(OPS_DIR)/simd_ops_avx.cpp $(OPS_DIR)/simd_ops_neon.cpp $(OPS_DIR)/simd_ops_fallback.cpp \\
            $(OPS_DIR)/matmul_cpu.cpp $(OPS_DIR)/im2col.cpp \\
            $(OPS_DIR)/conv_ops.cpp $(OPS_DIR)/augmentation.cpp \\
            $(CORE_DIR)/cuda/cuda_stub.cpp $(CORE_DIR)/metal/metal_stub.cpp

CORE_OBJS = $(BUILD_DIR)/memory_pool.o $(BUILD_DIR)/autograd.o $(BUILD_DIR)/broadcast.o \\
            $(BUILD_DIR)/tensor.o $(BUILD_DIR)/loss.o \\
            $(BUILD_DIR)/optimizer.o $(BUILD_DIR)/serialize.o $(BUILD_DIR)/dataloader.o \\
            $(BUILD_DIR)/model_zoo.o $(BUILD_DIR)/onnx_export.o $(BUILD_DIR)/onnx_import.o $(BUILD_DIR)/device.o \\
            $(BUILD_DIR)/layer_linear.o $(BUILD_DIR)/layer_activations.o $(BUILD_DIR)/layer_conv.o \\
            $(BUILD_DIR)/layer_normalization.o $(BUILD_DIR)/layer_embedding.o $(BUILD_DIR)/layer_recurrent.o \\
            $(BUILD_DIR)/layer_attention.o $(BUILD_DIR)/layer_sequential.o \\
            $(BUILD_DIR)/simd_ops_avx.o $(BUILD_DIR)/simd_ops_neon.o $(BUILD_DIR)/simd_ops_fallback.o \\
            $(BUILD_DIR)/matmul_cpu.o $(BUILD_DIR)/im2col.o \\
            $(BUILD_DIR)/conv_ops.o $(BUILD_DIR)/augmentation.o \\
            $(BUILD_DIR)/cuda_stub.o $(BUILD_DIR)/metal_stub.o

all: train infer

$(BUILD_DIR):
\tmkdir -p $(BUILD_DIR)

# Core object compilation rules
$(BUILD_DIR)/memory_pool.o: $(CORE_DIR)/memory_pool.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -c -o $@ $<

$(BUILD_DIR)/autograd.o: $(CORE_DIR)/autograd.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -c -o $@ $<

$(BUILD_DIR)/broadcast.o: $(CORE_DIR)/broadcast.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -c -o $@ $<

$(BUILD_DIR)/tensor.o: $(CORE_DIR)/tensor.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -c -o $@ $<

$(BUILD_DIR)/loss.o: $(CORE_DIR)/loss.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -c -o $@ $<

$(BUILD_DIR)/optimizer.o: $(CORE_DIR)/optimizer.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -c -o $@ $<

$(BUILD_DIR)/serialize.o: $(SERIAL_DIR)/serialize.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -c -o $@ $<

$(BUILD_DIR)/dataloader.o: $(CORE_DIR)/dataloader.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -c -o $@ $<

$(BUILD_DIR)/model_zoo.o: $(CORE_DIR)/model_zoo.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -c -o $@ $<

$(BUILD_DIR)/onnx_export.o: $(SERIAL_DIR)/onnx_export.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -c -o $@ $<

$(BUILD_DIR)/onnx_import.o: $(SERIAL_DIR)/onnx_import.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -c -o $@ $<

$(BUILD_DIR)/device.o: $(CORE_DIR)/device.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -c -o $@ $<

$(BUILD_DIR)/layer_linear.o: $(LAYERS_DIR)/linear.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -c -o $@ $<

$(BUILD_DIR)/layer_activations.o: $(LAYERS_DIR)/activations.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -c -o $@ $<

$(BUILD_DIR)/layer_conv.o: $(LAYERS_DIR)/conv.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -c -o $@ $<

$(BUILD_DIR)/layer_normalization.o: $(LAYERS_DIR)/normalization.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -c -o $@ $<

$(BUILD_DIR)/layer_embedding.o: $(LAYERS_DIR)/embedding.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -c -o $@ $<

$(BUILD_DIR)/layer_recurrent.o: $(LAYERS_DIR)/recurrent.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -c -o $@ $<

$(BUILD_DIR)/layer_attention.o: $(LAYERS_DIR)/attention.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -c -o $@ $<

$(BUILD_DIR)/layer_sequential.o: $(LAYERS_DIR)/sequential.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -c -o $@ $<

$(BUILD_DIR)/simd_ops_avx.o: $(OPS_DIR)/simd_ops_avx.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -I$(OPS_DIR) -c -o $@ $<

$(BUILD_DIR)/simd_ops_neon.o: $(OPS_DIR)/simd_ops_neon.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -I$(OPS_DIR) -c -o $@ $<

$(BUILD_DIR)/simd_ops_fallback.o: $(OPS_DIR)/simd_ops_fallback.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -I$(OPS_DIR) -c -o $@ $<

$(BUILD_DIR)/matmul_cpu.o: $(OPS_DIR)/matmul_cpu.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -I$(OPS_DIR) -c -o $@ $<

$(BUILD_DIR)/im2col.o: $(OPS_DIR)/im2col.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -I$(OPS_DIR) -c -o $@ $<

$(BUILD_DIR)/conv_ops.o: $(OPS_DIR)/conv_ops.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -I$(OPS_DIR) -c -o $@ $<

$(BUILD_DIR)/augmentation.o: $(OPS_DIR)/augmentation.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -c -o $@ $<

$(BUILD_DIR)/cuda_stub.o: $(CORE_DIR)/cuda/cuda_stub.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -c -o $@ $<

$(BUILD_DIR)/metal_stub.o: $(CORE_DIR)/metal/metal_stub.cpp | $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -c -o $@ $<

# Build static library
$(WHITEMATTER_LIB): $(CORE_OBJS)
\tar rcs $@ $^

# Build train and infer binaries
train: train.cpp $(WHITEMATTER_LIB)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -o $@ $< -L$(BUILD_DIR) -lwhitematter $(LDFLAGS)

infer: infer.cpp $(WHITEMATTER_LIB)
\t$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -o $@ $< -L$(BUILD_DIR) -lwhitematter $(LDFLAGS)

clean:
\trm -rf $(BUILD_DIR) train infer

.PHONY: all clean
'''
