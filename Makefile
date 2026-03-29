CXX = g++

CORE_DIR = core
DATASETS_DIR = datasets
EXAMPLES_DIR = examples
BUILD_DIR = build

UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),arm64)
    SIMD_FLAGS = -mcpu=apple-m1
else ifeq ($(UNAME_M),aarch64)
    SIMD_FLAGS = -march=armv8-a+simd
else
    SIMD_FLAGS = -mavx -mfma
endif

# OpenMP flags (macOS needs Homebrew libomp: brew install libomp)
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    OPENMP_FLAGS = -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
    OPENMP_LIBS = -L/opt/homebrew/opt/libomp/lib -lomp
else
    OPENMP_FLAGS = -fopenmp
    OPENMP_LIBS = -fopenmp
endif

CXXFLAGS = -std=c++17 -O3 -Wall -Wextra $(SIMD_FLAGS) -ffast-math -funroll-loops $(OPENMP_FLAGS)
CXXFLAGS += -I$(CORE_DIR) -I$(DATASETS_DIR)
LDFLAGS = $(OPENMP_LIBS)

# Use Apple Accelerate BLAS for fast matmul on macOS
ifeq ($(UNAME_S),Darwin)
    CXXFLAGS += -DACCELERATE_NEW_LAPACK
    LDFLAGS += -framework Accelerate
endif

# Debug mode: make DEBUG=1 for verbose CUDA/cuDNN logging
DEBUG ?= 0
ifeq ($(DEBUG),1)
    CXXFLAGS += -DWHITEMATTER_DEBUG
endif

# Use OpenBLAS for fast matmul on Linux: make OPENBLAS=1
OPENBLAS ?= 0
ifeq ($(OPENBLAS),1)
    CXXFLAGS += -DWHITEMATTER_OPENBLAS
    LDFLAGS += -lopenblas
endif

METAL ?= 0
CUDA ?= 0

LAYERS_DIR = $(CORE_DIR)/layers
SERIAL_DIR = $(CORE_DIR)/serialization
CORE_SRCS = $(CORE_DIR)/memory_pool.cpp $(CORE_DIR)/autograd.cpp $(CORE_DIR)/broadcast.cpp \
            $(CORE_DIR)/tensor.cpp $(CORE_DIR)/loss.cpp \
            $(CORE_DIR)/optimizer.cpp $(SERIAL_DIR)/serialize.cpp $(CORE_DIR)/dataloader.cpp \
            $(CORE_DIR)/model_zoo.cpp $(SERIAL_DIR)/onnx_export.cpp $(SERIAL_DIR)/onnx_import.cpp $(CORE_DIR)/device.cpp \
            $(LAYERS_DIR)/linear.cpp $(LAYERS_DIR)/activations.cpp $(LAYERS_DIR)/conv.cpp \
            $(LAYERS_DIR)/normalization.cpp $(LAYERS_DIR)/embedding.cpp $(LAYERS_DIR)/recurrent.cpp \
            $(LAYERS_DIR)/attention.cpp $(LAYERS_DIR)/kv_cache.cpp $(LAYERS_DIR)/sequential.cpp $(LAYERS_DIR)/positional.cpp \
            $(LAYERS_DIR)/upsample.cpp \
            $(CORE_DIR)/ops/simd_ops_avx.cpp $(CORE_DIR)/ops/simd_ops_neon.cpp $(CORE_DIR)/ops/simd_ops_fallback.cpp \
            $(CORE_DIR)/ops/matmul_cpu.cpp $(CORE_DIR)/ops/im2col.cpp \
            $(CORE_DIR)/ops/conv_ops.cpp $(CORE_DIR)/ops/augmentation.cpp \
            $(CORE_DIR)/ops/fp16.cpp
CORE_OBJS = $(BUILD_DIR)/memory_pool.o $(BUILD_DIR)/autograd.o $(BUILD_DIR)/broadcast.o \
            $(BUILD_DIR)/tensor.o $(BUILD_DIR)/loss.o \
            $(BUILD_DIR)/optimizer.o $(BUILD_DIR)/serialize.o $(BUILD_DIR)/dataloader.o \
            $(BUILD_DIR)/model_zoo.o $(BUILD_DIR)/onnx_export.o $(BUILD_DIR)/onnx_import.o $(BUILD_DIR)/device.o \
            $(BUILD_DIR)/layer_linear.o $(BUILD_DIR)/layer_activations.o $(BUILD_DIR)/layer_conv.o \
            $(BUILD_DIR)/layer_normalization.o $(BUILD_DIR)/layer_embedding.o $(BUILD_DIR)/layer_recurrent.o \
            $(BUILD_DIR)/layer_attention.o $(BUILD_DIR)/layer_kv_cache.o $(BUILD_DIR)/layer_sequential.o $(BUILD_DIR)/layer_positional.o \
            $(BUILD_DIR)/layer_upsample.o \
            $(BUILD_DIR)/simd_ops_avx.o $(BUILD_DIR)/simd_ops_neon.o $(BUILD_DIR)/simd_ops_fallback.o \
            $(BUILD_DIR)/matmul_cpu.o $(BUILD_DIR)/im2col.o \
            $(BUILD_DIR)/conv_ops.o $(BUILD_DIR)/augmentation.o \
            $(BUILD_DIR)/fp16.o

ifeq ($(METAL),1)
  ifeq ($(UNAME_S),Darwin)
    CORE_OBJS += $(BUILD_DIR)/metal_backend.o
    CXXFLAGS += -DWHITEMATTER_METAL
    LDFLAGS += -framework Metal -framework Foundation -framework MetalPerformanceShaders
  else
    CORE_OBJS += $(BUILD_DIR)/metal_stub.o
  endif
else
  CORE_OBJS += $(BUILD_DIR)/metal_stub.o
endif

ifeq ($(CUDA),1)
  CORE_OBJS += $(BUILD_DIR)/cuda_backend.o $(BUILD_DIR)/cuda_memory.o $(BUILD_DIR)/cuda_tensor_ops.o
  CXXFLAGS += -DWHITEMATTER_CUDA
  LDFLAGS += -lcudart -lcublas -lcudnn
  ifneq ($(CUDA_PATH),)
    LDFLAGS += -L$(CUDA_PATH)/lib64
    NVCC_PREFIX = $(CUDA_PATH)/bin/
  endif
  NVCC = $(NVCC_PREFIX)nvcc
  NVCC_FLAGS = -std=c++17 -O3 --gpu-architecture=sm_75 -I$(CORE_DIR) -I$(CORE_DIR)/cuda
else
  CORE_OBJS += $(BUILD_DIR)/cuda_stub.o
endif

DATASET_SRCS = $(DATASETS_DIR)/mnist.cpp $(DATASETS_DIR)/cifar10.cpp
DATASET_OBJS = $(BUILD_DIR)/mnist.o $(BUILD_DIR)/cifar10.o

LIB_OBJS = $(CORE_OBJS) $(DATASET_OBJS)

STATIC_LIB = $(BUILD_DIR)/libwhitematter.a

TESTS_DIR = tests

ML_TARGET = $(BUILD_DIR)/ml
CNN_MNIST_TARGET = $(BUILD_DIR)/cnn_mnist
CNN_CIFAR10_TARGET = $(BUILD_DIR)/cnn_cifar10
CATS_DOGS_TARGET = $(BUILD_DIR)/cats_vs_dogs
TRANSFORMER_TARGET = $(BUILD_DIR)/transformer_example
AUTOENCODER_TARGET = $(BUILD_DIR)/autoencoder
GAN_TARGET = $(BUILD_DIR)/gan
RNN_TEXT_GEN_TARGET = $(BUILD_DIR)/rnn_text_gen
RESNET18_TARGET = $(BUILD_DIR)/resnet18_cifar10
RESNET18_CUDA_TARGET = $(BUILD_DIR)/resnet18_cifar10_cuda
RESNET18_PREDICT_TARGET = $(BUILD_DIR)/resnet18_predict
RESNET18_EXPORT_TARGET = $(BUILD_DIR)/resnet18_export
MOBILENETV2_TARGET = $(BUILD_DIR)/mobilenetv2_cifar10
RESNET18_IMAGENETTE_TARGET = $(BUILD_DIR)/resnet18_imagenette
GPT_SHAKESPEARE_TARGET = $(BUILD_DIR)/gpt_shakespeare
TESTS_TARGET = $(BUILD_DIR)/run_tests

TEST_SRCS = $(TESTS_DIR)/test_tensor.cpp $(TESTS_DIR)/test_autograd.cpp \
            $(TESTS_DIR)/test_layers.cpp $(TESTS_DIR)/test_loss.cpp \
            $(TESTS_DIR)/test_optimizer.cpp $(TESTS_DIR)/test_grad_check.cpp \
            $(TESTS_DIR)/run_tests.cpp
TEST_OBJS = $(BUILD_DIR)/test_tensor.o $(BUILD_DIR)/test_autograd.o \
            $(BUILD_DIR)/test_layers.o $(BUILD_DIR)/test_loss.o \
            $(BUILD_DIR)/test_optimizer.o $(BUILD_DIR)/test_grad_check.o \
            $(BUILD_DIR)/run_tests.o

all: $(STATIC_LIB) $(ML_TARGET) $(CNN_MNIST_TARGET) $(CNN_CIFAR10_TARGET) $(TRANSFORMER_TARGET) $(AUTOENCODER_TARGET) $(GAN_TARGET) $(RNN_TEXT_GEN_TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/memory_pool.o: $(CORE_DIR)/memory_pool.cpp $(CORE_DIR)/memory_pool.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/autograd.o: $(CORE_DIR)/autograd.cpp $(CORE_DIR)/autograd.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/broadcast.o: $(CORE_DIR)/broadcast.cpp $(CORE_DIR)/broadcast.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/simd_ops_avx.o: $(CORE_DIR)/ops/simd_ops_avx.cpp $(CORE_DIR)/ops/simd_ops.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(CORE_DIR)/ops -c -o $@ $<

$(BUILD_DIR)/simd_ops_neon.o: $(CORE_DIR)/ops/simd_ops_neon.cpp $(CORE_DIR)/ops/simd_ops.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(CORE_DIR)/ops -c -o $@ $<

$(BUILD_DIR)/simd_ops_fallback.o: $(CORE_DIR)/ops/simd_ops_fallback.cpp $(CORE_DIR)/ops/simd_ops.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(CORE_DIR)/ops -c -o $@ $<

$(BUILD_DIR)/matmul_cpu.o: $(CORE_DIR)/ops/matmul_cpu.cpp $(CORE_DIR)/ops/matmul_cpu.h $(CORE_DIR)/ops/simd_ops.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(CORE_DIR)/ops -c -o $@ $<

$(BUILD_DIR)/im2col.o: $(CORE_DIR)/ops/im2col.cpp $(CORE_DIR)/ops/im2col.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(CORE_DIR)/ops -c -o $@ $<

$(BUILD_DIR)/conv_ops.o: $(CORE_DIR)/ops/conv_ops.cpp $(CORE_DIR)/tensor.h $(CORE_DIR)/ops/im2col.h $(CORE_DIR)/ops/matmul_cpu.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(CORE_DIR)/ops -c -o $@ $<

$(BUILD_DIR)/augmentation.o: $(CORE_DIR)/ops/augmentation.cpp $(CORE_DIR)/tensor.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/fp16.o: $(CORE_DIR)/ops/fp16.cpp $(CORE_DIR)/ops/fp16.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -c -o $@ $<

$(BUILD_DIR)/tensor.o: $(CORE_DIR)/tensor.cpp $(CORE_DIR)/tensor.h $(CORE_DIR)/memory_pool.h $(CORE_DIR)/broadcast.h $(CORE_DIR)/ops/simd_ops.h $(CORE_DIR)/ops/matmul_cpu.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/layer_linear.o: $(LAYERS_DIR)/linear.cpp $(CORE_DIR)/layer.h $(CORE_DIR)/tensor.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/layer_activations.o: $(LAYERS_DIR)/activations.cpp $(CORE_DIR)/layer.h $(CORE_DIR)/tensor.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/layer_conv.o: $(LAYERS_DIR)/conv.cpp $(CORE_DIR)/layer.h $(CORE_DIR)/tensor.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/layer_normalization.o: $(LAYERS_DIR)/normalization.cpp $(CORE_DIR)/layer.h $(CORE_DIR)/tensor.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/layer_embedding.o: $(LAYERS_DIR)/embedding.cpp $(CORE_DIR)/layer.h $(CORE_DIR)/tensor.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/layer_recurrent.o: $(LAYERS_DIR)/recurrent.cpp $(CORE_DIR)/layer.h $(CORE_DIR)/tensor.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/layer_attention.o: $(LAYERS_DIR)/attention.cpp $(CORE_DIR)/layer.h $(CORE_DIR)/tensor.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/layer_kv_cache.o: $(LAYERS_DIR)/kv_cache.cpp $(CORE_DIR)/layer.h $(CORE_DIR)/tensor.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/layer_sequential.o: $(LAYERS_DIR)/sequential.cpp $(CORE_DIR)/layer.h $(CORE_DIR)/tensor.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/layer_positional.o: $(LAYERS_DIR)/positional.cpp $(CORE_DIR)/layer.h $(CORE_DIR)/tensor.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/layer_upsample.o: $(LAYERS_DIR)/upsample.cpp $(CORE_DIR)/layer.h $(CORE_DIR)/tensor.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/loss.o: $(CORE_DIR)/loss.cpp $(CORE_DIR)/loss.h $(CORE_DIR)/tensor.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/optimizer.o: $(CORE_DIR)/optimizer.cpp $(CORE_DIR)/optimizer.h $(CORE_DIR)/tensor.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/serialize.o: $(SERIAL_DIR)/serialize.cpp $(CORE_DIR)/serialize.h $(CORE_DIR)/tensor.h $(CORE_DIR)/layer.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/dataloader.o: $(CORE_DIR)/dataloader.cpp $(CORE_DIR)/dataloader.h $(CORE_DIR)/tensor.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/model_zoo.o: $(CORE_DIR)/model_zoo.cpp $(CORE_DIR)/model_zoo.h $(CORE_DIR)/layer.h $(CORE_DIR)/serialize.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/onnx_export.o: $(SERIAL_DIR)/onnx_export.cpp $(CORE_DIR)/onnx_export.h $(CORE_DIR)/layer.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/onnx_import.o: $(SERIAL_DIR)/onnx_import.cpp $(CORE_DIR)/onnx_import.h $(CORE_DIR)/onnx_export.h $(CORE_DIR)/layer.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/device.o: $(CORE_DIR)/device.cpp $(CORE_DIR)/device.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/metal_stub.o: $(CORE_DIR)/metal/metal_stub.cpp $(CORE_DIR)/device.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/metal_backend.o: $(CORE_DIR)/metal/metal_backend.mm $(CORE_DIR)/metal/metal_backend.h $(CORE_DIR)/device.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(CORE_DIR)/metal -c -o $@ $<

$(BUILD_DIR)/cuda_stub.o: $(CORE_DIR)/cuda/cuda_stub.cpp $(CORE_DIR)/device.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -c -o $@ $<

$(BUILD_DIR)/cuda_backend.o: $(CORE_DIR)/cuda/cuda_backend.cu $(CORE_DIR)/cuda/cuda_backend.h $(CORE_DIR)/cuda/cuda_check.h $(CORE_DIR)/device.h | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -DWHITEMATTER_CUDA -c -o $@ $<

$(BUILD_DIR)/cuda_memory.o: $(CORE_DIR)/cuda/cuda_memory.cu $(CORE_DIR)/cuda/cuda_memory.h $(CORE_DIR)/cuda/cuda_check.h | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -DWHITEMATTER_CUDA -c -o $@ $<

$(BUILD_DIR)/cuda_tensor_ops.o: $(CORE_DIR)/cuda/cuda_tensor_ops.cpp $(CORE_DIR)/cuda/cuda_tensor_ops.h $(CORE_DIR)/cuda/cuda_backend.h $(CORE_DIR)/cuda/cuda_memory.h $(CORE_DIR)/tensor.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -I$(CORE_DIR)/cuda -c -o $@ $<
$(BUILD_DIR)/mnist.o: $(DATASETS_DIR)/mnist.cpp $(DATASETS_DIR)/mnist.h $(CORE_DIR)/tensor.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/cifar10.o: $(DATASETS_DIR)/cifar10.cpp $(DATASETS_DIR)/cifar10.h $(CORE_DIR)/tensor.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/ml.o: $(EXAMPLES_DIR)/ml.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/cnn_mnist.o: $(EXAMPLES_DIR)/cnn_mnist.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/cnn_cifar10.o: $(EXAMPLES_DIR)/cnn_cifar10.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/cats_vs_dogs.o: $(EXAMPLES_DIR)/cats_vs_dogs.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/transformer_example.o: $(EXAMPLES_DIR)/transformer_example.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/autoencoder.o: $(EXAMPLES_DIR)/autoencoder.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/gan.o: $(EXAMPLES_DIR)/gan.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/rnn_text_gen.o: $(EXAMPLES_DIR)/rnn_text_gen.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/resnet18_cifar10.o: $(EXAMPLES_DIR)/resnet18_cifar10.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/resnet18_cuda.o: $(EXAMPLES_DIR)/resnet18_cifar10_cuda.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/resnet18_predict.o: $(EXAMPLES_DIR)/resnet18_predict.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/resnet18_export.o: $(EXAMPLES_DIR)/resnet18_export.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/mobilenetv2_cifar10.o: $(EXAMPLES_DIR)/mobilenetv2_cifar10.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/gpt_shakespeare.o: $(EXAMPLES_DIR)/gpt_shakespeare.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/resnet18_imagenette.o: $(EXAMPLES_DIR)/resnet18_imagenette.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(STATIC_LIB): $(LIB_OBJS)
	ar rcs $@ $^

$(ML_TARGET): $(BUILD_DIR)/ml.o $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lwhitematter $(LDFLAGS)

$(CNN_MNIST_TARGET): $(BUILD_DIR)/cnn_mnist.o $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lwhitematter $(LDFLAGS)

$(CNN_CIFAR10_TARGET): $(BUILD_DIR)/cnn_cifar10.o $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lwhitematter $(LDFLAGS)

$(CATS_DOGS_TARGET): $(BUILD_DIR)/cats_vs_dogs.o $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lwhitematter $(LDFLAGS)

cats_dogs: $(CATS_DOGS_TARGET)

$(TRANSFORMER_TARGET): $(BUILD_DIR)/transformer_example.o $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lwhitematter $(LDFLAGS)

$(AUTOENCODER_TARGET): $(BUILD_DIR)/autoencoder.o $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lwhitematter $(LDFLAGS)

$(GAN_TARGET): $(BUILD_DIR)/gan.o $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lwhitematter $(LDFLAGS)

$(RNN_TEXT_GEN_TARGET): $(BUILD_DIR)/rnn_text_gen.o $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lwhitematter $(LDFLAGS)

$(RESNET18_TARGET): $(BUILD_DIR)/resnet18_cifar10.o $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lwhitematter $(LDFLAGS)

resnet18: $(RESNET18_TARGET)

run-resnet18: $(RESNET18_TARGET)
	./$(RESNET18_TARGET)

$(RESNET18_CUDA_TARGET): $(BUILD_DIR)/resnet18_cuda.o $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lwhitematter $(LDFLAGS)

resnet18-cuda: $(RESNET18_CUDA_TARGET)

run-resnet18-cuda: $(RESNET18_CUDA_TARGET)
	./$(RESNET18_CUDA_TARGET)

$(RESNET18_PREDICT_TARGET): $(BUILD_DIR)/resnet18_predict.o $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lwhitematter $(LDFLAGS)

resnet18-predict: $(RESNET18_PREDICT_TARGET)

$(RESNET18_EXPORT_TARGET): $(BUILD_DIR)/resnet18_export.o $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lwhitematter $(LDFLAGS)

resnet18-export: $(RESNET18_EXPORT_TARGET)

$(MOBILENETV2_TARGET): $(BUILD_DIR)/mobilenetv2_cifar10.o $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lwhitematter $(LDFLAGS)

mobilenetv2: $(MOBILENETV2_TARGET)

run-mobilenetv2: $(MOBILENETV2_TARGET)
	./$(MOBILENETV2_TARGET)

$(GPT_SHAKESPEARE_TARGET): $(BUILD_DIR)/gpt_shakespeare.o $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lwhitematter $(LDFLAGS)

gpt_shakespeare: $(GPT_SHAKESPEARE_TARGET)

run-gpt: $(GPT_SHAKESPEARE_TARGET)
	./$(GPT_SHAKESPEARE_TARGET)

$(RESNET18_IMAGENETTE_TARGET): $(BUILD_DIR)/resnet18_imagenette.o $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lwhitematter $(LDFLAGS)

resnet18-imagenette: $(RESNET18_IMAGENETTE_TARGET)

run-resnet18-imagenette: $(RESNET18_IMAGENETTE_TARGET)
	./$(RESNET18_IMAGENETTE_TARGET)

$(BUILD_DIR)/test_tensor.o: $(TESTS_DIR)/test_tensor.cpp $(TESTS_DIR)/test_framework.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(TESTS_DIR) -c -o $@ $<

$(BUILD_DIR)/test_autograd.o: $(TESTS_DIR)/test_autograd.cpp $(TESTS_DIR)/test_framework.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(TESTS_DIR) -c -o $@ $<

$(BUILD_DIR)/test_layers.o: $(TESTS_DIR)/test_layers.cpp $(TESTS_DIR)/test_framework.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(TESTS_DIR) -c -o $@ $<

$(BUILD_DIR)/test_loss.o: $(TESTS_DIR)/test_loss.cpp $(TESTS_DIR)/test_framework.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(TESTS_DIR) -c -o $@ $<

$(BUILD_DIR)/test_optimizer.o: $(TESTS_DIR)/test_optimizer.cpp $(TESTS_DIR)/test_framework.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(TESTS_DIR) -c -o $@ $<

$(BUILD_DIR)/test_grad_check.o: $(TESTS_DIR)/test_grad_check.cpp $(TESTS_DIR)/test_framework.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(TESTS_DIR) -c -o $@ $<

$(BUILD_DIR)/run_tests.o: $(TESTS_DIR)/run_tests.cpp $(TESTS_DIR)/test_framework.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(TESTS_DIR) -c -o $@ $<

$(TESTS_TARGET): $(TEST_OBJS) $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_OBJS) -L$(BUILD_DIR) -lwhitematter $(LDFLAGS)

test: $(TESTS_TARGET)
	./$(TESTS_TARGET)

test-tensor: $(TESTS_TARGET)
	./$(TESTS_TARGET) --tensor

test-autograd: $(TESTS_TARGET)
	./$(TESTS_TARGET) --autograd

test-layers: $(TESTS_TARGET)
	./$(TESTS_TARGET) --layers

test-loss: $(TESTS_TARGET)
	./$(TESTS_TARGET) --loss

test-optimizer: $(TESTS_TARGET)
	./$(TESTS_TARGET) --optimizer

test-gradcheck: $(TESTS_TARGET)
	./$(TESTS_TARGET) --gradcheck

clean:
	rm -rf $(BUILD_DIR)/*.o $(BUILD_DIR)/*.a $(BUILD_DIR)/ml $(BUILD_DIR)/cnn_mnist $(BUILD_DIR)/cnn_cifar10 $(BUILD_DIR)/cats_vs_dogs $(BUILD_DIR)/transformer_example $(BUILD_DIR)/autoencoder $(BUILD_DIR)/gan $(BUILD_DIR)/rnn_text_gen $(BUILD_DIR)/resnet18_cifar10 $(BUILD_DIR)/resnet18_cifar10_cuda $(BUILD_DIR)/resnet18_predict $(BUILD_DIR)/resnet18_export $(BUILD_DIR)/resnet18_imagenette $(BUILD_DIR)/mobilenetv2_cifar10 $(BUILD_DIR)/gpt_shakespeare $(BUILD_DIR)/run_tests

run: $(ML_TARGET)
	./$(ML_TARGET)

run-cnn: $(CNN_MNIST_TARGET)
	./$(CNN_MNIST_TARGET)

run-cifar: $(CNN_CIFAR10_TARGET)
	./$(CNN_CIFAR10_TARGET)

run-transformer: $(TRANSFORMER_TARGET)
	./$(TRANSFORMER_TARGET)

run-autoencoder: $(AUTOENCODER_TARGET)
	./$(AUTOENCODER_TARGET)

autoencoder: $(AUTOENCODER_TARGET)

run-gan: $(GAN_TARGET)
	./$(GAN_TARGET)

gan: $(GAN_TARGET)

run-rnn: $(RNN_TEXT_GEN_TARGET)
	./$(RNN_TEXT_GEN_TARGET)

rnn: $(RNN_TEXT_GEN_TARGET)

debug: CXXFLAGS = -std=c++17 -O0 -g -Wall -Wextra -I$(CORE_DIR) -I$(DATASETS_DIR)
debug: clean $(ML_TARGET)

dev:
	@echo "Starting backend and frontend..."
	@cd platform && python server.py & PID=$$!; \
	trap "kill $$PID 2>/dev/null; exit" INT TERM EXIT; \
	cd frontend && npm run dev; \
	kill $$PID 2>/dev/null

docker-build:
	docker compose build

lint:
	@echo "── C++ (cppcheck, if available) ──"
	-@which cppcheck > /dev/null 2>&1 && cppcheck --std=c++17 --quiet core/ || echo "  cppcheck not installed, skipping"
	@echo "── Python (ruff, if available) ──"
	-@cd platform && (python -m ruff check . 2>/dev/null || python -m flake8 . 2>/dev/null || echo "  No Python linter found, skipping")
	@echo "── Frontend (eslint) ──"
	@cd frontend && npm run lint

test-all: test
	@echo "── Platform tests ──"
	@cd platform && python -m pytest -q 2>/dev/null || echo "  No platform tests found"
	@echo "── Frontend lint ──"
	@cd frontend && npm run lint

.PHONY: all clean run run-cnn run-cifar run-transformer run-autoencoder autoencoder run-gan gan run-rnn rnn resnet18 run-resnet18 resnet18-cuda run-resnet18-cuda resnet18-predict resnet18-export resnet18-imagenette run-resnet18-imagenette mobilenetv2 run-mobilenetv2 gpt_shakespeare run-gpt debug test test-tensor test-autograd test-layers test-loss test-optimizer test-gradcheck dev docker-build lint test-all
