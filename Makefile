CXX = g++

# Directories
CORE_DIR = core
DATASETS_DIR = datasets
EXAMPLES_DIR = examples
BUILD_DIR = build

# Detect architecture for SIMD flags
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

# Core library objects
CORE_SRCS = $(CORE_DIR)/tensor.cpp $(CORE_DIR)/layer.cpp $(CORE_DIR)/loss.cpp \
            $(CORE_DIR)/optimizer.cpp $(CORE_DIR)/serialize.cpp $(CORE_DIR)/dataloader.cpp \
            $(CORE_DIR)/model_zoo.cpp $(CORE_DIR)/onnx_export.cpp
CORE_OBJS = $(BUILD_DIR)/tensor.o $(BUILD_DIR)/layer.o $(BUILD_DIR)/loss.o \
            $(BUILD_DIR)/optimizer.o $(BUILD_DIR)/serialize.o $(BUILD_DIR)/dataloader.o \
            $(BUILD_DIR)/model_zoo.o $(BUILD_DIR)/onnx_export.o

# Dataset objects
DATASET_SRCS = $(DATASETS_DIR)/mnist.cpp $(DATASETS_DIR)/cifar10.cpp
DATASET_OBJS = $(BUILD_DIR)/mnist.o $(BUILD_DIR)/cifar10.o

# All library objects
LIB_OBJS = $(CORE_OBJS) $(DATASET_OBJS)

# Test directory
TESTS_DIR = tests

# Example targets
ML_TARGET = $(BUILD_DIR)/ml
CNN_MNIST_TARGET = $(BUILD_DIR)/cnn_mnist
CNN_CIFAR10_TARGET = $(BUILD_DIR)/cnn_cifar10
TRANSFORMER_TARGET = $(BUILD_DIR)/transformer_example
AUTOENCODER_TARGET = $(BUILD_DIR)/autoencoder
GAN_TARGET = $(BUILD_DIR)/gan
RNN_TEXT_GEN_TARGET = $(BUILD_DIR)/rnn_text_gen
TESTS_TARGET = $(BUILD_DIR)/run_tests

# Test source files
TEST_SRCS = $(TESTS_DIR)/test_tensor.cpp $(TESTS_DIR)/test_autograd.cpp \
            $(TESTS_DIR)/test_layers.cpp $(TESTS_DIR)/test_loss.cpp \
            $(TESTS_DIR)/test_optimizer.cpp $(TESTS_DIR)/run_tests.cpp
TEST_OBJS = $(BUILD_DIR)/test_tensor.o $(BUILD_DIR)/test_autograd.o \
            $(BUILD_DIR)/test_layers.o $(BUILD_DIR)/test_loss.o \
            $(BUILD_DIR)/test_optimizer.o $(BUILD_DIR)/run_tests.o

all: $(ML_TARGET) $(CNN_MNIST_TARGET) $(CNN_CIFAR10_TARGET) $(TRANSFORMER_TARGET) $(AUTOENCODER_TARGET) $(GAN_TARGET) $(RNN_TEXT_GEN_TARGET)

# Ensure build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Core library compilation
$(BUILD_DIR)/tensor.o: $(CORE_DIR)/tensor.cpp $(CORE_DIR)/tensor.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/layer.o: $(CORE_DIR)/layer.cpp $(CORE_DIR)/layer.h $(CORE_DIR)/tensor.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/loss.o: $(CORE_DIR)/loss.cpp $(CORE_DIR)/loss.h $(CORE_DIR)/tensor.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/optimizer.o: $(CORE_DIR)/optimizer.cpp $(CORE_DIR)/optimizer.h $(CORE_DIR)/tensor.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/serialize.o: $(CORE_DIR)/serialize.cpp $(CORE_DIR)/serialize.h $(CORE_DIR)/tensor.h $(CORE_DIR)/layer.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/dataloader.o: $(CORE_DIR)/dataloader.cpp $(CORE_DIR)/dataloader.h $(CORE_DIR)/tensor.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/model_zoo.o: $(CORE_DIR)/model_zoo.cpp $(CORE_DIR)/model_zoo.h $(CORE_DIR)/layer.h $(CORE_DIR)/serialize.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/onnx_export.o: $(CORE_DIR)/onnx_export.cpp $(CORE_DIR)/onnx_export.h $(CORE_DIR)/layer.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Dataset compilation
$(BUILD_DIR)/mnist.o: $(DATASETS_DIR)/mnist.cpp $(DATASETS_DIR)/mnist.h $(CORE_DIR)/tensor.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/cifar10.o: $(DATASETS_DIR)/cifar10.cpp $(DATASETS_DIR)/cifar10.h $(CORE_DIR)/tensor.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Example compilation
$(BUILD_DIR)/ml.o: $(EXAMPLES_DIR)/ml.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/cnn_mnist.o: $(EXAMPLES_DIR)/cnn_mnist.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/cnn_cifar10.o: $(EXAMPLES_DIR)/cnn_cifar10.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/transformer_example.o: $(EXAMPLES_DIR)/transformer_example.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/autoencoder.o: $(EXAMPLES_DIR)/autoencoder.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/gan.o: $(EXAMPLES_DIR)/gan.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/rnn_text_gen.o: $(EXAMPLES_DIR)/rnn_text_gen.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Link examples
$(ML_TARGET): $(BUILD_DIR)/ml.o $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(CNN_MNIST_TARGET): $(BUILD_DIR)/cnn_mnist.o $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(CNN_CIFAR10_TARGET): $(BUILD_DIR)/cnn_cifar10.o $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(TRANSFORMER_TARGET): $(BUILD_DIR)/transformer_example.o $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(AUTOENCODER_TARGET): $(BUILD_DIR)/autoencoder.o $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(GAN_TARGET): $(BUILD_DIR)/gan.o $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(RNN_TEXT_GEN_TARGET): $(BUILD_DIR)/rnn_text_gen.o $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Test compilation
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

$(BUILD_DIR)/run_tests.o: $(TESTS_DIR)/run_tests.cpp $(TESTS_DIR)/test_framework.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(TESTS_DIR) -c -o $@ $<

# Link tests
$(TESTS_TARGET): $(TEST_OBJS) $(CORE_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Test targets
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

clean:
	rm -rf $(BUILD_DIR)/*.o $(BUILD_DIR)/ml $(BUILD_DIR)/cnn_mnist $(BUILD_DIR)/cnn_cifar10 $(BUILD_DIR)/transformer_example $(BUILD_DIR)/autoencoder $(BUILD_DIR)/gan $(BUILD_DIR)/rnn_text_gen $(BUILD_DIR)/run_tests

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

.PHONY: all clean run run-cnn run-cifar run-transformer run-autoencoder autoencoder run-gan gan debug test test-tensor test-autograd test-layers test-loss test-optimizer
