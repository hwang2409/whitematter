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
            $(CORE_DIR)/optimizer.cpp $(CORE_DIR)/serialize.cpp $(CORE_DIR)/dataloader.cpp
CORE_OBJS = $(BUILD_DIR)/tensor.o $(BUILD_DIR)/layer.o $(BUILD_DIR)/loss.o \
            $(BUILD_DIR)/optimizer.o $(BUILD_DIR)/serialize.o $(BUILD_DIR)/dataloader.o

# Dataset objects
DATASET_SRCS = $(DATASETS_DIR)/mnist.cpp $(DATASETS_DIR)/cifar10.cpp
DATASET_OBJS = $(BUILD_DIR)/mnist.o $(BUILD_DIR)/cifar10.o

# All library objects
LIB_OBJS = $(CORE_OBJS) $(DATASET_OBJS)

# Example targets
ML_TARGET = $(BUILD_DIR)/ml
CNN_MNIST_TARGET = $(BUILD_DIR)/cnn_mnist
CNN_CIFAR10_TARGET = $(BUILD_DIR)/cnn_cifar10
TRANSFORMER_TARGET = $(BUILD_DIR)/transformer_example

all: $(ML_TARGET) $(CNN_MNIST_TARGET) $(CNN_CIFAR10_TARGET) $(TRANSFORMER_TARGET)

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

# Link examples
$(ML_TARGET): $(BUILD_DIR)/ml.o $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(CNN_MNIST_TARGET): $(BUILD_DIR)/cnn_mnist.o $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(CNN_CIFAR10_TARGET): $(BUILD_DIR)/cnn_cifar10.o $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(TRANSFORMER_TARGET): $(BUILD_DIR)/transformer_example.o $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)/*.o $(BUILD_DIR)/ml $(BUILD_DIR)/cnn_mnist $(BUILD_DIR)/cnn_cifar10 $(BUILD_DIR)/transformer_example

run: $(ML_TARGET)
	./$(ML_TARGET)

run-cnn: $(CNN_MNIST_TARGET)
	./$(CNN_MNIST_TARGET)

run-cifar: $(CNN_CIFAR10_TARGET)
	./$(CNN_CIFAR10_TARGET)

run-transformer: $(TRANSFORMER_TARGET)
	./$(TRANSFORMER_TARGET)

debug: CXXFLAGS = -std=c++17 -O0 -g -Wall -Wextra -I$(CORE_DIR) -I$(DATASETS_DIR)
debug: clean $(ML_TARGET)

.PHONY: all clean run run-cnn run-cifar run-transformer debug
