CXX = g++

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
LDFLAGS = $(OPENMP_LIBS)

SOURCES = ml.cpp tensor.cpp layer.cpp loss.cpp optimizer.cpp mnist.cpp cifar10.cpp serialize.cpp
HEADERS = tensor.h layer.h loss.h optimizer.h mnist.h cifar10.h serialize.h
OBJECTS = $(SOURCES:.cpp=.o)
LIB_OBJECTS = tensor.o layer.o loss.o optimizer.o mnist.o cifar10.o serialize.o
TARGET = ml

all: $(TARGET) cnn_mnist cnn_cifar10 transformer_example

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

cnn_mnist: cnn_mnist.o $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

cnn_cifar10: cnn_cifar10.o $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

transformer_example: transformer_example.o $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJECTS) cnn_mnist.o cnn_cifar10.o transformer_example.o cifar10.o $(TARGET) cnn_mnist cnn_cifar10 transformer_example

run: $(TARGET)
	./$(TARGET)

run-cnn: cnn_mnist
	./cnn_mnist

run-cifar: cnn_cifar10
	./cnn_cifar10

run-transformer: transformer_example
	./transformer_example

debug: CXXFLAGS = -std=c++17 -O0 -g -Wall -Wextra
debug: clean $(TARGET)

.PHONY: all clean run run-cnn run-cifar run-transformer debug
