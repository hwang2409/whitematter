# Inference executable Makefile
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -ffast-math

# Detect macOS and add OpenMP flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CXXFLAGS += -mcpu=apple-m1 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
    LDFLAGS += -L/opt/homebrew/opt/libomp/lib -lomp
endif

# Path to whitematter core
PROJECT_ROOT = ../..
CORE_DIR = $(PROJECT_ROOT)/core
BUILD_DIR = $(PROJECT_ROOT)/build

OBJS = $(BUILD_DIR)/tensor.o $(BUILD_DIR)/layer.o $(BUILD_DIR)/serialize.o

infer: infer.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -o $@ $^ $(LDFLAGS)

clean:
	rm -f infer

.PHONY: clean
