"""
Makefile template for compiling training and inference code.
"""


def generate_makefile() -> str:
    """Generate Makefile for compiling training and inference code."""
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
