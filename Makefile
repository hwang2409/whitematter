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

CXXFLAGS = -std=c++17 -O3 -Wall -Wextra $(SIMD_FLAGS) -ffast-math -funroll-loops

SOURCES = ml.cpp tensor.cpp layer.cpp loss.cpp optimizer.cpp mnist.cpp
HEADERS = tensor.h layer.h loss.h optimizer.h mnist.h
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = ml

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJECTS) $(TARGET)

run: $(TARGET)
	./$(TARGET)

debug: CXXFLAGS = -std=c++17 -O0 -g -Wall -Wextra
debug: clean $(TARGET)

.PHONY: all clean run debug
