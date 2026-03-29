#ifndef MMAP_TENSOR_H
#define MMAP_TENSOR_H

// Memory-mapped tensor loader for large binary tensor files.
//
// Instead of malloc + fread (which loads the entire dataset into RAM),
// this uses mmap to map the file into virtual address space.  The OS
// manages the page cache -- only accessed pages stay resident in RAM.
// This is critical for large datasets (e.g. 5.7 GB ImageNette at 224x224)
// that would otherwise exhaust physical memory.
//
// Binary format (same as load_tensor_bin):
//   magic (4B = 0x54454E53) | ndim (4B) | shape (ndim * 8B) | float data

#include "tensor.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

struct MmapTensor {
    float* data;
    std::vector<size_t> shape;
    size_t total_elements;
    int fd;
    void* map_base;
    size_t map_size;

    MmapTensor() : data(nullptr), total_elements(0), fd(-1),
                   map_base(MAP_FAILED), map_size(0) {}

    // Move constructor
    MmapTensor(MmapTensor&& other) noexcept
        : data(other.data), shape(std::move(other.shape)),
          total_elements(other.total_elements), fd(other.fd),
          map_base(other.map_base), map_size(other.map_size) {
        other.data = nullptr;
        other.fd = -1;
        other.map_base = MAP_FAILED;
        other.map_size = 0;
    }

    // Move assignment
    MmapTensor& operator=(MmapTensor&& other) noexcept {
        if (this != &other) {
            cleanup();
            data = other.data;
            shape = std::move(other.shape);
            total_elements = other.total_elements;
            fd = other.fd;
            map_base = other.map_base;
            map_size = other.map_size;
            other.data = nullptr;
            other.fd = -1;
            other.map_base = MAP_FAILED;
            other.map_size = 0;
        }
        return *this;
    }

    // No copy
    MmapTensor(const MmapTensor&) = delete;
    MmapTensor& operator=(const MmapTensor&) = delete;

    ~MmapTensor() { cleanup(); }

    // Get a contiguous batch starting at index `start` with `count` samples.
    // Copies just that slice from the mmap region into a new Tensor.
    TensorPtr get_batch(size_t start, size_t count) const {
        size_t elements_per_sample = total_elements / shape[0];
        size_t batch_elements = count * elements_per_sample;

        std::vector<size_t> batch_shape = shape;
        batch_shape[0] = count;

        auto tensor = Tensor::create(batch_shape, false);
        std::memcpy(tensor->data(), data + start * elements_per_sample,
                     batch_elements * sizeof(float));
        return tensor;
    }

private:
    void cleanup() {
        if (map_base != MAP_FAILED && map_base != nullptr) {
            munmap(map_base, map_size);
            map_base = MAP_FAILED;
        }
        if (fd >= 0) {
            close(fd);
            fd = -1;
        }
        data = nullptr;
    }
};

// Load a binary tensor file via mmap instead of malloc+fread.
// The returned MmapTensor owns the mapping; data is valid until destruction.
inline MmapTensor load_tensor_mmap(const std::string& path) {
    MmapTensor result;
    result.fd = open(path.c_str(), O_RDONLY);
    if (result.fd < 0) {
        fprintf(stderr, "Failed to open %s\n", path.c_str());
        exit(1);
    }

    // Read header: magic + ndim + shape
    uint32_t magic, ndim;
    if (read(result.fd, &magic, 4) != 4) {
        fprintf(stderr, "Failed to read magic from %s\n", path.c_str());
        exit(1);
    }
    if (magic != 0x54454E53) {
        fprintf(stderr, "Bad magic in %s (expected 0x54454E53, got 0x%08X)\n",
                path.c_str(), magic);
        exit(1);
    }
    if (read(result.fd, &ndim, 4) != 4) {
        fprintf(stderr, "Failed to read ndim from %s\n", path.c_str());
        exit(1);
    }

    result.total_elements = 1;
    for (uint32_t i = 0; i < ndim; i++) {
        uint64_t dim;
        if (read(result.fd, &dim, 8) != 8) {
            fprintf(stderr, "Failed to read dim %u from %s\n", i, path.c_str());
            exit(1);
        }
        result.shape.push_back(static_cast<size_t>(dim));
        result.total_elements *= static_cast<size_t>(dim);
    }

    // Header size: 4 (magic) + 4 (ndim) + ndim * 8 (dims)
    size_t header_size = 4 + 4 + ndim * 8;
    size_t data_size = result.total_elements * sizeof(float);
    result.map_size = header_size + data_size;

    // Verify file size matches expected
    struct stat st;
    if (fstat(result.fd, &st) != 0) {
        fprintf(stderr, "fstat failed for %s\n", path.c_str());
        exit(1);
    }
    if (static_cast<size_t>(st.st_size) < result.map_size) {
        fprintf(stderr, "File %s is truncated: expected %zu bytes, got %lld\n",
                path.c_str(), result.map_size, (long long)st.st_size);
        exit(1);
    }

    // mmap the entire file read-only
    result.map_base = mmap(nullptr, result.map_size, PROT_READ, MAP_PRIVATE,
                           result.fd, 0);
    if (result.map_base == MAP_FAILED) {
        fprintf(stderr, "mmap failed for %s\n", path.c_str());
        exit(1);
    }

    // Advise the OS: we will access data sequentially (enables read-ahead)
    madvise(result.map_base, result.map_size, MADV_SEQUENTIAL);

    // Data pointer starts after header
    result.data = reinterpret_cast<float*>(
        static_cast<char*>(result.map_base) + header_size);

    return result;
}

#endif // MMAP_TENSOR_H
