#ifndef CIFAR10_H
#define CIFAR10_H

#include "tensor.h"
#include <string>
#include <utility>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

struct CIFAR10Dataset {
    TensorPtr images;   // Shape: [N, 3, 32, 32] normalized
    TensorPtr labels;   // Shape: [N]
    size_t num_samples;
};

// Load CIFAR-10 dataset
// data_dir should contain: data_batch_1.bin ... data_batch_5.bin, test_batch.bin
CIFAR10Dataset load_cifar10_train(const std::string& data_dir);
CIFAR10Dataset load_cifar10_test(const std::string& data_dir);

// Load a single CIFAR-10 binary file
// Returns images in [N, 3, 32, 32] format, normalized with ImageNet-style mean/std
CIFAR10Dataset load_cifar10_batch(const std::string& filepath);

// CIFAR-10 class names
const char* cifar10_class_name(int label);

// DataLoader that works with CIFAR10Dataset
class CIFAR10DataLoader {
public:
    TensorPtr images;
    TensorPtr labels;
    size_t batch_size;
    size_t num_samples;
    size_t current_idx;
    bool shuffle;
    bool augment;  // Apply data augmentation (random crop + flip)
    std::vector<size_t> indices;

    CIFAR10DataLoader(const CIFAR10Dataset& dataset, size_t batch_size,
                      bool shuffle = true, bool augment = true);

    void reset();
    bool has_next() const;
    std::pair<TensorPtr, TensorPtr> next_batch();
    size_t num_batches() const;
};

// Async DataLoader with background prefetching for better performance
class AsyncCIFAR10DataLoader {
public:
    AsyncCIFAR10DataLoader(const CIFAR10Dataset& dataset, size_t batch_size,
                           bool shuffle = true, bool augment = true,
                           size_t num_workers = 2, size_t prefetch_count = 4);
    ~AsyncCIFAR10DataLoader();

    void reset();
    bool has_next();
    std::pair<TensorPtr, TensorPtr> next_batch();
    size_t num_batches() const;

private:
    // Dataset references
    TensorPtr images;
    TensorPtr labels;
    size_t batch_size;
    size_t num_samples;
    bool shuffle;
    bool augment;

    // Index management
    std::vector<size_t> indices;
    std::atomic<size_t> next_batch_idx;
    size_t total_batches;

    // Threading
    size_t prefetch_count;
    std::vector<std::thread> workers;
    std::atomic<bool> stop_flag;
    std::atomic<bool> epoch_done;

    // Thread-safe batch queue
    struct Batch {
        TensorPtr images;
        TensorPtr labels;
        size_t batch_idx;
    };
    std::queue<Batch> batch_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_not_empty;
    std::condition_variable queue_not_full;

    // Fetched batch tracking
    size_t fetched_count;
    std::mutex fetch_mutex;

    // Worker function
    void worker_loop();
    Batch prepare_batch(size_t batch_idx);
};

#endif
