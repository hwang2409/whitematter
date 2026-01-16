#ifndef DATALOADER_H
#define DATALOADER_H

#include "tensor.h"
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>

// A batch of data (images + labels)
struct Batch {
    TensorPtr data;
    TensorPtr labels;
};

// Generic dataset interface
struct Dataset {
    TensorPtr data;
    TensorPtr labels;
    size_t num_samples;

    // Get shape of a single sample (excluding batch dimension)
    std::vector<size_t> sample_shape() const {
        if (data->shape.size() < 2) return {};
        return std::vector<size_t>(data->shape.begin() + 1, data->shape.end());
    }
};

// Thread-safe batch queue
class BatchQueue {
public:
    explicit BatchQueue(size_t max_size) : max_size_(max_size), done_(false) {}

    // Producer: add a batch to the queue (blocks if full)
    // Returns false if queue is done
    bool push(Batch batch) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this] { return queue_.size() < max_size_ || done_; });
        if (done_) return false;
        queue_.push(std::move(batch));
        not_empty_.notify_one();
        return true;
    }

    // Consumer: get a batch from the queue (blocks if empty)
    // Returns false if queue is done and empty
    bool pop(Batch& batch) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this] { return !queue_.empty() || done_; });
        if (queue_.empty()) return false;
        batch = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return true;
    }

    // Signal that no more batches will be added
    void set_done() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            done_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!queue_.empty()) queue_.pop();
        done_ = false;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    bool is_done() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return done_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::queue<Batch> queue_;
    size_t max_size_;
    bool done_;
};

// Multi-threaded data loader with prefetching
class ThreadedDataLoader {
public:
    // num_workers: number of prefetch threads (0 = synchronous loading)
    // prefetch_factor: batches to prefetch per worker
    ThreadedDataLoader(const Dataset& dataset, size_t batch_size,
                       bool shuffle = true, size_t num_workers = 2,
                       size_t prefetch_factor = 2);

    ~ThreadedDataLoader();

    // Start a new epoch (shuffles if enabled, starts workers)
    void reset();

    // Check if more batches available
    bool has_next();

    // Get next batch (blocks until available)
    Batch next_batch();

    // Convenience method returning pair for compatibility
    std::pair<TensorPtr, TensorPtr> next_batch_pair() {
        auto batch = next_batch();
        return {batch.data, batch.labels};
    }

    size_t num_batches() const {
        return (num_samples_ + batch_size_ - 1) / batch_size_;
    }

    // Stop workers (called automatically on destruction/reset)
    void stop_workers();

private:
    // Worker thread function
    void worker_loop();

    // Create a batch from indices [start, end)
    Batch create_batch(size_t start_idx, size_t end_idx);

    // Dataset reference
    TensorPtr data_;
    TensorPtr labels_;
    size_t num_samples_;
    std::vector<size_t> sample_shape_;
    size_t sample_size_;

    // Configuration
    size_t batch_size_;
    bool shuffle_;
    size_t num_workers_;

    // State
    std::vector<size_t> indices_;
    std::atomic<size_t> next_batch_idx_;
    size_t total_batches_;
    size_t batches_consumed_;

    // Threading
    BatchQueue queue_;
    std::vector<std::thread> workers_;
    std::atomic<bool> running_;
    std::atomic<size_t> batches_pushed_;  // Track successfully pushed batches
};

#endif
