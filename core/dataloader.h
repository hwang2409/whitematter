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

struct Batch {
    TensorPtr data;
    TensorPtr labels;
};

struct Dataset {
    TensorPtr data;
    TensorPtr labels;
    size_t num_samples;

    std::vector<size_t> sample_shape() const {
        if (data->shape.size() < 2) return {};
        return std::vector<size_t>(data->shape.begin() + 1, data->shape.end());
    }
};

class BatchQueue {
public:
    explicit BatchQueue(size_t max_size) : max_size_(max_size), done_(false) {}

    bool push(Batch batch) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this] { return queue_.size() < max_size_ || done_; });
        if (done_) return false;
        queue_.push(std::move(batch));
        not_empty_.notify_one();
        return true;
    }

    bool pop(Batch& batch) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this] { return !queue_.empty() || done_; });
        if (queue_.empty()) return false;
        batch = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return true;
    }

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

class ThreadedDataLoader {
public:
    ThreadedDataLoader(const Dataset& dataset, size_t batch_size,
                       bool shuffle = true, size_t num_workers = 2,
                       size_t prefetch_factor = 2);

    ~ThreadedDataLoader();

    void reset();
    bool has_next();
    Batch next_batch();

    std::pair<TensorPtr, TensorPtr> next_batch_pair() {
        auto batch = next_batch();
        return {batch.data, batch.labels};
    }

    size_t num_batches() const {
        return (num_samples_ + batch_size_ - 1) / batch_size_;
    }

    void stop_workers();

private:
    void worker_loop();
    Batch create_batch(size_t start_idx, size_t end_idx);

    TensorPtr data_;
    TensorPtr labels_;
    size_t num_samples_;
    std::vector<size_t> sample_shape_;
    size_t sample_size_;

    size_t batch_size_;
    bool shuffle_;
    size_t num_workers_;

    std::vector<size_t> indices_;
    std::atomic<size_t> next_batch_idx_;
    size_t total_batches_;
    size_t batches_consumed_;

    BatchQueue queue_;
    std::vector<std::thread> workers_;
    std::atomic<bool> running_;
    std::atomic<size_t> batches_pushed_;
};

#endif
