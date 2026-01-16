#include "dataloader.h"
#include <algorithm>
#include <random>

ThreadedDataLoader::ThreadedDataLoader(const Dataset& dataset, size_t batch_size,
                                       bool shuffle, size_t num_workers,
                                       size_t prefetch_factor)
    : data_(dataset.data),
      labels_(dataset.labels),
      num_samples_(dataset.num_samples),
      sample_shape_(dataset.sample_shape()),
      batch_size_(batch_size),
      shuffle_(shuffle),
      num_workers_(num_workers),
      next_batch_idx_(0),
      total_batches_(0),
      batches_consumed_(0),
      queue_(std::max(num_workers * prefetch_factor, size_t(1)) + 1),
      running_(false),
      batches_pushed_(0) {

    // Calculate sample size (elements per sample)
    sample_size_ = 1;
    for (auto d : sample_shape_) {
        sample_size_ *= d;
    }

    // Initialize indices
    indices_.resize(num_samples_);
    for (size_t i = 0; i < num_samples_; i++) {
        indices_[i] = i;
    }

    total_batches_ = num_batches();
}

ThreadedDataLoader::~ThreadedDataLoader() {
    stop_workers();
}

void ThreadedDataLoader::reset() {
    // Stop any existing workers
    stop_workers();

    // Reset state
    batches_consumed_ = 0;
    next_batch_idx_ = 0;
    batches_pushed_ = 0;
    queue_.reset();

    // Shuffle indices if enabled
    if (shuffle_) {
        static thread_local std::mt19937 rng(std::random_device{}());
        std::shuffle(indices_.begin(), indices_.end(), rng);
    }

    // Start workers if using multi-threading
    if (num_workers_ > 0) {
        running_ = true;
        workers_.clear();
        workers_.reserve(num_workers_);
        for (size_t i = 0; i < num_workers_; i++) {
            workers_.emplace_back(&ThreadedDataLoader::worker_loop, this);
        }
    }
}

void ThreadedDataLoader::stop_workers() {
    running_ = false;
    queue_.set_done();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
}

bool ThreadedDataLoader::has_next() {
    return batches_consumed_ < total_batches_;
}

Batch ThreadedDataLoader::next_batch() {
    if (num_workers_ == 0) {
        // Synchronous mode: create batch directly
        size_t start = batches_consumed_ * batch_size_;
        size_t end = std::min(start + batch_size_, num_samples_);
        batches_consumed_++;
        return create_batch(start, end);
    } else {
        // Async mode: get from queue
        Batch batch;
        if (queue_.pop(batch)) {
            batches_consumed_++;
            return batch;
        }
        // Queue is done and empty - shouldn't happen if has_next() was checked
        return Batch{nullptr, nullptr};
    }
}

void ThreadedDataLoader::worker_loop() {
    while (running_.load()) {
        // Atomically get next batch index to process
        size_t batch_idx = next_batch_idx_.fetch_add(1);

        if (batch_idx >= total_batches_) {
            // No more batches to claim - just exit, don't call set_done() yet
            break;
        }

        // Calculate sample range for this batch
        size_t start = batch_idx * batch_size_;
        size_t end = std::min(start + batch_size_, num_samples_);

        // Create the batch
        Batch batch = create_batch(start, end);

        // Try to push - if not running anymore, exit without pushing
        if (!running_.load()) break;

        if (queue_.push(std::move(batch))) {
            // Successfully pushed - increment counter
            size_t pushed = batches_pushed_.fetch_add(1) + 1;

            // If this was the last batch, signal queue done
            if (pushed == total_batches_) {
                queue_.set_done();
            }
        }
    }
}

Batch ThreadedDataLoader::create_batch(size_t start_idx, size_t end_idx) {
    size_t actual_batch_size = end_idx - start_idx;

    // Create output shape: [batch_size, ...sample_shape]
    std::vector<size_t> batch_shape = {actual_batch_size};
    batch_shape.insert(batch_shape.end(), sample_shape_.begin(), sample_shape_.end());

    auto batch_data = Tensor::create(batch_shape, false);
    auto batch_labels = Tensor::create({actual_batch_size}, false);

    // Copy data for each sample in the batch
    for (size_t i = 0; i < actual_batch_size; i++) {
        size_t idx = indices_[start_idx + i];

        // Copy sample data
        std::copy(
            data_->data.begin() + idx * sample_size_,
            data_->data.begin() + (idx + 1) * sample_size_,
            batch_data->data.begin() + i * sample_size_
        );

        // Copy label
        batch_labels->data[i] = labels_->data[idx];
    }

    return Batch{batch_data, batch_labels};
}
