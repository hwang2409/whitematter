#include "cifar10.h"
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <random>

// CIFAR-10 normalization constants (per-channel mean and std)
static const float CIFAR10_MEAN[3] = {0.4914f, 0.4822f, 0.4465f};
static const float CIFAR10_STD[3] = {0.2470f, 0.2435f, 0.2616f};

// CIFAR-10 class names
static const char* CIFAR10_CLASSES[10] = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

const char* cifar10_class_name(int label) {
    if (label < 0 || label >= 10) return "unknown";
    return CIFAR10_CLASSES[label];
}

CIFAR10Dataset load_cifar10_batch(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open CIFAR-10 file: " + filepath);
    }

    // CIFAR-10 binary format:
    // Each image: 1 byte label + 3072 bytes (32*32*3) RGB data
    // Data is stored as: R channel (1024), G channel (1024), B channel (1024)
    // Each batch file contains 10000 images

    // Get file size to determine number of samples
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    const size_t bytes_per_image = 1 + 3 * 32 * 32;  // 1 label + 3072 pixels
    size_t num_samples = file_size / bytes_per_image;

    CIFAR10Dataset dataset;
    dataset.num_samples = num_samples;
    dataset.images = Tensor::create({num_samples, 3, 32, 32}, false);
    dataset.labels = Tensor::create({num_samples}, false);

    std::vector<unsigned char> buffer(bytes_per_image);

    for (size_t i = 0; i < num_samples; i++) {
        file.read(reinterpret_cast<char*>(buffer.data()), bytes_per_image);
        if (!file) {
            throw std::runtime_error("Error reading CIFAR-10 file at sample " + std::to_string(i));
        }

        // First byte is the label
        dataset.labels->data[i] = static_cast<float>(buffer[0]);

        // Next 3072 bytes are RGB data (channel-first: R, G, B)
        size_t img_offset = i * 3 * 32 * 32;
        for (size_t c = 0; c < 3; c++) {
            for (size_t h = 0; h < 32; h++) {
                for (size_t w = 0; w < 32; w++) {
                    size_t src_idx = 1 + c * 32 * 32 + h * 32 + w;  // +1 to skip label
                    size_t dst_idx = img_offset + c * 32 * 32 + h * 32 + w;

                    // Normalize: (pixel/255 - mean) / std
                    float pixel = static_cast<float>(buffer[src_idx]) / 255.0f;
                    dataset.images->data[dst_idx] = (pixel - CIFAR10_MEAN[c]) / CIFAR10_STD[c];
                }
            }
        }
    }

    return dataset;
}

CIFAR10Dataset load_cifar10_train(const std::string& data_dir) {
    // Load all 5 training batch files and concatenate
    std::vector<CIFAR10Dataset> batches;
    size_t total_samples = 0;

    for (int i = 1; i <= 5; i++) {
        std::string filepath = data_dir + "/data_batch_" + std::to_string(i) + ".bin";
        auto batch = load_cifar10_batch(filepath);
        total_samples += batch.num_samples;
        batches.push_back(std::move(batch));
    }

    // Concatenate all batches
    CIFAR10Dataset dataset;
    dataset.num_samples = total_samples;
    dataset.images = Tensor::create({total_samples, 3, 32, 32}, false);
    dataset.labels = Tensor::create({total_samples}, false);

    size_t offset = 0;
    for (const auto& batch : batches) {
        // Copy images
        std::copy(batch.images->data.begin(), batch.images->data.end(),
                  dataset.images->data.begin() + offset * 3 * 32 * 32);

        // Copy labels
        std::copy(batch.labels->data.begin(), batch.labels->data.end(),
                  dataset.labels->data.begin() + offset);

        offset += batch.num_samples;
    }

    printf("Loaded CIFAR-10 training set: %zu samples\n", dataset.num_samples);
    return dataset;
}

CIFAR10Dataset load_cifar10_test(const std::string& data_dir) {
    std::string filepath = data_dir + "/test_batch.bin";
    auto dataset = load_cifar10_batch(filepath);
    printf("Loaded CIFAR-10 test set: %zu samples\n", dataset.num_samples);
    return dataset;
}

// CIFAR10DataLoader implementation

CIFAR10DataLoader::CIFAR10DataLoader(const CIFAR10Dataset& dataset, size_t batch_size,
                                     bool shuffle, bool augment)
    : images(dataset.images), labels(dataset.labels),
      batch_size(batch_size), num_samples(dataset.num_samples),
      current_idx(0), shuffle(shuffle), augment(augment) {

    indices.resize(num_samples);
    for (size_t i = 0; i < num_samples; i++) {
        indices[i] = i;
    }
    if (shuffle) {
        reset();
    }
}

void CIFAR10DataLoader::reset() {
    current_idx = 0;
    if (shuffle) {
        static thread_local std::mt19937 gen(std::random_device{}());
        std::shuffle(indices.begin(), indices.end(), gen);
    }
}

bool CIFAR10DataLoader::has_next() const {
    return current_idx < num_samples;
}

size_t CIFAR10DataLoader::num_batches() const {
    return (num_samples + batch_size - 1) / batch_size;
}

std::pair<TensorPtr, TensorPtr> CIFAR10DataLoader::next_batch() {
    size_t actual_batch_size = std::min(batch_size, num_samples - current_idx);

    // Create batch tensors
    auto batch_images = Tensor::create({actual_batch_size, 3, 32, 32}, false);
    auto batch_labels = Tensor::create({actual_batch_size}, false);

    const size_t img_size = 3 * 32 * 32;

    for (size_t i = 0; i < actual_batch_size; i++) {
        size_t idx = indices[current_idx + i];

        // Copy image
        std::copy(images->data.begin() + idx * img_size,
                  images->data.begin() + (idx + 1) * img_size,
                  batch_images->data.begin() + i * img_size);

        // Copy label
        batch_labels->data[i] = labels->data[idx];
    }

    current_idx += actual_batch_size;

    // Apply data augmentation during training
    if (augment) {
        // Random crop with padding: 32 -> pad(4) -> 40 -> random_crop(32)
        batch_images = batch_images->pad2d(4)->random_crop(32, 32);
        // Random horizontal flip with p=0.5
        batch_images = batch_images->random_flip_horizontal(0.5f);
    }

    return {batch_images, batch_labels};
}

// ============================================================================
// AsyncCIFAR10DataLoader implementation - multi-threaded prefetching
// ============================================================================

AsyncCIFAR10DataLoader::AsyncCIFAR10DataLoader(const CIFAR10Dataset& dataset,
                                               size_t batch_size,
                                               bool shuffle, bool augment,
                                               size_t num_workers,
                                               size_t prefetch_count)
    : images(dataset.images), labels(dataset.labels),
      batch_size(batch_size), num_samples(dataset.num_samples),
      shuffle(shuffle), augment(augment),
      next_batch_idx(0),
      prefetch_count(prefetch_count),
      stop_flag(false), epoch_done(false), fetched_count(0) {

    total_batches = (num_samples + batch_size - 1) / batch_size;

    // Initialize indices
    indices.resize(num_samples);
    for (size_t i = 0; i < num_samples; i++) {
        indices[i] = i;
    }

    // Shuffle if requested
    if (shuffle) {
        static thread_local std::mt19937 gen(std::random_device{}());
        std::shuffle(indices.begin(), indices.end(), gen);
    }

    // Start worker threads
    for (size_t i = 0; i < num_workers; i++) {
        workers.emplace_back(&AsyncCIFAR10DataLoader::worker_loop, this);
    }
}

AsyncCIFAR10DataLoader::~AsyncCIFAR10DataLoader() {
    // Signal workers to stop
    stop_flag = true;
    queue_not_full.notify_all();

    // Join all worker threads
    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void AsyncCIFAR10DataLoader::reset() {
    // Wait for workers to drain
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        // Clear the queue
        while (!batch_queue.empty()) {
            batch_queue.pop();
        }
    }

    // Reset state
    next_batch_idx = 0;
    fetched_count = 0;
    epoch_done = false;

    // Reshuffle if needed
    if (shuffle) {
        static thread_local std::mt19937 gen(std::random_device{}());
        std::shuffle(indices.begin(), indices.end(), gen);
    }

    // Wake up workers
    queue_not_full.notify_all();
}

bool AsyncCIFAR10DataLoader::has_next() {
    std::lock_guard<std::mutex> lock(fetch_mutex);
    return fetched_count < total_batches;
}

size_t AsyncCIFAR10DataLoader::num_batches() const {
    return total_batches;
}

std::pair<TensorPtr, TensorPtr> AsyncCIFAR10DataLoader::next_batch() {
    Batch batch;

    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // Wait for a batch to be available
        queue_not_empty.wait(lock, [this] {
            return !batch_queue.empty() || stop_flag;
        });

        if (stop_flag && batch_queue.empty()) {
            // Return empty tensors if stopped
            return {Tensor::create({0}, false), Tensor::create({0}, false)};
        }

        batch = std::move(batch_queue.front());
        batch_queue.pop();
    }

    // Notify workers that there's space in the queue
    queue_not_full.notify_one();

    // Update fetched count
    {
        std::lock_guard<std::mutex> lock(fetch_mutex);
        fetched_count++;
    }

    return {batch.images, batch.labels};
}

AsyncCIFAR10DataLoader::Batch AsyncCIFAR10DataLoader::prepare_batch(size_t batch_idx) {
    size_t start_idx = batch_idx * batch_size;
    size_t actual_batch_size = std::min(batch_size, num_samples - start_idx);

    auto batch_images = Tensor::create({actual_batch_size, 3, 32, 32}, false);
    auto batch_labels = Tensor::create({actual_batch_size}, false);

    const size_t img_size = 3 * 32 * 32;

    for (size_t i = 0; i < actual_batch_size; i++) {
        size_t idx = indices[start_idx + i];

        // Copy image
        std::copy(images->data.begin() + idx * img_size,
                  images->data.begin() + (idx + 1) * img_size,
                  batch_images->data.begin() + i * img_size);

        // Copy label
        batch_labels->data[i] = labels->data[idx];
    }

    // Apply data augmentation
    if (augment) {
        batch_images = batch_images->pad2d(4)->random_crop(32, 32);
        batch_images = batch_images->random_flip_horizontal(0.5f);
    }

    return {batch_images, batch_labels, batch_idx};
}

void AsyncCIFAR10DataLoader::worker_loop() {
    while (!stop_flag) {
        // Get the next batch index to process
        size_t batch_idx = next_batch_idx.fetch_add(1);

        if (batch_idx >= total_batches) {
            // No more batches to process in this epoch
            // Wait for reset or stop
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_not_full.wait(lock, [this] {
                return stop_flag || next_batch_idx < total_batches;
            });
            continue;
        }

        // Prepare the batch
        Batch batch = prepare_batch(batch_idx);

        // Add to queue
        {
            std::unique_lock<std::mutex> lock(queue_mutex);

            // Wait if queue is full
            queue_not_full.wait(lock, [this] {
                return batch_queue.size() < prefetch_count || stop_flag;
            });

            if (stop_flag) break;

            batch_queue.push(std::move(batch));
        }

        // Notify consumer
        queue_not_empty.notify_one();
    }
}
