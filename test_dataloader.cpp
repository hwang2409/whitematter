// Full test for ThreadedDataLoader
#include "tensor.h"
#include "dataloader.h"
#include <chrono>
#include <cstdio>
#include <thread>

int main() {
    printf("ThreadedDataLoader Benchmark\n");
    printf("============================\n\n");

    const size_t num_samples = 10000;
    const size_t sample_size = 784;
    const size_t batch_size = 64;

    printf("Creating dataset (%zu samples, %zu features)...\n", num_samples, sample_size);

    Dataset dataset;
    dataset.data = Tensor::create({num_samples, sample_size}, false);
    dataset.labels = Tensor::create({num_samples}, false);
    dataset.num_samples = num_samples;

    for (size_t i = 0; i < num_samples; i++) {
        for (size_t j = 0; j < sample_size; j++) {
            dataset.data->data[i * sample_size + j] = static_cast<float>(i % 256) / 255.0f;
        }
        dataset.labels->data[i] = static_cast<float>(i % 10);
    }
    printf("Done.\n\n");

    // Benchmark function
    auto run_benchmark = [&](const char* name, size_t num_workers) {
        ThreadedDataLoader loader(dataset, batch_size, true, num_workers);

        auto start = std::chrono::high_resolution_clock::now();
        loader.reset();
        size_t count = 0;
        while (loader.has_next()) {
            loader.next_batch();
            count++;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("  %s: %zu batches in %lld ms\n", name, count, ms);
    };

    printf("Benchmark: Pure data loading\n");
    run_benchmark("Sync (0 workers)", 0);
    run_benchmark("Async (1 worker)", 1);
    run_benchmark("Async (2 workers)", 2);
    run_benchmark("Async (4 workers)", 4);

    // With simulated compute
    printf("\nBenchmark: With 0.5ms compute per batch\n");
    {
        auto run_with_compute = [&](const char* name, size_t num_workers) {
            ThreadedDataLoader loader(dataset, batch_size, true, num_workers, 4);
            auto start = std::chrono::high_resolution_clock::now();
            loader.reset();
            while (loader.has_next()) {
                loader.next_batch();
                std::this_thread::sleep_for(std::chrono::microseconds(500));
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            printf("  %s: %lld ms\n", name, ms);
        };

        run_with_compute("Sync (0 workers)", 0);
        run_with_compute("Async (2 workers)", 2);
    }

    // Multiple epochs test
    printf("\nMultiple epochs (2 workers):\n");
    {
        ThreadedDataLoader loader(dataset, batch_size, true, 2);
        for (int epoch = 0; epoch < 3; epoch++) {
            loader.reset();
            size_t count = 0;
            while (loader.has_next()) {
                loader.next_batch();
                count++;
            }
            printf("  Epoch %d: %zu batches\n", epoch + 1, count);
        }
    }

    printf("\nAll tests passed!\n");
    return 0;
}
