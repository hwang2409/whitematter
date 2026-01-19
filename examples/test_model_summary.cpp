#include "tensor.h"
#include "layer.h"
#include <cstdio>
#include <cmath>

int main() {
    printf("Model Summary Test\n");
    printf("==================\n\n");

    // Test 1: Simple MLP
    printf("1. Simple MLP Summary\n");
    {
        Sequential model({
            new Linear(784, 256),
            new ReLU(),
            new Linear(256, 128),
            new ReLU(),
            new Linear(128, 10)
        });

        printf("\n--- Using model.summary() ---\n");
        model.summary({1, 784});

        printf("\n--- Using print_model_info() ---\n");
        print_model_info(&model, "MLP");

        // Verify parameter count
        ModelSummary info = get_model_summary(&model);
        // 784*256 + 256 + 256*128 + 128 + 128*10 + 10 = 200704 + 256 + 32768 + 128 + 1280 + 10 = 235,146
        size_t expected_params = 784*256 + 256 + 256*128 + 128 + 128*10 + 10;
        printf("\n   Expected params: %zu\n", expected_params);
        printf("   Actual params: %zu\n", info.total_params);
        printf("   %s\n\n", info.total_params == expected_params ? "PASSED" : "FAILED");
    }

    // Test 2: CNN for MNIST
    printf("2. CNN Summary (MNIST-style)\n");
    {
        Sequential model({
            new Conv2d(1, 16, 3, 1, 1),   // [1, 28, 28] -> [16, 28, 28]
            new BatchNorm2d(16),
            new ReLU(),
            new MaxPool2d(2, 2),          // [16, 28, 28] -> [16, 14, 14]
            new Conv2d(16, 32, 3, 1, 1),  // [16, 14, 14] -> [32, 14, 14]
            new BatchNorm2d(32),
            new ReLU(),
            new MaxPool2d(2, 2),          // [32, 14, 14] -> [32, 7, 7]
            new Flatten(),                // -> [1568]
            new Linear(1568, 128),
            new ReLU(),
            new Linear(128, 10)
        });

        model.summary({1, 1, 28, 28});

        ModelSummary info = get_model_summary(&model);
        printf("\n   Total params: %s\n", format_number(info.total_params).c_str());
        printf("   Memory (fp32): %s\n", format_memory(info.param_memory_bytes).c_str());
        printf("   Memory (fp16): %s\n", format_memory(info.param_memory_fp16_bytes).c_str());
        printf("   Layers: %zu\n", info.num_layers);
        printf("   %s\n\n", info.total_params > 0 ? "PASSED" : "FAILED");
    }

    // Test 3: Transformer-style model
    printf("3. Transformer Block Summary\n");
    {
        Sequential model({
            new Embedding(1000, 64),       // vocab=1000, embed=64
            new MultiHeadAttention(64, 4), // embed_dim=64, num_heads=4
            new LayerNorm(64),
            new Linear(64, 256),           // FFN
            new ReLU(),
            new Linear(256, 64),
            new LayerNorm(64),
            new Linear(64, 1000)           // Output projection
        });

        model.summary({1, 32});  // batch=1, seq_len=32

        ModelSummary info = get_model_summary(&model);
        printf("\n   Total params: %s\n", format_number(info.total_params).c_str());
        printf("   Memory (fp32): %s\n", format_memory(info.param_memory_bytes).c_str());
        printf("   Trainable: %s\n", format_number(info.trainable_params).c_str());
        printf("   %s\n\n", info.total_params > 0 ? "PASSED" : "FAILED");
    }

    // Test 4: Format utilities
    printf("4. Format Utilities\n");
    {
        bool pass = true;

        // Test format_number
        std::string n1 = format_number(1234567);
        pass &= (n1 == "1,234,567");
        printf("   format_number(1234567) = %s (expected: 1,234,567) %s\n",
               n1.c_str(), n1 == "1,234,567" ? "✓" : "✗");

        std::string n2 = format_number(999);
        pass &= (n2 == "999");
        printf("   format_number(999) = %s (expected: 999) %s\n",
               n2.c_str(), n2 == "999" ? "✓" : "✗");

        // Test format_memory
        std::string m1 = format_memory(1024);
        printf("   format_memory(1024) = %s\n", m1.c_str());

        std::string m2 = format_memory(1024 * 1024);
        printf("   format_memory(1MB) = %s\n", m2.c_str());

        std::string m3 = format_memory(1024 * 1024 * 1024);
        printf("   format_memory(1GB) = %s\n", m3.c_str());

        std::string m4 = format_memory(500);
        printf("   format_memory(500) = %s\n", m4.c_str());

        printf("   %s\n\n", pass ? "PASSED" : "FAILED");
    }

    // Test 5: count_parameters convenience functions
    printf("5. Convenience Functions\n");
    {
        Sequential model({
            new Linear(10, 20),  // 10*20 + 20 = 220
            new ReLU(),
            new Linear(20, 5)    // 20*5 + 5 = 105
        });

        size_t total = count_parameters(&model);
        size_t trainable = count_trainable_parameters(&model);
        size_t expected = 10*20 + 20 + 20*5 + 5;  // 325

        printf("   count_parameters: %zu (expected: %zu)\n", total, expected);
        printf("   count_trainable_parameters: %zu\n", trainable);
        printf("   %s\n\n", total == expected ? "PASSED" : "FAILED");
    }

    // Test 6: Large model memory estimation
    printf("6. Memory Estimation\n");
    {
        // Create a larger model to test memory formatting
        Sequential model({
            new Linear(1024, 2048),
            new ReLU(),
            new Linear(2048, 2048),
            new ReLU(),
            new Linear(2048, 1024)
        });

        ModelSummary info = get_model_summary(&model);

        printf("   Total params: %s\n", format_number(info.total_params).c_str());
        printf("   Parameter memory (fp32): %s\n", format_memory(info.param_memory_bytes).c_str());
        printf("   Parameter memory (fp16): %s\n", format_memory(info.param_memory_fp16_bytes).c_str());
        printf("   Gradient memory: %s\n", format_memory(info.grad_memory_bytes).c_str());
        printf("   Total training memory: %s\n", format_memory(info.total_memory_bytes).c_str());

        // Verify calculations
        size_t expected_params = 1024*2048 + 2048 + 2048*2048 + 2048 + 2048*1024 + 1024;
        bool params_correct = info.total_params == expected_params;
        bool memory_correct = info.param_memory_bytes == expected_params * 4;

        printf("   Params correct: %s\n", params_correct ? "yes" : "no");
        printf("   Memory correct: %s\n", memory_correct ? "yes" : "no");
        printf("   %s\n\n", (params_correct && memory_correct) ? "PASSED" : "FAILED");
    }

    // Test 7: Empty model
    printf("7. Edge Case - Empty Model\n");
    {
        Sequential model;  // Empty

        model.summary();

        ModelSummary info = get_model_summary(&model);
        printf("   Total params: %zu\n", info.total_params);
        printf("   Layers: %zu\n", info.num_layers);
        printf("   %s\n\n", (info.total_params == 0 && info.num_layers == 0) ? "PASSED" : "FAILED");
    }

    // Test 8: VGG-style deep network
    printf("8. Deep Network (VGG-style)\n");
    {
        Sequential model({
            // Block 1
            new Conv2d(3, 64, 3, 1, 1),
            new BatchNorm2d(64),
            new ReLU(),
            new Conv2d(64, 64, 3, 1, 1),
            new BatchNorm2d(64),
            new ReLU(),
            new MaxPool2d(2, 2),

            // Block 2
            new Conv2d(64, 128, 3, 1, 1),
            new BatchNorm2d(128),
            new ReLU(),
            new Conv2d(128, 128, 3, 1, 1),
            new BatchNorm2d(128),
            new ReLU(),
            new MaxPool2d(2, 2),

            // Classifier
            new Flatten(),
            new Linear(128 * 8 * 8, 512),
            new ReLU(),
            new Dropout(0.5),
            new Linear(512, 10)
        });

        printf("   VGG-style model for 32x32 images:\n");
        model.summary({1, 3, 32, 32});

        ModelSummary info = get_model_summary(&model);
        printf("\n   Summary:\n");
        printf("   - Layers: %zu\n", info.num_layers);
        printf("   - Parameters: %s\n", format_number(info.total_params).c_str());
        printf("   - Memory (fp32): %s\n", format_memory(info.param_memory_bytes).c_str());
        printf("   - Memory (fp16): %s (50%% savings)\n", format_memory(info.param_memory_fp16_bytes).c_str());
        printf("   %s\n\n", info.total_params > 0 ? "PASSED" : "FAILED");
    }

    printf("All tests completed!\n");
    return 0;
}
