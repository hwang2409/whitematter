#include "test_framework.h"
#include "../core/tensor.h"
#include "../core/layer.h"
#include <cmath>

// =============================================================================
// Linear Layer Tests
// =============================================================================

void test_linear_forward() {
    Linear layer(10, 5);
    auto input = Tensor::randn({4, 10});
    auto output = layer.forward(input);

    TEST_ASSERT_SHAPE(output, std::vector<size_t>({4, 5}));
}

void test_linear_parameters() {
    Linear layer(10, 5);
    auto params = layer.parameters();

    TEST_ASSERT_EQ(params.size(), 2u);  // weight and bias
    TEST_ASSERT_SHAPE(params[0], std::vector<size_t>({10, 5}));  // weight
    TEST_ASSERT_SHAPE(params[1], std::vector<size_t>({5}));      // bias
}

void test_linear_num_parameters() {
    Linear layer(10, 5);
    size_t expected = 10 * 5 + 5;  // weight + bias
    TEST_ASSERT_EQ(layer.num_parameters(), expected);
}

void test_linear_gradient() {
    Linear layer(3, 2);
    auto input = Tensor::randn({2, 3}, true);
    auto output = layer.forward(input);
    auto loss = output->sum();

    loss->backward();

    // Weight and bias should have gradients
    TEST_ASSERT_EQ(layer.weight->grad.size(), 6u);
    TEST_ASSERT_EQ(layer.bias->grad.size(), 2u);
}

// =============================================================================
// Activation Layer Tests
// =============================================================================

void test_relu_forward() {
    ReLU relu;
    auto input = Tensor::create({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, std::vector<size_t>{5});
    auto output = relu.forward(input);

    TEST_ASSERT_NEAR(output->data[0], 0.0f, 1e-5f);
    TEST_ASSERT_NEAR(output->data[1], 0.0f, 1e-5f);
    TEST_ASSERT_NEAR(output->data[2], 0.0f, 1e-5f);
    TEST_ASSERT_NEAR(output->data[3], 1.0f, 1e-5f);
    TEST_ASSERT_NEAR(output->data[4], 2.0f, 1e-5f);
}

void test_sigmoid_forward() {
    Sigmoid sigmoid;
    auto input = Tensor::create({0.0f}, std::vector<size_t>{1});
    auto output = sigmoid.forward(input);

    TEST_ASSERT_NEAR(output->data[0], 0.5f, 1e-5f);
}

void test_tanh_forward() {
    Tanh tanh_layer;
    auto input = Tensor::create({0.0f}, std::vector<size_t>{1});
    auto output = tanh_layer.forward(input);

    TEST_ASSERT_NEAR(output->data[0], 0.0f, 1e-5f);
}

void test_softmax_forward() {
    Softmax softmax;
    auto input = Tensor::create({1.0f, 2.0f, 3.0f}, {1, 3});
    auto output = softmax.forward(input);

    // Softmax should sum to 1
    float sum = output->data[0] + output->data[1] + output->data[2];
    TEST_ASSERT_NEAR(sum, 1.0f, 1e-5f);

    // Values should be in increasing order
    TEST_ASSERT(output->data[0] < output->data[1]);
    TEST_ASSERT(output->data[1] < output->data[2]);
}

void test_logsoftmax_forward() {
    LogSoftmax logsoftmax;
    auto input = Tensor::create({1.0f, 2.0f, 3.0f}, {1, 3});
    auto output = logsoftmax.forward(input);

    // exp(log_softmax) should sum to 1
    float sum = std::exp(output->data[0]) + std::exp(output->data[1]) + std::exp(output->data[2]);
    TEST_ASSERT_NEAR(sum, 1.0f, 1e-5f);
}

// =============================================================================
// Dropout Tests
// =============================================================================

void test_dropout_training() {
    Dropout dropout(0.5f);
    dropout.train();

    auto input = Tensor::ones({100, 100});
    auto output = dropout.forward(input);

    // Some values should be zeroed out
    int zeros = 0;
    for (size_t i = 0; i < output->size(); i++) {
        if (output->data[i] == 0.0f) zeros++;
    }

    // Approximately half should be zeros (with some tolerance)
    float drop_rate = static_cast<float>(zeros) / output->size();
    TEST_ASSERT(drop_rate > 0.3f && drop_rate < 0.7f);
}

void test_dropout_eval() {
    Dropout dropout(0.5f);
    dropout.eval();

    auto input = Tensor::ones({10, 10});
    auto output = dropout.forward(input);

    // In eval mode, all values should pass through unchanged
    for (size_t i = 0; i < output->size(); i++) {
        TEST_ASSERT_NEAR(output->data[i], 1.0f, 1e-5f);
    }
}

// =============================================================================
// Conv2d Tests
// =============================================================================

void test_conv2d_forward() {
    Conv2d conv(3, 16, 3, 1, 1);  // 3->16 channels, 3x3 kernel, stride 1, padding 1
    auto input = Tensor::randn({2, 3, 28, 28});  // [N, C, H, W]
    auto output = conv.forward(input);

    // With padding=1, kernel=3, stride=1: output size = input size
    TEST_ASSERT_SHAPE(output, std::vector<size_t>({2, 16, 28, 28}));
}

void test_conv2d_parameters() {
    Conv2d conv(3, 16, 3);
    auto params = conv.parameters();

    TEST_ASSERT_EQ(params.size(), 2u);  // weight and bias
    TEST_ASSERT_SHAPE(params[0], std::vector<size_t>({16, 3, 3, 3}));  // [out, in, kH, kW]
    TEST_ASSERT_SHAPE(params[1], std::vector<size_t>({16}));           // bias
}

void test_conv2d_no_padding() {
    Conv2d conv(1, 1, 3, 1, 0);  // No padding
    auto input = Tensor::randn({1, 1, 5, 5});
    auto output = conv.forward(input);

    // Without padding: 5 - 3 + 1 = 3
    TEST_ASSERT_SHAPE(output, std::vector<size_t>({1, 1, 3, 3}));
}

void test_conv2d_stride() {
    Conv2d conv(1, 1, 3, 2, 1);  // Stride 2
    auto input = Tensor::randn({1, 1, 8, 8});
    auto output = conv.forward(input);

    // With padding=1, kernel=3, stride=2: output = (8 + 2*1 - 3) / 2 + 1 = 4
    TEST_ASSERT_SHAPE(output, std::vector<size_t>({1, 1, 4, 4}));
}

// =============================================================================
// ConvTranspose2d Tests
// =============================================================================

void test_conv_transpose2d_forward() {
    ConvTranspose2d conv(16, 3, 3, 1, 1);  // 16->3 channels, 3x3 kernel, stride 1, padding 1
    auto input = Tensor::randn({2, 16, 28, 28});  // [N, C, H, W]
    auto output = conv.forward(input);

    // With stride=1, padding=1, kernel=3: output_size = (28 - 1) * 1 - 2*1 + 3 = 28
    TEST_ASSERT_SHAPE(output, std::vector<size_t>({2, 3, 28, 28}));
}

void test_conv_transpose2d_upsample() {
    // ConvTranspose2d with stride=2 doubles spatial dimensions (upsampling)
    ConvTranspose2d conv(16, 8, 4, 2, 1);  // stride=2, kernel=4, padding=1
    auto input = Tensor::randn({1, 16, 7, 7});
    auto output = conv.forward(input);

    // output_size = (7 - 1) * 2 - 2*1 + 4 = 14
    TEST_ASSERT_SHAPE(output, std::vector<size_t>({1, 8, 14, 14}));
}

void test_conv_transpose2d_parameters() {
    ConvTranspose2d conv(16, 8, 3);
    auto params = conv.parameters();

    TEST_ASSERT_EQ(params.size(), 2u);  // weight and bias
    // Weight shape for ConvTranspose2d: [in_channels, out_channels, kH, kW]
    TEST_ASSERT_SHAPE(params[0], std::vector<size_t>({16, 8, 3, 3}));
    TEST_ASSERT_SHAPE(params[1], std::vector<size_t>({8}));  // bias
}

void test_conv_transpose2d_gradient() {
    ConvTranspose2d conv(4, 2, 3, 1, 1);
    auto input = Tensor::randn({1, 4, 4, 4}, true);
    auto output = conv.forward(input);
    auto loss = output->sum();

    loss->backward();

    // Input, weight and bias should have gradients
    TEST_ASSERT(input->grad.size() > 0);
    TEST_ASSERT(conv.weight->grad.size() > 0);
    TEST_ASSERT(conv.bias->grad.size() > 0);
}

void test_conv_transpose2d_output_padding() {
    // output_padding is used to resolve ambiguity when stride > 1
    ConvTranspose2d conv(8, 4, 3, 2, 1, 1);  // output_padding = 1
    auto input = Tensor::randn({1, 8, 4, 4});
    auto output = conv.forward(input);

    // output_size = (4 - 1) * 2 - 2*1 + 3 + 1 = 8
    TEST_ASSERT_SHAPE(output, std::vector<size_t>({1, 4, 8, 8}));
}

void test_conv_transpose2d_inverse_conv2d() {
    // ConvTranspose2d can "undo" the spatial reduction of Conv2d
    // First downsample with Conv2d
    Conv2d down(1, 8, 4, 2, 1);  // 8x8 -> 4x4
    ConvTranspose2d up(8, 1, 4, 2, 1);  // 4x4 -> 8x8

    auto input = Tensor::randn({1, 1, 8, 8});
    auto downsized = down.forward(input);
    auto upsized = up.forward(downsized);

    TEST_ASSERT_SHAPE(downsized, std::vector<size_t>({1, 8, 4, 4}));
    TEST_ASSERT_SHAPE(upsized, std::vector<size_t>({1, 1, 8, 8}));
}

// =============================================================================
// Pooling Tests
// =============================================================================

void test_maxpool2d_forward() {
    MaxPool2d pool(2, 2);
    auto input = Tensor::randn({2, 3, 8, 8});
    auto output = pool.forward(input);

    // 8 / 2 = 4
    TEST_ASSERT_SHAPE(output, std::vector<size_t>({2, 3, 4, 4}));
}

void test_avgpool2d_forward() {
    AvgPool2d pool(2, 2);
    auto input = Tensor::randn({2, 3, 8, 8});
    auto output = pool.forward(input);

    // 8 / 2 = 4
    TEST_ASSERT_SHAPE(output, std::vector<size_t>({2, 3, 4, 4}));
}

void test_maxpool2d_values() {
    MaxPool2d pool(2, 2);
    // Create input with known values
    auto input = Tensor::create({
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    }, {1, 1, 4, 4});

    auto output = pool.forward(input);

    TEST_ASSERT_SHAPE(output, std::vector<size_t>({1, 1, 2, 2}));
    // Max of each 2x2 region
    TEST_ASSERT_NEAR(output->data[0], 6.0f, 1e-5f);   // max(1,2,5,6)
    TEST_ASSERT_NEAR(output->data[1], 8.0f, 1e-5f);   // max(3,4,7,8)
    TEST_ASSERT_NEAR(output->data[2], 14.0f, 1e-5f);  // max(9,10,13,14)
    TEST_ASSERT_NEAR(output->data[3], 16.0f, 1e-5f);  // max(11,12,15,16)
}

// =============================================================================
// Normalization Tests
// =============================================================================

void test_batchnorm2d_forward() {
    BatchNorm2d bn(16);
    auto input = Tensor::randn({4, 16, 8, 8});
    auto output = bn.forward(input);

    TEST_ASSERT_SHAPE(output, std::vector<size_t>({4, 16, 8, 8}));
}

void test_batchnorm2d_parameters() {
    BatchNorm2d bn(16);
    auto params = bn.parameters();

    TEST_ASSERT_EQ(params.size(), 2u);  // gamma and beta
    TEST_ASSERT_SHAPE(params[0], std::vector<size_t>({16}));
    TEST_ASSERT_SHAPE(params[1], std::vector<size_t>({16}));
}

void test_batchnorm2d_train_eval() {
    BatchNorm2d bn(4);

    bn.train();
    TEST_ASSERT(bn.training == true);

    bn.eval();
    TEST_ASSERT(bn.training == false);
}

void test_layernorm_forward() {
    LayerNorm ln(64);
    auto input = Tensor::randn({4, 10, 64});
    auto output = ln.forward(input);

    TEST_ASSERT_SHAPE(output, std::vector<size_t>({4, 10, 64}));
}

void test_layernorm_parameters() {
    LayerNorm ln(64);
    auto params = ln.parameters();

    TEST_ASSERT_EQ(params.size(), 2u);  // gamma and beta
    TEST_ASSERT_SHAPE(params[0], std::vector<size_t>({64}));
    TEST_ASSERT_SHAPE(params[1], std::vector<size_t>({64}));
}

// =============================================================================
// Flatten Tests
// =============================================================================

void test_flatten_forward() {
    Flatten flatten;
    auto input = Tensor::randn({2, 3, 4, 4});
    auto output = flatten.forward(input);

    // Flatten all dims except batch
    TEST_ASSERT_SHAPE(output, std::vector<size_t>({2, 48}));
}

// =============================================================================
// Embedding Tests
// =============================================================================

void test_embedding_forward() {
    Embedding embed(100, 32);  // vocab=100, dim=32
    auto indices = Tensor::create({0.0f, 5.0f, 10.0f}, {1, 3});
    auto output = embed.forward(indices);

    TEST_ASSERT_SHAPE(output, std::vector<size_t>({1, 3, 32}));
}

void test_embedding_parameters() {
    Embedding embed(100, 32);
    auto params = embed.parameters();

    TEST_ASSERT_EQ(params.size(), 1u);
    TEST_ASSERT_SHAPE(params[0], std::vector<size_t>({100, 32}));
}

// =============================================================================
// LSTM Tests
// =============================================================================

void test_lstm_forward() {
    LSTM lstm(32, 64, true);  // input=32, hidden=64, batch_first=true
    auto input = Tensor::randn({4, 10, 32});  // [batch, seq, input]
    auto output = lstm.forward(input);

    TEST_ASSERT_SHAPE(output, std::vector<size_t>({4, 10, 64}));
}

void test_lstm_hidden_state() {
    LSTM lstm(32, 64, true);
    auto input = Tensor::randn({4, 10, 32});
    lstm.forward(input);

    // h_n and c_n should be set after forward
    TEST_ASSERT(lstm.h_n != nullptr);
    TEST_ASSERT(lstm.c_n != nullptr);
    TEST_ASSERT_SHAPE(lstm.h_n, std::vector<size_t>({4, 64}));
    TEST_ASSERT_SHAPE(lstm.c_n, std::vector<size_t>({4, 64}));
}

void test_lstm_parameters() {
    LSTM lstm(32, 64);
    auto params = lstm.parameters();

    TEST_ASSERT_EQ(params.size(), 4u);  // weight_ih, weight_hh, bias_ih, bias_hh
}

// =============================================================================
// GRU Tests
// =============================================================================

void test_gru_forward() {
    GRU gru(32, 64, true);  // input=32, hidden=64, batch_first=true
    auto input = Tensor::randn({4, 10, 32});
    auto output = gru.forward(input);

    TEST_ASSERT_SHAPE(output, std::vector<size_t>({4, 10, 64}));
}

void test_gru_hidden_state() {
    GRU gru(32, 64, true);
    auto input = Tensor::randn({4, 10, 32});
    gru.forward(input);

    TEST_ASSERT(gru.h_n != nullptr);
    TEST_ASSERT_SHAPE(gru.h_n, std::vector<size_t>({4, 64}));
}

// =============================================================================
// MultiHeadAttention Tests
// =============================================================================

void test_multihead_attention_forward() {
    MultiHeadAttention mha(64, 8);  // embed_dim=64, num_heads=8
    auto input = Tensor::randn({2, 10, 64});  // [batch, seq, embed]
    auto output = mha.forward(input);

    TEST_ASSERT_SHAPE(output, std::vector<size_t>({2, 10, 64}));
}

void test_multihead_attention_cross() {
    MultiHeadAttention mha(64, 8);
    auto query = Tensor::randn({2, 10, 64});
    auto key = Tensor::randn({2, 20, 64});
    auto value = Tensor::randn({2, 20, 64});

    auto output = mha.forward(query, key, value);

    // Output shape matches query sequence length
    TEST_ASSERT_SHAPE(output, std::vector<size_t>({2, 10, 64}));
}

void test_multihead_attention_causal_mask() {
    auto mask = MultiHeadAttention::causal_mask(5);

    // Causal mask should have shape for 5x5 attention
    TEST_ASSERT_EQ(mask->size(), 25u);  // 5*5 = 25

    // The mask should have a causal structure:
    // - Lower triangular (including diagonal) should be 0 (attend)
    // - Upper triangular should be large negative (don't attend)
    // Position (0, 0) should be 0 (can attend to self)
    TEST_ASSERT_NEAR(mask->data[0], 0.0f, 1e-5f);
}

// =============================================================================
// Sequential Tests
// =============================================================================

void test_sequential_forward() {
    Sequential model({
        new Linear(10, 20),
        new ReLU(),
        new Linear(20, 5)
    });

    auto input = Tensor::randn({4, 10});
    auto output = model.forward(input);

    TEST_ASSERT_SHAPE(output, std::vector<size_t>({4, 5}));
}

void test_sequential_parameters() {
    Sequential model({
        new Linear(10, 20),
        new ReLU(),
        new Linear(20, 5)
    });

    auto params = model.parameters();

    // Linear(10, 20) has weight + bias
    // ReLU has no parameters
    // Linear(20, 5) has weight + bias
    TEST_ASSERT_EQ(params.size(), 4u);
}

void test_sequential_train_eval() {
    Sequential model({
        new Dropout(0.5f),
        new Linear(10, 5)
    });

    model.train();
    // Dropout should be in training mode

    model.eval();
    // Dropout should be in eval mode
}

void test_sequential_add() {
    Sequential model;
    model.add(new Linear(10, 20));
    model.add(new ReLU());
    model.add(new Linear(20, 5));

    auto input = Tensor::randn({4, 10});
    auto output = model.forward(input);

    TEST_ASSERT_SHAPE(output, std::vector<size_t>({4, 5}));
}

// =============================================================================
// Model Summary Tests
// =============================================================================

void test_model_summary() {
    Sequential model({
        new Linear(10, 20),
        new ReLU(),
        new Linear(20, 5)
    });

    auto summary = get_model_summary(&model);

    // Linear(10, 20): 10*20 + 20 = 220
    // Linear(20, 5): 20*5 + 5 = 105
    // Total: 325
    TEST_ASSERT_EQ(summary.total_params, 325u);
    TEST_ASSERT_EQ(summary.trainable_params, 325u);
    TEST_ASSERT_EQ(summary.non_trainable_params, 0u);
}

void test_count_parameters() {
    Linear layer(100, 50);
    size_t expected = 100 * 50 + 50;  // weight + bias
    TEST_ASSERT_EQ(count_parameters(&layer), expected);
}

void test_format_memory() {
    std::string mb = format_memory(1024 * 1024);
    TEST_ASSERT(mb.find("MB") != std::string::npos || mb.find("1.00") != std::string::npos);
}

void test_format_number() {
    std::string formatted = format_number(1234567);
    TEST_ASSERT(formatted.find(',') != std::string::npos);
}

// =============================================================================
// Test Suite Registration
// =============================================================================

TestSuite* create_layer_tests() {
    auto* suite = new TestSuite("Layers");

    // Linear tests
    suite->add_test("linear_forward", test_linear_forward);
    suite->add_test("linear_parameters", test_linear_parameters);
    suite->add_test("linear_num_parameters", test_linear_num_parameters);
    suite->add_test("linear_gradient", test_linear_gradient);

    // Activation tests
    suite->add_test("relu_forward", test_relu_forward);
    suite->add_test("sigmoid_forward", test_sigmoid_forward);
    suite->add_test("tanh_forward", test_tanh_forward);
    suite->add_test("softmax_forward", test_softmax_forward);
    suite->add_test("logsoftmax_forward", test_logsoftmax_forward);

    // Dropout tests
    suite->add_test("dropout_training", test_dropout_training);
    suite->add_test("dropout_eval", test_dropout_eval);

    // Conv2d tests
    suite->add_test("conv2d_forward", test_conv2d_forward);
    suite->add_test("conv2d_parameters", test_conv2d_parameters);
    suite->add_test("conv2d_no_padding", test_conv2d_no_padding);
    suite->add_test("conv2d_stride", test_conv2d_stride);

    // ConvTranspose2d tests
    suite->add_test("conv_transpose2d_forward", test_conv_transpose2d_forward);
    suite->add_test("conv_transpose2d_upsample", test_conv_transpose2d_upsample);
    suite->add_test("conv_transpose2d_parameters", test_conv_transpose2d_parameters);
    suite->add_test("conv_transpose2d_gradient", test_conv_transpose2d_gradient);
    suite->add_test("conv_transpose2d_output_padding", test_conv_transpose2d_output_padding);
    suite->add_test("conv_transpose2d_inverse_conv2d", test_conv_transpose2d_inverse_conv2d);

    // Pooling tests
    suite->add_test("maxpool2d_forward", test_maxpool2d_forward);
    suite->add_test("avgpool2d_forward", test_avgpool2d_forward);
    suite->add_test("maxpool2d_values", test_maxpool2d_values);

    // Normalization tests
    suite->add_test("batchnorm2d_forward", test_batchnorm2d_forward);
    suite->add_test("batchnorm2d_parameters", test_batchnorm2d_parameters);
    suite->add_test("batchnorm2d_train_eval", test_batchnorm2d_train_eval);
    suite->add_test("layernorm_forward", test_layernorm_forward);
    suite->add_test("layernorm_parameters", test_layernorm_parameters);

    // Flatten tests
    suite->add_test("flatten_forward", test_flatten_forward);

    // Embedding tests
    suite->add_test("embedding_forward", test_embedding_forward);
    suite->add_test("embedding_parameters", test_embedding_parameters);

    // LSTM tests
    suite->add_test("lstm_forward", test_lstm_forward);
    suite->add_test("lstm_hidden_state", test_lstm_hidden_state);
    suite->add_test("lstm_parameters", test_lstm_parameters);

    // GRU tests
    suite->add_test("gru_forward", test_gru_forward);
    suite->add_test("gru_hidden_state", test_gru_hidden_state);

    // MultiHeadAttention tests
    suite->add_test("mha_forward", test_multihead_attention_forward);
    suite->add_test("mha_cross", test_multihead_attention_cross);
    suite->add_test("mha_causal_mask", test_multihead_attention_causal_mask);

    // Sequential tests
    suite->add_test("sequential_forward", test_sequential_forward);
    suite->add_test("sequential_parameters", test_sequential_parameters);
    suite->add_test("sequential_train_eval", test_sequential_train_eval);
    suite->add_test("sequential_add", test_sequential_add);

    // Model summary tests
    suite->add_test("model_summary", test_model_summary);
    suite->add_test("count_parameters", test_count_parameters);
    suite->add_test("format_memory", test_format_memory);
    suite->add_test("format_number", test_format_number);

    return suite;
}
