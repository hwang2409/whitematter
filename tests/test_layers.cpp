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

void test_lstm_variable_sequence_lengths() {
    // Test LSTM with different sequence lengths
    LSTM lstm(16, 32, true);

    // Short sequence
    auto input_short = Tensor::randn({2, 5, 16});
    auto output_short = lstm.forward(input_short);
    TEST_ASSERT_SHAPE(output_short, std::vector<size_t>({2, 5, 32}));

    // Long sequence
    auto input_long = Tensor::randn({2, 20, 16});
    auto output_long = lstm.forward(input_long);
    TEST_ASSERT_SHAPE(output_long, std::vector<size_t>({2, 20, 32}));

    // Single timestep
    auto input_single = Tensor::randn({2, 1, 16});
    auto output_single = lstm.forward(input_single);
    TEST_ASSERT_SHAPE(output_single, std::vector<size_t>({2, 1, 32}));
}

void test_lstm_seq_first() {
    // Test with batch_first=false (seq, batch, input)
    LSTM lstm(16, 32, false);
    auto input = Tensor::randn({10, 4, 16});  // [seq, batch, input]
    auto output = lstm.forward(input);

    TEST_ASSERT_SHAPE(output, std::vector<size_t>({10, 4, 32}));  // [seq, batch, hidden]
    TEST_ASSERT_SHAPE(lstm.h_n, std::vector<size_t>({4, 32}));
}

void test_lstm_initial_hidden_state() {
    LSTM lstm(16, 32, true);
    auto input = Tensor::randn({2, 5, 16});

    // Provide custom initial hidden states
    auto h0 = Tensor::ones({2, 32}, false);
    auto c0 = Tensor::zeros({2, 32}, false);

    auto output = lstm.forward(input, h0, c0);

    TEST_ASSERT_SHAPE(output, std::vector<size_t>({2, 5, 32}));

    // Output should be different from default zero initialization
    // (since we passed ones for h0)
    // This is a smoke test - the values should be computed correctly
    TEST_ASSERT(output->data.size() > 0);
}

void test_lstm_gradient_flow() {
    LSTM lstm(8, 16, true);
    auto input = Tensor::randn({2, 4, 8}, true);
    auto output = lstm.forward(input);
    auto loss = output->sum();

    loss->backward();

    // Input should have gradients
    TEST_ASSERT(input->grad.size() > 0);
    bool has_nonzero_input_grad = false;
    for (size_t i = 0; i < input->grad.size(); i++) {
        if (std::abs(input->grad[i]) > 1e-7f) {
            has_nonzero_input_grad = true;
            break;
        }
    }
    TEST_ASSERT(has_nonzero_input_grad);

    // Weight matrices should have gradients
    TEST_ASSERT(lstm.weight_ih->grad.size() > 0);
    TEST_ASSERT(lstm.weight_hh->grad.size() > 0);
    TEST_ASSERT(lstm.bias_ih->grad.size() > 0);
    TEST_ASSERT(lstm.bias_hh->grad.size() > 0);

    // Check weight_ih gradient is non-zero
    bool has_nonzero_weight_grad = false;
    for (size_t i = 0; i < lstm.weight_ih->grad.size(); i++) {
        if (std::abs(lstm.weight_ih->grad[i]) > 1e-7f) {
            has_nonzero_weight_grad = true;
            break;
        }
    }
    TEST_ASSERT(has_nonzero_weight_grad);
}

void test_lstm_weight_shapes() {
    LSTM lstm(10, 20, true);
    auto params = lstm.parameters();

    // weight_ih: [4*hidden_size, input_size] = [80, 10]
    TEST_ASSERT_SHAPE(params[0], std::vector<size_t>({80, 10}));

    // weight_hh: [4*hidden_size, hidden_size] = [80, 20]
    TEST_ASSERT_SHAPE(params[1], std::vector<size_t>({80, 20}));

    // bias_ih: [4*hidden_size] = [80]
    TEST_ASSERT_SHAPE(params[2], std::vector<size_t>({80}));

    // bias_hh: [4*hidden_size] = [80]
    TEST_ASSERT_SHAPE(params[3], std::vector<size_t>({80}));
}

void test_lstm_output_values_bounded() {
    // LSTM output uses tanh, so hidden state should be in [-1, 1]
    LSTM lstm(8, 16, true);
    auto input = Tensor::randn({4, 10, 8});
    auto output = lstm.forward(input);

    for (size_t i = 0; i < output->data.size(); i++) {
        TEST_ASSERT(output->data[i] >= -1.0f && output->data[i] <= 1.0f);
    }
}

void test_lstm_forget_gate_bias() {
    // LSTM initializes forget gate bias to 1.0 for better gradient flow
    LSTM lstm(8, 16, true);

    // Forget gate bias is in bias_ih[hidden_size:2*hidden_size]
    for (size_t i = 16; i < 32; i++) {
        TEST_ASSERT_NEAR(lstm.bias_ih->data[i], 1.0f, 1e-5f);
    }
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

void test_gru_parameters() {
    GRU gru(32, 64);
    auto params = gru.parameters();

    TEST_ASSERT_EQ(params.size(), 4u);  // weight_ih, weight_hh, bias_ih, bias_hh
}

void test_gru_variable_sequence_lengths() {
    // Test GRU with different sequence lengths
    GRU gru(16, 32, true);

    // Short sequence
    auto input_short = Tensor::randn({2, 5, 16});
    auto output_short = gru.forward(input_short);
    TEST_ASSERT_SHAPE(output_short, std::vector<size_t>({2, 5, 32}));

    // Long sequence
    auto input_long = Tensor::randn({2, 20, 16});
    auto output_long = gru.forward(input_long);
    TEST_ASSERT_SHAPE(output_long, std::vector<size_t>({2, 20, 32}));

    // Single timestep
    auto input_single = Tensor::randn({2, 1, 16});
    auto output_single = gru.forward(input_single);
    TEST_ASSERT_SHAPE(output_single, std::vector<size_t>({2, 1, 32}));
}

void test_gru_seq_first() {
    // Test with batch_first=false (seq, batch, input)
    GRU gru(16, 32, false);
    auto input = Tensor::randn({10, 4, 16});  // [seq, batch, input]
    auto output = gru.forward(input);

    TEST_ASSERT_SHAPE(output, std::vector<size_t>({10, 4, 32}));  // [seq, batch, hidden]
    TEST_ASSERT_SHAPE(gru.h_n, std::vector<size_t>({4, 32}));
}

void test_gru_initial_hidden_state() {
    GRU gru(16, 32, true);
    auto input = Tensor::randn({2, 5, 16});

    // Provide custom initial hidden state
    auto h0 = Tensor::ones({2, 32}, false);
    auto output = gru.forward(input, h0);

    TEST_ASSERT_SHAPE(output, std::vector<size_t>({2, 5, 32}));
    TEST_ASSERT(output->data.size() > 0);
}

void test_gru_gradient_flow() {
    GRU gru(8, 16, true);
    auto input = Tensor::randn({2, 4, 8}, true);
    auto output = gru.forward(input);
    auto loss = output->sum();

    loss->backward();

    // Input should have gradients
    TEST_ASSERT(input->grad.size() > 0);
    bool has_nonzero_input_grad = false;
    for (size_t i = 0; i < input->grad.size(); i++) {
        if (std::abs(input->grad[i]) > 1e-7f) {
            has_nonzero_input_grad = true;
            break;
        }
    }
    TEST_ASSERT(has_nonzero_input_grad);

    // Weight matrices should have gradients
    TEST_ASSERT(gru.weight_ih->grad.size() > 0);
    TEST_ASSERT(gru.weight_hh->grad.size() > 0);
    TEST_ASSERT(gru.bias_ih->grad.size() > 0);
    TEST_ASSERT(gru.bias_hh->grad.size() > 0);

    // Check weight_ih gradient is non-zero
    bool has_nonzero_weight_grad = false;
    for (size_t i = 0; i < gru.weight_ih->grad.size(); i++) {
        if (std::abs(gru.weight_ih->grad[i]) > 1e-7f) {
            has_nonzero_weight_grad = true;
            break;
        }
    }
    TEST_ASSERT(has_nonzero_weight_grad);
}

void test_gru_weight_shapes() {
    GRU gru(10, 20, true);
    auto params = gru.parameters();

    // weight_ih: [3*hidden_size, input_size] = [60, 10]
    TEST_ASSERT_SHAPE(params[0], std::vector<size_t>({60, 10}));

    // weight_hh: [3*hidden_size, hidden_size] = [60, 20]
    TEST_ASSERT_SHAPE(params[1], std::vector<size_t>({60, 20}));

    // bias_ih: [3*hidden_size] = [60]
    TEST_ASSERT_SHAPE(params[2], std::vector<size_t>({60}));

    // bias_hh: [3*hidden_size] = [60]
    TEST_ASSERT_SHAPE(params[3], std::vector<size_t>({60}));
}

void test_gru_output_continuity() {
    // Final hidden state should match the last timestep of output
    GRU gru(8, 16, true);
    auto input = Tensor::randn({2, 5, 8});
    auto output = gru.forward(input);

    // h_n should equal output[:, -1, :]
    for (size_t b = 0; b < 2; b++) {
        for (size_t h = 0; h < 16; h++) {
            size_t output_idx = b * 5 * 16 + 4 * 16 + h;  // last timestep
            size_t h_n_idx = b * 16 + h;
            TEST_ASSERT_NEAR(gru.h_n->data[h_n_idx], output->data[output_idx], 1e-5f);
        }
    }
}

void test_gru_deterministic() {
    // Same input should produce same output (with same weights)
    GRU gru1(8, 16, true);
    GRU gru2(8, 16, true);

    // Copy weights from gru1 to gru2
    for (size_t i = 0; i < gru1.weight_ih->data.size(); i++) {
        gru2.weight_ih->data[i] = gru1.weight_ih->data[i];
    }
    for (size_t i = 0; i < gru1.weight_hh->data.size(); i++) {
        gru2.weight_hh->data[i] = gru1.weight_hh->data[i];
    }
    for (size_t i = 0; i < gru1.bias_ih->data.size(); i++) {
        gru2.bias_ih->data[i] = gru1.bias_ih->data[i];
    }
    for (size_t i = 0; i < gru1.bias_hh->data.size(); i++) {
        gru2.bias_hh->data[i] = gru1.bias_hh->data[i];
    }

    auto input = Tensor::randn({2, 5, 8});
    auto output1 = gru1.forward(input);
    auto output2 = gru2.forward(input);

    for (size_t i = 0; i < output1->data.size(); i++) {
        TEST_ASSERT_NEAR(output1->data[i], output2->data[i], 1e-5f);
    }
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

void test_multihead_attention_parameters() {
    MultiHeadAttention mha(64, 8);
    auto params = mha.parameters();

    // W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o = 8 parameters
    TEST_ASSERT_EQ(params.size(), 8u);

    // Check weight shapes: [embed_dim, embed_dim]
    TEST_ASSERT_SHAPE(params[0], std::vector<size_t>({64, 64}));  // W_q
    TEST_ASSERT_SHAPE(params[1], std::vector<size_t>({64, 64}));  // W_k
    TEST_ASSERT_SHAPE(params[2], std::vector<size_t>({64, 64}));  // W_v
    TEST_ASSERT_SHAPE(params[3], std::vector<size_t>({64, 64}));  // W_o

    // Check bias shapes: [embed_dim]
    TEST_ASSERT_SHAPE(params[4], std::vector<size_t>({64}));  // b_q
    TEST_ASSERT_SHAPE(params[5], std::vector<size_t>({64}));  // b_k
    TEST_ASSERT_SHAPE(params[6], std::vector<size_t>({64}));  // b_v
    TEST_ASSERT_SHAPE(params[7], std::vector<size_t>({64}));  // b_o
}

void test_multihead_attention_with_mask() {
    MultiHeadAttention mha(32, 4);  // embed_dim=32, num_heads=4
    auto input = Tensor::randn({2, 6, 32});

    // Create a causal mask
    auto mask = MultiHeadAttention::causal_mask(6);

    auto output = mha.forward(input, input, input, mask);

    TEST_ASSERT_SHAPE(output, std::vector<size_t>({2, 6, 32}));

    // Output should still be valid
    for (size_t i = 0; i < output->data.size(); i++) {
        TEST_ASSERT(!std::isnan(output->data[i]));
        TEST_ASSERT(!std::isinf(output->data[i]));
    }
}

void test_multihead_attention_causal_mask_structure() {
    auto mask = MultiHeadAttention::causal_mask(4);

    // Check shape: [1, 1, 4, 4]
    TEST_ASSERT_EQ(mask->shape.size(), 4u);
    TEST_ASSERT_EQ(mask->shape[0], 1u);
    TEST_ASSERT_EQ(mask->shape[1], 1u);
    TEST_ASSERT_EQ(mask->shape[2], 4u);
    TEST_ASSERT_EQ(mask->shape[3], 4u);

    // Verify causal structure
    // Position (i, j) should be 0 if j <= i (can attend), -inf if j > i (can't attend)
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            float val = mask->data[i * 4 + j];
            if (j <= i) {
                TEST_ASSERT_NEAR(val, 0.0f, 1e-5f);
            } else {
                TEST_ASSERT(val < -1e6f);  // Large negative value
            }
        }
    }
}

void test_multihead_attention_no_mask() {
    // Test that attention without mask attends to all positions
    MultiHeadAttention mha(32, 4);
    auto input = Tensor::randn({1, 5, 32});

    auto output_no_mask = mha.forward(input, input, input, nullptr);

    TEST_ASSERT_SHAPE(output_no_mask, std::vector<size_t>({1, 5, 32}));

    // Check that attention weights exist and are valid probabilities
    TEST_ASSERT(mha.attn_weights != nullptr);
    TEST_ASSERT_EQ(mha.attn_weights->shape.size(), 4u);  // [batch, heads, seq_q, seq_k]
}

void test_multihead_attention_attention_weights_sum() {
    MultiHeadAttention mha(32, 4);
    auto input = Tensor::randn({2, 6, 32});
    mha.forward(input);

    // Attention weights should sum to 1 along the last dimension (seq_k)
    auto attn = mha.attn_weights;
    size_t batch = attn->shape[0];
    size_t heads = attn->shape[1];
    size_t seq_q = attn->shape[2];
    size_t seq_k = attn->shape[3];

    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < heads; h++) {
            for (size_t i = 0; i < seq_q; i++) {
                float sum = 0.0f;
                for (size_t j = 0; j < seq_k; j++) {
                    size_t idx = b * heads * seq_q * seq_k + h * seq_q * seq_k + i * seq_k + j;
                    sum += attn->data[idx];
                }
                TEST_ASSERT_NEAR(sum, 1.0f, 1e-4f);
            }
        }
    }
}

void test_multihead_attention_gradient_flow() {
    MultiHeadAttention mha(32, 4);
    auto input = Tensor::randn({2, 5, 32}, true);
    auto output = mha.forward(input);
    auto loss = output->sum();

    loss->backward();

    // Input should have gradients
    TEST_ASSERT(input->grad.size() > 0);
    bool has_nonzero_input_grad = false;
    for (size_t i = 0; i < input->grad.size(); i++) {
        if (std::abs(input->grad[i]) > 1e-7f) {
            has_nonzero_input_grad = true;
            break;
        }
    }
    TEST_ASSERT(has_nonzero_input_grad);

    // Weight matrices should have gradients
    TEST_ASSERT(mha.W_q->grad.size() > 0);
    TEST_ASSERT(mha.W_k->grad.size() > 0);
    TEST_ASSERT(mha.W_v->grad.size() > 0);
    TEST_ASSERT(mha.W_o->grad.size() > 0);

    // Check W_q gradient is non-zero
    bool has_nonzero_weight_grad = false;
    for (size_t i = 0; i < mha.W_q->grad.size(); i++) {
        if (std::abs(mha.W_q->grad[i]) > 1e-7f) {
            has_nonzero_weight_grad = true;
            break;
        }
    }
    TEST_ASSERT(has_nonzero_weight_grad);

    // Biases should have gradients
    TEST_ASSERT(mha.b_q->grad.size() > 0);
    TEST_ASSERT(mha.b_o->grad.size() > 0);
}

void test_multihead_attention_cross_gradient() {
    // Test gradient flow in cross-attention
    MultiHeadAttention mha(32, 4);
    auto query = Tensor::randn({2, 5, 32}, true);
    auto key = Tensor::randn({2, 8, 32}, true);
    auto value = Tensor::randn({2, 8, 32}, true);

    auto output = mha.forward(query, key, value);
    auto loss = output->sum();

    loss->backward();

    // All inputs should have gradients
    TEST_ASSERT(query->grad.size() > 0);
    TEST_ASSERT(key->grad.size() > 0);
    TEST_ASSERT(value->grad.size() > 0);

    // Check that gradients are non-zero
    bool query_has_grad = false;
    for (size_t i = 0; i < query->grad.size(); i++) {
        if (std::abs(query->grad[i]) > 1e-7f) {
            query_has_grad = true;
            break;
        }
    }
    TEST_ASSERT(query_has_grad);

    bool key_has_grad = false;
    for (size_t i = 0; i < key->grad.size(); i++) {
        if (std::abs(key->grad[i]) > 1e-7f) {
            key_has_grad = true;
            break;
        }
    }
    TEST_ASSERT(key_has_grad);

    bool value_has_grad = false;
    for (size_t i = 0; i < value->grad.size(); i++) {
        if (std::abs(value->grad[i]) > 1e-7f) {
            value_has_grad = true;
            break;
        }
    }
    TEST_ASSERT(value_has_grad);
}

void test_multihead_attention_head_dim() {
    // Test that head_dim = embed_dim / num_heads
    MultiHeadAttention mha(64, 8);  // 64/8 = 8

    TEST_ASSERT_EQ(mha.head_dim, 8u);
    TEST_ASSERT_EQ(mha.num_heads, 8u);
    TEST_ASSERT_EQ(mha.embed_dim, 64u);
}

void test_multihead_attention_single_head() {
    // Test with single head (standard attention)
    MultiHeadAttention mha(32, 1);
    auto input = Tensor::randn({2, 5, 32});
    auto output = mha.forward(input);

    TEST_ASSERT_SHAPE(output, std::vector<size_t>({2, 5, 32}));
    TEST_ASSERT_EQ(mha.head_dim, 32u);
}

void test_multihead_attention_variable_seq_len() {
    MultiHeadAttention mha(32, 4);

    // Short sequence
    auto input_short = Tensor::randn({2, 3, 32});
    auto output_short = mha.forward(input_short);
    TEST_ASSERT_SHAPE(output_short, std::vector<size_t>({2, 3, 32}));

    // Long sequence
    auto input_long = Tensor::randn({2, 20, 32});
    auto output_long = mha.forward(input_long);
    TEST_ASSERT_SHAPE(output_long, std::vector<size_t>({2, 20, 32}));

    // Single token
    auto input_single = Tensor::randn({2, 1, 32});
    auto output_single = mha.forward(input_single);
    TEST_ASSERT_SHAPE(output_single, std::vector<size_t>({2, 1, 32}));
}

void test_multihead_attention_numerical_stability() {
    // Test with large/small values
    MultiHeadAttention mha(32, 4);

    // Normal input
    auto input = Tensor::randn({1, 5, 32});

    // Scale input to test numerical stability
    for (size_t i = 0; i < input->data.size(); i++) {
        input->data[i] *= 10.0f;  // Larger values
    }

    auto output = mha.forward(input);

    // Output should not have NaN or Inf
    for (size_t i = 0; i < output->data.size(); i++) {
        TEST_ASSERT(!std::isnan(output->data[i]));
        TEST_ASSERT(!std::isinf(output->data[i]));
    }
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
    suite->add_test("lstm_variable_sequence_lengths", test_lstm_variable_sequence_lengths);
    suite->add_test("lstm_seq_first", test_lstm_seq_first);
    suite->add_test("lstm_initial_hidden_state", test_lstm_initial_hidden_state);
    suite->add_test("lstm_gradient_flow", test_lstm_gradient_flow);
    suite->add_test("lstm_weight_shapes", test_lstm_weight_shapes);
    suite->add_test("lstm_output_values_bounded", test_lstm_output_values_bounded);
    suite->add_test("lstm_forget_gate_bias", test_lstm_forget_gate_bias);

    // GRU tests
    suite->add_test("gru_forward", test_gru_forward);
    suite->add_test("gru_hidden_state", test_gru_hidden_state);
    suite->add_test("gru_parameters", test_gru_parameters);
    suite->add_test("gru_variable_sequence_lengths", test_gru_variable_sequence_lengths);
    suite->add_test("gru_seq_first", test_gru_seq_first);
    suite->add_test("gru_initial_hidden_state", test_gru_initial_hidden_state);
    suite->add_test("gru_gradient_flow", test_gru_gradient_flow);
    suite->add_test("gru_weight_shapes", test_gru_weight_shapes);
    suite->add_test("gru_output_continuity", test_gru_output_continuity);
    suite->add_test("gru_deterministic", test_gru_deterministic);

    // MultiHeadAttention tests
    suite->add_test("mha_forward", test_multihead_attention_forward);
    suite->add_test("mha_cross", test_multihead_attention_cross);
    suite->add_test("mha_causal_mask", test_multihead_attention_causal_mask);
    suite->add_test("mha_parameters", test_multihead_attention_parameters);
    suite->add_test("mha_with_mask", test_multihead_attention_with_mask);
    suite->add_test("mha_causal_mask_structure", test_multihead_attention_causal_mask_structure);
    suite->add_test("mha_no_mask", test_multihead_attention_no_mask);
    suite->add_test("mha_attention_weights_sum", test_multihead_attention_attention_weights_sum);
    suite->add_test("mha_gradient_flow", test_multihead_attention_gradient_flow);
    suite->add_test("mha_cross_gradient", test_multihead_attention_cross_gradient);
    suite->add_test("mha_head_dim", test_multihead_attention_head_dim);
    suite->add_test("mha_single_head", test_multihead_attention_single_head);
    suite->add_test("mha_variable_seq_len", test_multihead_attention_variable_seq_len);
    suite->add_test("mha_numerical_stability", test_multihead_attention_numerical_stability);

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
