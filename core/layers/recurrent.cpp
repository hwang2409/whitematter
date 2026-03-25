#include "../layer.h"
#include <random>
#include <cmath>
#include <cassert>

static std::mt19937 recurrent_rng(123);

// LSTM implementation
LSTM::LSTM(size_t input_size, size_t hidden_size, bool batch_first)
    : input_size(input_size), hidden_size(hidden_size), batch_first(batch_first) {
    // Xavier/Glorot initialization
    float std_ih = std::sqrt(2.0f / (input_size + hidden_size));
    float std_hh = std::sqrt(2.0f / (hidden_size + hidden_size));
    std::normal_distribution<float> dist_ih(0.0f, std_ih);
    std::normal_distribution<float> dist_hh(0.0f, std_hh);

    // Weight matrices: [4*hidden_size, input_size] and [4*hidden_size, hidden_size]
    weight_ih = Tensor::create({4 * hidden_size, input_size}, true);
    weight_hh = Tensor::create({4 * hidden_size, hidden_size}, true);
    bias_ih = Tensor::zeros({4 * hidden_size}, true);
    bias_hh = Tensor::zeros({4 * hidden_size}, true);

    for (size_t i = 0; i < weight_ih->size(); i++) weight_ih->data()[i] = dist_ih(recurrent_rng);
    for (size_t i = 0; i < weight_hh->size(); i++) weight_hh->data()[i] = dist_hh(recurrent_rng);

    // Initialize forget gate bias to 1.0 for better gradient flow
    for (size_t i = hidden_size; i < 2 * hidden_size; i++) {
        bias_ih->data()[i] = 1.0f;
    }
}

TensorPtr LSTM::forward(const TensorPtr& input) {
    // Initialize hidden states to zeros
    size_t batch_size = batch_first ? input->shape[0] : input->shape[1];
    auto h0 = Tensor::zeros({batch_size, hidden_size}, false);
    auto c0 = Tensor::zeros({batch_size, hidden_size}, false);
    return forward(input, h0, c0);
}

TensorPtr LSTM::forward(const TensorPtr& input, const TensorPtr& h0, const TensorPtr& c0) {
    // Input: [batch, seq, input_size] if batch_first else [seq, batch, input_size]
    // Output: [batch, seq, hidden_size] if batch_first else [seq, batch, hidden_size]
    assert(input->shape.size() == 3);

    size_t batch_size, seq_len;
    if (batch_first) {
        batch_size = input->shape[0];
        seq_len = input->shape[1];
        assert(input->shape[2] == input_size);
    } else {
        seq_len = input->shape[0];
        batch_size = input->shape[1];
        assert(input->shape[2] == input_size);
    }

    bool track = input->requires_grad && GradMode::is_enabled();

    // Output tensor for all hidden states
    std::vector<size_t> out_shape;
    if (batch_first) {
        out_shape = {batch_size, seq_len, hidden_size};
    } else {
        out_shape = {seq_len, batch_size, hidden_size};
    }
    auto output = Tensor::create(out_shape, track);

    // Store intermediate values for backward pass
    std::vector<std::vector<float>> all_i(seq_len), all_f(seq_len), all_g(seq_len), all_o(seq_len);
    std::vector<std::vector<float>> all_c(seq_len + 1), all_h(seq_len + 1);
    std::vector<std::vector<float>> all_tanh_c(seq_len);

    // Initialize h and c from h0 and c0
    all_h[0].resize(batch_size * hidden_size);
    all_c[0].resize(batch_size * hidden_size);
    for (size_t i = 0; i < batch_size * hidden_size; i++) {
        all_h[0][i] = h0->data()[i];
        all_c[0][i] = c0->data()[i];
    }

    // Forward through time
    for (size_t t = 0; t < seq_len; t++) {
        all_i[t].resize(batch_size * hidden_size);
        all_f[t].resize(batch_size * hidden_size);
        all_g[t].resize(batch_size * hidden_size);
        all_o[t].resize(batch_size * hidden_size);
        all_c[t + 1].resize(batch_size * hidden_size);
        all_h[t + 1].resize(batch_size * hidden_size);
        all_tanh_c[t].resize(batch_size * hidden_size);

        for (size_t b = 0; b < batch_size; b++) {
            // Compute gates for each batch element
            // gates = x_t @ W_ih^T + h_{t-1} @ W_hh^T + b_ih + b_hh
            std::vector<float> gates(4 * hidden_size, 0.0f);

            // x_t @ W_ih^T
            size_t x_offset = batch_first ?
                (b * seq_len * input_size + t * input_size) :
                (t * batch_size * input_size + b * input_size);

            for (size_t g = 0; g < 4 * hidden_size; g++) {
                for (size_t i = 0; i < input_size; i++) {
                    gates[g] += input->data()[x_offset + i] * weight_ih->data()[g * input_size + i];
                }
                gates[g] += bias_ih->data()[g];
            }

            // h_{t-1} @ W_hh^T
            for (size_t g = 0; g < 4 * hidden_size; g++) {
                for (size_t h = 0; h < hidden_size; h++) {
                    gates[g] += all_h[t][b * hidden_size + h] * weight_hh->data()[g * hidden_size + h];
                }
                gates[g] += bias_hh->data()[g];
            }

            // Split into i, f, g, o and apply activations
            for (size_t h = 0; h < hidden_size; h++) {
                size_t idx = b * hidden_size + h;
                float i_gate = 1.0f / (1.0f + std::exp(-gates[h]));                      // sigmoid
                float f_gate = 1.0f / (1.0f + std::exp(-gates[hidden_size + h]));        // sigmoid
                float g_gate = std::tanh(gates[2 * hidden_size + h]);                     // tanh
                float o_gate = 1.0f / (1.0f + std::exp(-gates[3 * hidden_size + h]));    // sigmoid

                all_i[t][idx] = i_gate;
                all_f[t][idx] = f_gate;
                all_g[t][idx] = g_gate;
                all_o[t][idx] = o_gate;

                // c_t = f_t * c_{t-1} + i_t * g_t
                float c_new = f_gate * all_c[t][idx] + i_gate * g_gate;
                all_c[t + 1][idx] = c_new;

                // h_t = o_t * tanh(c_t)
                float tanh_c = std::tanh(c_new);
                all_tanh_c[t][idx] = tanh_c;
                float h_new = o_gate * tanh_c;
                all_h[t + 1][idx] = h_new;

                // Store in output
                size_t out_offset = batch_first ?
                    (b * seq_len * hidden_size + t * hidden_size + h) :
                    (t * batch_size * hidden_size + b * hidden_size + h);
                output->data()[out_offset] = h_new;
            }
        }
    }

    // Store final hidden and cell states
    h_n = Tensor::create({batch_size, hidden_size}, false);
    c_n = Tensor::create({batch_size, hidden_size}, false);
    for (size_t i = 0; i < batch_size * hidden_size; i++) {
        h_n->data()[i] = all_h[seq_len][i];
        c_n->data()[i] = all_c[seq_len][i];
    }

    if (track) {
        auto input_ptr = input;
        auto weight_ih_ptr = weight_ih;
        auto weight_hh_ptr = weight_hh;
        auto bias_ih_ptr = bias_ih;
        auto bias_hh_ptr = bias_hh;
        size_t hs = hidden_size;
        size_t is = input_size;
        bool bf = batch_first;

        output->parents = {input_ptr, weight_ih_ptr, weight_hh_ptr, bias_ih_ptr, bias_hh_ptr};
        output->grad_fn = [=]() mutable {
            // Backpropagation through time (BPTT)
            std::vector<float> dh_next(batch_size * hs, 0.0f);
            std::vector<float> dc_next(batch_size * hs, 0.0f);

            for (int t = static_cast<int>(seq_len) - 1; t >= 0; t--) {
                for (size_t b = 0; b < batch_size; b++) {
                    for (size_t h = 0; h < hs; h++) {
                        size_t idx = b * hs + h;
                        size_t out_offset = bf ?
                            (b * seq_len * hs + t * hs + h) :
                            (t * batch_size * hs + b * hs + h);

                        // Gradient from output and from next time step
                        float dh = output->grad()[out_offset] + dh_next[idx];

                        // h_t = o_t * tanh(c_t)
                        float do_gate = dh * all_tanh_c[t][idx];
                        float dtanh_c = dh * all_o[t][idx];

                        // tanh'(c_t) = 1 - tanh^2(c_t)
                        float dc = dtanh_c * (1.0f - all_tanh_c[t][idx] * all_tanh_c[t][idx]) + dc_next[idx];

                        // c_t = f_t * c_{t-1} + i_t * g_t
                        float di_gate = dc * all_g[t][idx];
                        float df_gate = dc * all_c[t][idx];
                        float dg_gate = dc * all_i[t][idx];
                        dc_next[idx] = dc * all_f[t][idx];

                        // Gate gradients (pre-activation)
                        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
                        // tanh'(x) = 1 - tanh^2(x)
                        float di_pre = di_gate * all_i[t][idx] * (1.0f - all_i[t][idx]);
                        float df_pre = df_gate * all_f[t][idx] * (1.0f - all_f[t][idx]);
                        float dg_pre = dg_gate * (1.0f - all_g[t][idx] * all_g[t][idx]);
                        float do_pre = do_gate * all_o[t][idx] * (1.0f - all_o[t][idx]);

                        // Gradients for weights and biases
                        size_t x_offset = bf ?
                            (b * seq_len * is + t * is) :
                            (t * batch_size * is + b * is);

                        // d_bias_ih and d_bias_hh
                        bias_ih_ptr->grad()[h] += di_pre;
                        bias_ih_ptr->grad()[hs + h] += df_pre;
                        bias_ih_ptr->grad()[2 * hs + h] += dg_pre;
                        bias_ih_ptr->grad()[3 * hs + h] += do_pre;

                        bias_hh_ptr->grad()[h] += di_pre;
                        bias_hh_ptr->grad()[hs + h] += df_pre;
                        bias_hh_ptr->grad()[2 * hs + h] += dg_pre;
                        bias_hh_ptr->grad()[3 * hs + h] += do_pre;

                        // d_weight_ih: [4*hs, is]
                        for (size_t i = 0; i < is; i++) {
                            weight_ih_ptr->grad()[h * is + i] += di_pre * input_ptr->data()[x_offset + i];
                            weight_ih_ptr->grad()[(hs + h) * is + i] += df_pre * input_ptr->data()[x_offset + i];
                            weight_ih_ptr->grad()[(2 * hs + h) * is + i] += dg_pre * input_ptr->data()[x_offset + i];
                            weight_ih_ptr->grad()[(3 * hs + h) * is + i] += do_pre * input_ptr->data()[x_offset + i];
                        }

                        // d_weight_hh: [4*hs, hs]
                        for (size_t hh = 0; hh < hs; hh++) {
                            weight_hh_ptr->grad()[h * hs + hh] += di_pre * all_h[t][b * hs + hh];
                            weight_hh_ptr->grad()[(hs + h) * hs + hh] += df_pre * all_h[t][b * hs + hh];
                            weight_hh_ptr->grad()[(2 * hs + h) * hs + hh] += dg_pre * all_h[t][b * hs + hh];
                            weight_hh_ptr->grad()[(3 * hs + h) * hs + hh] += do_pre * all_h[t][b * hs + hh];
                        }

                        // d_input
                        if (input_ptr->requires_grad) {
                            for (size_t i = 0; i < is; i++) {
                                input_ptr->grad()[x_offset + i] +=
                                    di_pre * weight_ih_ptr->data()[h * is + i] +
                                    df_pre * weight_ih_ptr->data()[(hs + h) * is + i] +
                                    dg_pre * weight_ih_ptr->data()[(2 * hs + h) * is + i] +
                                    do_pre * weight_ih_ptr->data()[(3 * hs + h) * is + i];
                            }
                        }

                        // d_h_prev
                        dh_next[idx] = 0.0f;
                        for (size_t hh = 0; hh < hs; hh++) {
                            dh_next[b * hs + hh] +=
                                di_pre * weight_hh_ptr->data()[h * hs + hh] +
                                df_pre * weight_hh_ptr->data()[(hs + h) * hs + hh] +
                                dg_pre * weight_hh_ptr->data()[(2 * hs + h) * hs + hh] +
                                do_pre * weight_hh_ptr->data()[(3 * hs + h) * hs + hh];
                        }
                    }
                }
            }
        };
    }

    return output;
}

std::vector<TensorPtr> LSTM::parameters() {
    return {weight_ih, weight_hh, bias_ih, bias_hh};
}

// GRU implementation
GRU::GRU(size_t input_size, size_t hidden_size, bool batch_first)
    : input_size(input_size), hidden_size(hidden_size), batch_first(batch_first) {
    // Xavier/Glorot initialization
    float std_ih = std::sqrt(2.0f / (input_size + hidden_size));
    float std_hh = std::sqrt(2.0f / (hidden_size + hidden_size));
    std::normal_distribution<float> dist_ih(0.0f, std_ih);
    std::normal_distribution<float> dist_hh(0.0f, std_hh);

    // Weight matrices: [3*hidden_size, input_size] and [3*hidden_size, hidden_size]
    weight_ih = Tensor::create({3 * hidden_size, input_size}, true);
    weight_hh = Tensor::create({3 * hidden_size, hidden_size}, true);
    bias_ih = Tensor::zeros({3 * hidden_size}, true);
    bias_hh = Tensor::zeros({3 * hidden_size}, true);

    for (size_t i = 0; i < weight_ih->size(); i++) weight_ih->data()[i] = dist_ih(recurrent_rng);
    for (size_t i = 0; i < weight_hh->size(); i++) weight_hh->data()[i] = dist_hh(recurrent_rng);
}

TensorPtr GRU::forward(const TensorPtr& input) {
    // Initialize hidden state to zeros
    size_t batch_size = batch_first ? input->shape[0] : input->shape[1];
    auto h0 = Tensor::zeros({batch_size, hidden_size}, false);
    return forward(input, h0);
}

TensorPtr GRU::forward(const TensorPtr& input, const TensorPtr& h0) {
    // Input: [batch, seq, input_size] if batch_first else [seq, batch, input_size]
    // Output: [batch, seq, hidden_size] if batch_first else [seq, batch, hidden_size]
    assert(input->shape.size() == 3);

    size_t batch_size, seq_len;
    if (batch_first) {
        batch_size = input->shape[0];
        seq_len = input->shape[1];
        assert(input->shape[2] == input_size);
    } else {
        seq_len = input->shape[0];
        batch_size = input->shape[1];
        assert(input->shape[2] == input_size);
    }

    bool track = input->requires_grad && GradMode::is_enabled();

    // Output tensor for all hidden states
    std::vector<size_t> out_shape;
    if (batch_first) {
        out_shape = {batch_size, seq_len, hidden_size};
    } else {
        out_shape = {seq_len, batch_size, hidden_size};
    }
    auto output = Tensor::create(out_shape, track);

    // Store intermediate values for backward pass
    std::vector<std::vector<float>> all_r(seq_len), all_z(seq_len), all_n(seq_len);
    std::vector<std::vector<float>> all_h(seq_len + 1);
    std::vector<std::vector<float>> all_hh_n(seq_len);  // W_hn @ h_{t-1} + b_hn before reset gate

    // Initialize h from h0
    all_h[0].resize(batch_size * hidden_size);
    for (size_t i = 0; i < batch_size * hidden_size; i++) {
        all_h[0][i] = h0->data()[i];
    }

    // Forward through time
    for (size_t t = 0; t < seq_len; t++) {
        all_r[t].resize(batch_size * hidden_size);
        all_z[t].resize(batch_size * hidden_size);
        all_n[t].resize(batch_size * hidden_size);
        all_h[t + 1].resize(batch_size * hidden_size);
        all_hh_n[t].resize(batch_size * hidden_size);

        for (size_t b = 0; b < batch_size; b++) {
            // Compute gates for each batch element
            // r = sigmoid(x @ W_ir^T + b_ir + h @ W_hr^T + b_hr)
            // z = sigmoid(x @ W_iz^T + b_iz + h @ W_hz^T + b_hz)
            // n = tanh(x @ W_in^T + b_in + r * (h @ W_hn^T + b_hn))
            // h_new = (1 - z) * n + z * h

            std::vector<float> gates_ih(3 * hidden_size, 0.0f);
            std::vector<float> gates_hh(3 * hidden_size, 0.0f);

            size_t x_offset = batch_first ?
                (b * seq_len * input_size + t * input_size) :
                (t * batch_size * input_size + b * input_size);

            // x @ W_ih^T + b_ih
            for (size_t g = 0; g < 3 * hidden_size; g++) {
                for (size_t i = 0; i < input_size; i++) {
                    gates_ih[g] += input->data()[x_offset + i] * weight_ih->data()[g * input_size + i];
                }
                gates_ih[g] += bias_ih->data()[g];
            }

            // h @ W_hh^T + b_hh
            for (size_t g = 0; g < 3 * hidden_size; g++) {
                for (size_t h = 0; h < hidden_size; h++) {
                    gates_hh[g] += all_h[t][b * hidden_size + h] * weight_hh->data()[g * hidden_size + h];
                }
                gates_hh[g] += bias_hh->data()[g];
            }

            // Apply gates
            for (size_t h = 0; h < hidden_size; h++) {
                size_t idx = b * hidden_size + h;

                // r = sigmoid(gates_ih[r] + gates_hh[r])
                float r_gate = 1.0f / (1.0f + std::exp(-(gates_ih[h] + gates_hh[h])));

                // z = sigmoid(gates_ih[z] + gates_hh[z])
                float z_gate = 1.0f / (1.0f + std::exp(-(gates_ih[hidden_size + h] + gates_hh[hidden_size + h])));

                // Store hh_n for backward pass (before reset gate multiplication)
                all_hh_n[t][idx] = gates_hh[2 * hidden_size + h];

                // n = tanh(gates_ih[n] + r * gates_hh[n])
                float n_gate = std::tanh(gates_ih[2 * hidden_size + h] + r_gate * gates_hh[2 * hidden_size + h]);

                // h_new = (1 - z) * n + z * h_prev
                float h_new = (1.0f - z_gate) * n_gate + z_gate * all_h[t][idx];

                all_r[t][idx] = r_gate;
                all_z[t][idx] = z_gate;
                all_n[t][idx] = n_gate;
                all_h[t + 1][idx] = h_new;

                // Store in output
                size_t out_offset = batch_first ?
                    (b * seq_len * hidden_size + t * hidden_size + h) :
                    (t * batch_size * hidden_size + b * hidden_size + h);
                output->data()[out_offset] = h_new;
            }
        }
    }

    // Store final hidden state
    h_n = Tensor::create({batch_size, hidden_size}, false);
    for (size_t i = 0; i < batch_size * hidden_size; i++) {
        h_n->data()[i] = all_h[seq_len][i];
    }

    if (track) {
        auto input_ptr = input;
        auto weight_ih_ptr = weight_ih;
        auto weight_hh_ptr = weight_hh;
        auto bias_ih_ptr = bias_ih;
        auto bias_hh_ptr = bias_hh;
        size_t hs = hidden_size;
        size_t is = input_size;
        bool bf = batch_first;

        output->parents = {input_ptr, weight_ih_ptr, weight_hh_ptr, bias_ih_ptr, bias_hh_ptr};
        output->grad_fn = [=]() mutable {
            // Backpropagation through time (BPTT)
            std::vector<float> dh_next(batch_size * hs, 0.0f);

            for (int t = static_cast<int>(seq_len) - 1; t >= 0; t--) {
                for (size_t b = 0; b < batch_size; b++) {
                    for (size_t h = 0; h < hs; h++) {
                        size_t idx = b * hs + h;
                        size_t out_offset = bf ?
                            (b * seq_len * hs + t * hs + h) :
                            (t * batch_size * hs + b * hs + h);

                        // Gradient from output and from next time step
                        float dh = output->grad()[out_offset] + dh_next[idx];

                        // h_t = (1 - z_t) * n_t + z_t * h_{t-1}
                        float dz = dh * (all_h[t][idx] - all_n[t][idx]);
                        float dn = dh * (1.0f - all_z[t][idx]);
                        dh_next[idx] = dh * all_z[t][idx];

                        // n_t = tanh(...), so dn_pre = dn * (1 - n^2)
                        float dn_pre = dn * (1.0f - all_n[t][idx] * all_n[t][idx]);

                        // n = tanh(x @ W_in^T + b_in + r * (h @ W_hn^T + b_hn))
                        float dr_from_n = dn_pre * all_hh_n[t][idx];

                        // r_t = sigmoid(...), so dr_pre = dr * r * (1 - r)
                        float dr = dr_from_n;
                        float dr_pre = dr * all_r[t][idx] * (1.0f - all_r[t][idx]);

                        // z_t = sigmoid(...), so dz_pre = dz * z * (1 - z)
                        float dz_pre = dz * all_z[t][idx] * (1.0f - all_z[t][idx]);

                        size_t x_offset = bf ?
                            (b * seq_len * is + t * is) :
                            (t * batch_size * is + b * is);

                        // Gradients for biases
                        bias_ih_ptr->grad()[h] += dr_pre;
                        bias_ih_ptr->grad()[hs + h] += dz_pre;
                        bias_ih_ptr->grad()[2 * hs + h] += dn_pre;

                        bias_hh_ptr->grad()[h] += dr_pre;
                        bias_hh_ptr->grad()[hs + h] += dz_pre;
                        bias_hh_ptr->grad()[2 * hs + h] += dn_pre * all_r[t][idx];

                        // Gradients for weight_ih
                        for (size_t i = 0; i < is; i++) {
                            weight_ih_ptr->grad()[h * is + i] += dr_pre * input_ptr->data()[x_offset + i];
                            weight_ih_ptr->grad()[(hs + h) * is + i] += dz_pre * input_ptr->data()[x_offset + i];
                            weight_ih_ptr->grad()[(2 * hs + h) * is + i] += dn_pre * input_ptr->data()[x_offset + i];
                        }

                        // Gradients for weight_hh
                        for (size_t hh = 0; hh < hs; hh++) {
                            weight_hh_ptr->grad()[h * hs + hh] += dr_pre * all_h[t][b * hs + hh];
                            weight_hh_ptr->grad()[(hs + h) * hs + hh] += dz_pre * all_h[t][b * hs + hh];
                            weight_hh_ptr->grad()[(2 * hs + h) * hs + hh] += dn_pre * all_r[t][idx] * all_h[t][b * hs + hh];
                        }

                        // Gradient for input
                        if (input_ptr->requires_grad) {
                            for (size_t i = 0; i < is; i++) {
                                input_ptr->grad()[x_offset + i] +=
                                    dr_pre * weight_ih_ptr->data()[h * is + i] +
                                    dz_pre * weight_ih_ptr->data()[(hs + h) * is + i] +
                                    dn_pre * weight_ih_ptr->data()[(2 * hs + h) * is + i];
                            }
                        }

                        // Gradient for h_{t-1} (accumulate for next iteration)
                        for (size_t hh = 0; hh < hs; hh++) {
                            dh_next[b * hs + hh] +=
                                dr_pre * weight_hh_ptr->data()[h * hs + hh] +
                                dz_pre * weight_hh_ptr->data()[(hs + h) * hs + hh] +
                                dn_pre * all_r[t][idx] * weight_hh_ptr->data()[(2 * hs + h) * hs + hh];
                        }
                    }
                }
            }
        };
    }

    return output;
}

std::vector<TensorPtr> GRU::parameters() {
    return {weight_ih, weight_hh, bias_ih, bias_hh};
}

std::string LSTM::extra_repr() const {
    return std::to_string(input_size) + ", " + std::to_string(hidden_size) +
           ", batch_first=" + (batch_first ? "true" : "false");
}

std::string GRU::extra_repr() const {
    return std::to_string(input_size) + ", " + std::to_string(hidden_size) +
           ", batch_first=" + (batch_first ? "true" : "false");
}

std::vector<size_t> LSTM::compute_output_shape(const std::vector<size_t>& input_shape) const {
    // [N, seq, input_size] -> [N, seq, hidden_size] (batch_first)
    // [seq, N, input_size] -> [seq, N, hidden_size] (!batch_first)
    if (input_shape.size() != 3) return input_shape;
    std::vector<size_t> output_shape = input_shape;
    output_shape[2] = hidden_size;  // last dim becomes hidden_size
    return output_shape;
}

std::vector<size_t> GRU::compute_output_shape(const std::vector<size_t>& input_shape) const {
    // [N, seq, input_size] -> [N, seq, hidden_size] (batch_first)
    if (input_shape.size() != 3) return input_shape;
    std::vector<size_t> output_shape = input_shape;
    output_shape[2] = hidden_size;
    return output_shape;
}
