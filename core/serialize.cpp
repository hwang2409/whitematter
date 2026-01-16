#include "serialize.h"
#include <cstdio>

// Helper functions for binary I/O
static bool write_uint32(std::ofstream& out, uint32_t val) {
    out.write(reinterpret_cast<const char*>(&val), sizeof(val));
    return out.good();
}

static bool read_uint32(std::ifstream& in, uint32_t& val) {
    in.read(reinterpret_cast<char*>(&val), sizeof(val));
    return in.good();
}

static bool write_float(std::ofstream& out, float val) {
    out.write(reinterpret_cast<const char*>(&val), sizeof(val));
    return out.good();
}

static bool read_float(std::ifstream& in, float& val) {
    in.read(reinterpret_cast<char*>(&val), sizeof(val));
    return in.good();
}

static bool write_int(std::ofstream& out, int val) {
    out.write(reinterpret_cast<const char*>(&val), sizeof(val));
    return out.good();
}

static bool read_int(std::ifstream& in, int& val) {
    in.read(reinterpret_cast<char*>(&val), sizeof(val));
    return in.good();
}

// Tensor serialization
bool save_tensor(const TensorPtr& tensor, std::ofstream& out) {
    if (!tensor || !out.good()) return false;

    // Write number of dimensions
    uint32_t ndim = static_cast<uint32_t>(tensor->shape.size());
    if (!write_uint32(out, ndim)) return false;

    // Write shape
    for (size_t dim : tensor->shape) {
        if (!write_uint32(out, static_cast<uint32_t>(dim))) return false;
    }

    // Write data
    size_t size = tensor->size();
    out.write(reinterpret_cast<const char*>(tensor->data.data()), size * sizeof(float));

    return out.good();
}

bool save_tensor(const TensorPtr& tensor, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        fprintf(stderr, "Error: Cannot open file for writing: %s\n", path.c_str());
        return false;
    }

    // Write magic number
    if (!write_uint32(out, TENSOR_MAGIC)) return false;

    return save_tensor(tensor, out);
}

TensorPtr load_tensor(std::ifstream& in) {
    if (!in.good()) return nullptr;

    // Read number of dimensions
    uint32_t ndim;
    if (!read_uint32(in, ndim)) return nullptr;

    // Read shape
    std::vector<size_t> shape(ndim);
    size_t total_size = 1;
    for (uint32_t i = 0; i < ndim; i++) {
        uint32_t dim;
        if (!read_uint32(in, dim)) return nullptr;
        shape[i] = dim;
        total_size *= dim;
    }

    // Read data
    std::vector<float> data(total_size);
    in.read(reinterpret_cast<char*>(data.data()), total_size * sizeof(float));
    if (!in.good()) return nullptr;

    return Tensor::create(data, shape, true);
}

TensorPtr load_tensor(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        fprintf(stderr, "Error: Cannot open file for reading: %s\n", path.c_str());
        return nullptr;
    }

    // Verify magic number
    uint32_t magic;
    if (!read_uint32(in, magic) || magic != TENSOR_MAGIC) {
        fprintf(stderr, "Error: Invalid tensor file format: %s\n", path.c_str());
        return nullptr;
    }

    return load_tensor(in);
}

// Model serialization
bool save_model(Module* module, const std::string& path) {
    if (!module) return false;

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        fprintf(stderr, "Error: Cannot open file for writing: %s\n", path.c_str());
        return false;
    }

    // Write magic number
    if (!write_uint32(out, MODEL_MAGIC)) return false;

    // Get all parameters
    auto params = module->parameters();

    // Write number of parameters
    if (!write_uint32(out, static_cast<uint32_t>(params.size()))) return false;

    // Write each parameter tensor
    for (const auto& param : params) {
        if (!save_tensor(param, out)) return false;
    }

    printf("Model saved: %zu parameters to %s\n", params.size(), path.c_str());
    return true;
}

bool load_model(Module* module, const std::string& path) {
    if (!module) return false;

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        fprintf(stderr, "Error: Cannot open file for reading: %s\n", path.c_str());
        return false;
    }

    // Verify magic number
    uint32_t magic;
    if (!read_uint32(in, magic) || magic != MODEL_MAGIC) {
        fprintf(stderr, "Error: Invalid model file format: %s\n", path.c_str());
        return false;
    }

    // Read number of parameters
    uint32_t num_params;
    if (!read_uint32(in, num_params)) return false;

    // Get model's parameters
    auto params = module->parameters();
    if (params.size() != num_params) {
        fprintf(stderr, "Error: Parameter count mismatch. File has %u, model has %zu\n",
                num_params, params.size());
        return false;
    }

    // Load each parameter tensor
    for (size_t i = 0; i < num_params; i++) {
        TensorPtr loaded = load_tensor(in);
        if (!loaded) return false;

        // Verify shape matches
        if (loaded->shape != params[i]->shape) {
            fprintf(stderr, "Error: Shape mismatch for parameter %zu\n", i);
            return false;
        }

        // Copy data into existing parameter
        params[i]->data = std::move(loaded->data);
    }

    printf("Model loaded: %u parameters from %s\n", num_params, path.c_str());
    return true;
}

// Optimizer state serialization
bool save_optimizer(Optimizer* optimizer, const std::string& path) {
    if (!optimizer) return false;

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        fprintf(stderr, "Error: Cannot open file for writing: %s\n", path.c_str());
        return false;
    }

    // Write magic number
    if (!write_uint32(out, OPTIM_MAGIC)) return false;

    // Try to cast to specific optimizer types
    if (auto* sgd = dynamic_cast<SGD*>(optimizer)) {
        // Type identifier: 1 = SGD
        if (!write_uint32(out, 1)) return false;
        if (!write_float(out, sgd->lr)) return false;
        if (!write_float(out, sgd->momentum)) return false;

        // Write velocity buffers
        if (!write_uint32(out, static_cast<uint32_t>(sgd->velocity.size()))) return false;
        for (const auto& vel : sgd->velocity) {
            if (!write_uint32(out, static_cast<uint32_t>(vel.size()))) return false;
            out.write(reinterpret_cast<const char*>(vel.data()), vel.size() * sizeof(float));
        }
    } else if (auto* adam = dynamic_cast<Adam*>(optimizer)) {
        // Type identifier: 2 = Adam
        if (!write_uint32(out, 2)) return false;
        if (!write_float(out, adam->lr)) return false;
        if (!write_float(out, adam->beta1)) return false;
        if (!write_float(out, adam->beta2)) return false;
        if (!write_float(out, adam->eps)) return false;
        if (!write_int(out, adam->t)) return false;

        // Write m buffers
        if (!write_uint32(out, static_cast<uint32_t>(adam->m.size()))) return false;
        for (const auto& m_buf : adam->m) {
            if (!write_uint32(out, static_cast<uint32_t>(m_buf.size()))) return false;
            out.write(reinterpret_cast<const char*>(m_buf.data()), m_buf.size() * sizeof(float));
        }

        // Write v buffers
        for (const auto& v_buf : adam->v) {
            if (!write_uint32(out, static_cast<uint32_t>(v_buf.size()))) return false;
            out.write(reinterpret_cast<const char*>(v_buf.data()), v_buf.size() * sizeof(float));
        }
    } else {
        fprintf(stderr, "Error: Unknown optimizer type\n");
        return false;
    }

    printf("Optimizer state saved to %s\n", path.c_str());
    return out.good();
}

bool load_optimizer(Optimizer* optimizer, const std::string& path) {
    if (!optimizer) return false;

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        fprintf(stderr, "Error: Cannot open file for reading: %s\n", path.c_str());
        return false;
    }

    // Verify magic number
    uint32_t magic;
    if (!read_uint32(in, magic) || magic != OPTIM_MAGIC) {
        fprintf(stderr, "Error: Invalid optimizer file format: %s\n", path.c_str());
        return false;
    }

    // Read optimizer type
    uint32_t opt_type;
    if (!read_uint32(in, opt_type)) return false;

    if (opt_type == 1) {
        auto* sgd = dynamic_cast<SGD*>(optimizer);
        if (!sgd) {
            fprintf(stderr, "Error: File contains SGD state but optimizer is not SGD\n");
            return false;
        }

        if (!read_float(in, sgd->lr)) return false;
        if (!read_float(in, sgd->momentum)) return false;

        uint32_t num_vel;
        if (!read_uint32(in, num_vel)) return false;

        if (num_vel != sgd->velocity.size()) {
            fprintf(stderr, "Error: Velocity buffer count mismatch\n");
            return false;
        }

        for (auto& vel : sgd->velocity) {
            uint32_t vel_size;
            if (!read_uint32(in, vel_size)) return false;
            if (vel_size != vel.size()) {
                fprintf(stderr, "Error: Velocity buffer size mismatch\n");
                return false;
            }
            in.read(reinterpret_cast<char*>(vel.data()), vel.size() * sizeof(float));
        }
    } else if (opt_type == 2) {
        auto* adam = dynamic_cast<Adam*>(optimizer);
        if (!adam) {
            fprintf(stderr, "Error: File contains Adam state but optimizer is not Adam\n");
            return false;
        }

        if (!read_float(in, adam->lr)) return false;
        if (!read_float(in, adam->beta1)) return false;
        if (!read_float(in, adam->beta2)) return false;
        if (!read_float(in, adam->eps)) return false;
        if (!read_int(in, adam->t)) return false;

        uint32_t num_bufs;
        if (!read_uint32(in, num_bufs)) return false;

        if (num_bufs != adam->m.size()) {
            fprintf(stderr, "Error: Adam buffer count mismatch\n");
            return false;
        }

        // Read m buffers
        for (auto& m_buf : adam->m) {
            uint32_t buf_size;
            if (!read_uint32(in, buf_size)) return false;
            if (buf_size != m_buf.size()) {
                fprintf(stderr, "Error: Adam m buffer size mismatch\n");
                return false;
            }
            in.read(reinterpret_cast<char*>(m_buf.data()), m_buf.size() * sizeof(float));
        }

        // Read v buffers
        for (auto& v_buf : adam->v) {
            uint32_t buf_size;
            if (!read_uint32(in, buf_size)) return false;
            if (buf_size != v_buf.size()) {
                fprintf(stderr, "Error: Adam v buffer size mismatch\n");
                return false;
            }
            in.read(reinterpret_cast<char*>(v_buf.data()), v_buf.size() * sizeof(float));
        }
    } else {
        fprintf(stderr, "Error: Unknown optimizer type in file: %u\n", opt_type);
        return false;
    }

    printf("Optimizer state loaded from %s\n", path.c_str());
    return in.good();
}

// Checkpoint functions
bool save_checkpoint(const std::string& path, Module* model, Optimizer* optimizer,
                     int epoch, float loss, float accuracy) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        fprintf(stderr, "Error: Cannot open file for writing: %s\n", path.c_str());
        return false;
    }

    // Write checkpoint header
    if (!write_uint32(out, MODEL_MAGIC)) return false;
    if (!write_uint32(out, 1)) return false;  // Version number
    if (!write_int(out, epoch)) return false;
    if (!write_float(out, loss)) return false;
    if (!write_float(out, accuracy)) return false;

    // Write model parameters
    auto params = model->parameters();
    if (!write_uint32(out, static_cast<uint32_t>(params.size()))) return false;
    for (const auto& param : params) {
        if (!save_tensor(param, out)) return false;
    }

    // Write optimizer state
    if (optimizer) {
        if (!write_uint32(out, 1)) return false;  // Has optimizer

        if (auto* sgd = dynamic_cast<SGD*>(optimizer)) {
            if (!write_uint32(out, 1)) return false;  // SGD type
            if (!write_float(out, sgd->lr)) return false;
            if (!write_float(out, sgd->momentum)) return false;

            if (!write_uint32(out, static_cast<uint32_t>(sgd->velocity.size()))) return false;
            for (const auto& vel : sgd->velocity) {
                if (!write_uint32(out, static_cast<uint32_t>(vel.size()))) return false;
                out.write(reinterpret_cast<const char*>(vel.data()), vel.size() * sizeof(float));
            }
        } else if (auto* adam = dynamic_cast<Adam*>(optimizer)) {
            if (!write_uint32(out, 2)) return false;  // Adam type
            if (!write_float(out, adam->lr)) return false;
            if (!write_float(out, adam->beta1)) return false;
            if (!write_float(out, adam->beta2)) return false;
            if (!write_float(out, adam->eps)) return false;
            if (!write_int(out, adam->t)) return false;

            if (!write_uint32(out, static_cast<uint32_t>(adam->m.size()))) return false;
            for (const auto& m_buf : adam->m) {
                if (!write_uint32(out, static_cast<uint32_t>(m_buf.size()))) return false;
                out.write(reinterpret_cast<const char*>(m_buf.data()), m_buf.size() * sizeof(float));
            }
            for (const auto& v_buf : adam->v) {
                if (!write_uint32(out, static_cast<uint32_t>(v_buf.size()))) return false;
                out.write(reinterpret_cast<const char*>(v_buf.data()), v_buf.size() * sizeof(float));
            }
        }
    } else {
        if (!write_uint32(out, 0)) return false;  // No optimizer
    }

    printf("Checkpoint saved: epoch=%d, loss=%.4f, accuracy=%.2f%% to %s\n",
           epoch, loss, accuracy * 100.0f, path.c_str());
    return out.good();
}

bool load_checkpoint(const std::string& path, Module* model, Optimizer* optimizer,
                     Checkpoint& info) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        fprintf(stderr, "Error: Cannot open file for reading: %s\n", path.c_str());
        return false;
    }

    // Read and verify header
    uint32_t magic, version;
    if (!read_uint32(in, magic) || magic != MODEL_MAGIC) {
        fprintf(stderr, "Error: Invalid checkpoint file format: %s\n", path.c_str());
        return false;
    }
    if (!read_uint32(in, version) || version != 1) {
        fprintf(stderr, "Error: Unsupported checkpoint version: %u\n", version);
        return false;
    }

    // Read checkpoint info
    if (!read_int(in, info.epoch)) return false;
    if (!read_float(in, info.loss)) return false;
    if (!read_float(in, info.accuracy)) return false;

    // Read model parameters
    uint32_t num_params;
    if (!read_uint32(in, num_params)) return false;

    auto params = model->parameters();
    if (params.size() != num_params) {
        fprintf(stderr, "Error: Parameter count mismatch. File has %u, model has %zu\n",
                num_params, params.size());
        return false;
    }

    for (size_t i = 0; i < num_params; i++) {
        TensorPtr loaded = load_tensor(in);
        if (!loaded) return false;

        if (loaded->shape != params[i]->shape) {
            fprintf(stderr, "Error: Shape mismatch for parameter %zu\n", i);
            return false;
        }

        params[i]->data = std::move(loaded->data);
    }

    // Read optimizer state if present
    uint32_t has_optimizer;
    if (!read_uint32(in, has_optimizer)) return false;

    if (has_optimizer && optimizer) {
        uint32_t opt_type;
        if (!read_uint32(in, opt_type)) return false;

        if (opt_type == 1) {
            auto* sgd = dynamic_cast<SGD*>(optimizer);
            if (!sgd) {
                fprintf(stderr, "Warning: Checkpoint has SGD state but optimizer is not SGD, skipping\n");
            } else {
                if (!read_float(in, sgd->lr)) return false;
                if (!read_float(in, sgd->momentum)) return false;

                uint32_t num_vel;
                if (!read_uint32(in, num_vel)) return false;

                for (auto& vel : sgd->velocity) {
                    uint32_t vel_size;
                    if (!read_uint32(in, vel_size)) return false;
                    in.read(reinterpret_cast<char*>(vel.data()), vel.size() * sizeof(float));
                }
            }
        } else if (opt_type == 2) {
            auto* adam = dynamic_cast<Adam*>(optimizer);
            if (!adam) {
                fprintf(stderr, "Warning: Checkpoint has Adam state but optimizer is not Adam, skipping\n");
            } else {
                if (!read_float(in, adam->lr)) return false;
                if (!read_float(in, adam->beta1)) return false;
                if (!read_float(in, adam->beta2)) return false;
                if (!read_float(in, adam->eps)) return false;
                if (!read_int(in, adam->t)) return false;

                uint32_t num_bufs;
                if (!read_uint32(in, num_bufs)) return false;

                for (auto& m_buf : adam->m) {
                    uint32_t buf_size;
                    if (!read_uint32(in, buf_size)) return false;
                    in.read(reinterpret_cast<char*>(m_buf.data()), m_buf.size() * sizeof(float));
                }
                for (auto& v_buf : adam->v) {
                    uint32_t buf_size;
                    if (!read_uint32(in, buf_size)) return false;
                    in.read(reinterpret_cast<char*>(v_buf.data()), v_buf.size() * sizeof(float));
                }
            }
        }
    }

    printf("Checkpoint loaded: epoch=%d, loss=%.4f, accuracy=%.2f%% from %s\n",
           info.epoch, info.loss, info.accuracy * 100.0f, path.c_str());
    return in.good();
}
