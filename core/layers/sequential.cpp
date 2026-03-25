#include "../layer.h"
#include <cstdio>
#include <string>

size_t Module::num_parameters() const {
    size_t total = 0;
    auto* self = const_cast<Module*>(this);
    for (const auto& p : self->parameters()) {
        total += p->size();
    }
    return total;
}

size_t Module::num_trainable_parameters() const {
    size_t total = 0;
    auto* self = const_cast<Module*>(this);
    for (const auto& p : self->parameters()) {
        if (p->requires_grad) {
            total += p->size();
        }
    }
    return total;
}

Sequential::Sequential(std::initializer_list<Module*> modules) {
    for (Module* m : modules) {
        layers.push_back(std::shared_ptr<Module>(m));
    }
}

void Sequential::add(Module* module) {
    layers.push_back(std::shared_ptr<Module>(module));
}

void Sequential::add(std::shared_ptr<Module> module) {
    layers.push_back(module);
}

TensorPtr Sequential::forward(const TensorPtr& input) {
    TensorPtr x = input;
    for (const auto& layer : layers) {
        x = layer->forward(x);
    }
    return x;
}

std::vector<TensorPtr> Sequential::parameters() {
    std::vector<TensorPtr> params;
    for (const auto& layer : layers) {
        auto layer_params = layer->parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}

void Sequential::train() {
    for (const auto& layer : layers) {
        Module* m = layer.get();
        if (auto dropout = dynamic_cast<Dropout*>(m)) {
            dropout->train();
        } else if (auto bn = dynamic_cast<BatchNorm2d*>(m)) {
            bn->train();
        }
    }
}

void Sequential::eval() {
    for (const auto& layer : layers) {
        Module* m = layer.get();
        if (auto dropout = dynamic_cast<Dropout*>(m)) {
            dropout->eval();
        } else if (auto bn = dynamic_cast<BatchNorm2d*>(m)) {
            bn->eval();
        }
    }
}

static std::string format_shape(const std::vector<size_t>& shape) {
    if (shape.empty()) return "-";
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) s += ", ";
        s += std::to_string(shape[i]);
    }
    s += "]";
    return s;
}

void Sequential::summary(const std::vector<size_t>& input_shape) const {
    printf("==============================================================================\n");
    printf("Layer (type)                    Output Shape              Param #\n");
    printf("==============================================================================\n");

    size_t total_params = 0;
    size_t trainable_params = 0;

    std::vector<size_t> current_shape = input_shape;
    bool tracking_shapes = !input_shape.empty();

    for (size_t i = 0; i < layers.size(); i++) {
        Module* layer = layers[i].get();
        size_t layer_params = layer->num_parameters();
        size_t layer_trainable = layer->num_trainable_parameters();

        total_params += layer_params;
        trainable_params += layer_trainable;

        std::string shape_str = "-";
        if (tracking_shapes) {
            current_shape = layer->compute_output_shape(current_shape);
            shape_str = format_shape(current_shape);
        }

        std::string layer_name = layer->name();
        std::string extra = layer->extra_repr();
        std::string full_name = layer_name;
        if (!extra.empty()) {
            full_name += "(" + extra + ")";
        }

        if (full_name.length() > 32) {
            full_name = full_name.substr(0, 29) + "...";
        }

        std::string param_str = std::to_string(layer_params);
        int insert_pos = static_cast<int>(param_str.length()) - 3;
        while (insert_pos > 0) {
            param_str.insert(insert_pos, ",");
            insert_pos -= 3;
        }

        printf("%-32s %-25s %s\n", full_name.c_str(), shape_str.c_str(), param_str.c_str());
    }

    printf("==============================================================================\n");

    auto format_with_commas = [](size_t n) -> std::string {
        std::string s = std::to_string(n);
        int pos = static_cast<int>(s.length()) - 3;
        while (pos > 0) {
            s.insert(pos, ",");
            pos -= 3;
        }
        return s;
    };

    printf("Total params: %s\n", format_with_commas(total_params).c_str());
    printf("Trainable params: %s\n", format_with_commas(trainable_params).c_str());
    printf("Non-trainable params: %s\n", format_with_commas(total_params - trainable_params).c_str());
    printf("==============================================================================\n");
}

std::string format_number(size_t n) {
    std::string s = std::to_string(n);
    int pos = static_cast<int>(s.length()) - 3;
    while (pos > 0) {
        s.insert(pos, ",");
        pos -= 3;
    }
    return s;
}

std::string format_memory(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_idx = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit_idx < 4) {
        size /= 1024.0;
        unit_idx++;
    }

    char buf[32];
    if (unit_idx == 0) {
        snprintf(buf, sizeof(buf), "%zu %s", bytes, units[unit_idx]);
    } else {
        snprintf(buf, sizeof(buf), "%.2f %s", size, units[unit_idx]);
    }
    return std::string(buf);
}

ModelSummary get_model_summary(Module* model) {
    ModelSummary info;

    info.total_params = model->num_parameters();
    info.trainable_params = model->num_trainable_parameters();
    info.non_trainable_params = info.total_params - info.trainable_params;

    info.param_memory_bytes = info.total_params * sizeof(float);
    info.param_memory_fp16_bytes = info.total_params * sizeof(uint16_t);
    info.grad_memory_bytes = info.trainable_params * sizeof(float);
    info.total_memory_bytes = info.param_memory_bytes + info.grad_memory_bytes;

    info.num_layers = 0;
    Sequential* seq = dynamic_cast<Sequential*>(model);
    if (seq) {
        info.num_layers = seq->layers.size();
    }

    return info;
}

void print_model_info(Module* model, const std::string& name) {
    ModelSummary info = get_model_summary(model);

    printf("==============================================================================\n");
    printf("%s Summary\n", name.c_str());
    printf("==============================================================================\n");
    printf("Total parameters:       %s\n", format_number(info.total_params).c_str());
    printf("Trainable parameters:   %s\n", format_number(info.trainable_params).c_str());
    printf("Non-trainable params:   %s\n", format_number(info.non_trainable_params).c_str());
    printf("------------------------------------------------------------------------------\n");
    printf("Parameter memory (fp32): %s\n", format_memory(info.param_memory_bytes).c_str());
    printf("Parameter memory (fp16): %s\n", format_memory(info.param_memory_fp16_bytes).c_str());
    printf("Gradient memory:         %s\n", format_memory(info.grad_memory_bytes).c_str());
    printf("Total training memory:   %s\n", format_memory(info.total_memory_bytes).c_str());
    if (info.num_layers > 0) {
        printf("------------------------------------------------------------------------------\n");
        printf("Number of layers:        %zu\n", info.num_layers);
    }
    printf("==============================================================================\n");
}
