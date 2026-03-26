def generate_image_training_code(
    layers_code: str,
    optimizer_code: str,
    scheduler_code: str,
    epochs: int,
    batch_size: int,
    dataset_config: dict
) -> str:

    has_scheduler = bool(scheduler_code)
    scheduler_step = "scheduler.step();" if has_scheduler else ""
    scheduler_include = scheduler_code if has_scheduler else ""

    num_classes = dataset_config.get("num_classes", 10)
    input_shape = dataset_config.get("input_shape", [3, 32, 32])

    code = f'''// Auto-generated training code
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include "serialize.h"

// Binary tensor loading
struct TensorFile {{
    std::vector<size_t> shape;
    std::vector<float> data;
}};

TensorFile load_tensor_file(const std::string& path) {{
    TensorFile result;
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0x54454E53) throw std::runtime_error("Invalid tensor file");

    uint32_t ndim;
    f.read(reinterpret_cast<char*>(&ndim), 4);

    result.shape.resize(ndim);
    for (uint32_t i = 0; i < ndim; i++) {{
        uint64_t dim;
        f.read(reinterpret_cast<char*>(&dim), 8);
        result.shape[i] = dim;
    }}

    size_t total = 1;
    for (auto d : result.shape) total *= d;
    result.data.resize(total);
    f.read(reinterpret_cast<char*>(result.data.data()), total * sizeof(float));

    return result;
}}

struct CustomDataset {{
    TensorPtr images;
    TensorPtr labels;
    size_t num_samples;
}};

CustomDataset load_dataset(const std::string& data_dir, bool train) {{
    std::string prefix = train ? "train" : "test";
    auto images_file = load_tensor_file(data_dir + "/" + prefix + "_images.bin");
    auto labels_file = load_tensor_file(data_dir + "/" + prefix + "_labels.bin");

    CustomDataset ds;
    ds.images = Tensor::create(images_file.shape, false);
    std::memcpy(ds.images->data(), images_file.data.data(), images_file.data.size() * sizeof(float));
    ds.labels = Tensor::create(labels_file.shape, false);
    std::memcpy(ds.labels->data(), labels_file.data.data(), labels_file.data.size() * sizeof(float));
    ds.num_samples = images_file.shape[0];
    return ds;
}}

class CustomDataLoader {{
public:
    CustomDataLoader(const CustomDataset& dataset, size_t batch_size, bool shuffle)
        : dataset(dataset), batch_size(batch_size), shuffle(shuffle), current_idx(0) {{
        indices.resize(dataset.num_samples);
        for (size_t i = 0; i < dataset.num_samples; i++) indices[i] = i;
        if (shuffle) reset();
    }}

    void reset() {{
        current_idx = 0;
        if (shuffle) {{
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);
        }}
    }}

    bool has_next() const {{ return current_idx < dataset.num_samples; }}

    size_t num_batches() const {{
        return (dataset.num_samples + batch_size - 1) / batch_size;
    }}

    std::pair<TensorPtr, TensorPtr> next_batch() {{
        size_t actual_batch = std::min(batch_size, dataset.num_samples - current_idx);

        // Get image shape (without batch dimension)
        std::vector<size_t> img_shape(dataset.images->shape.begin() + 1, dataset.images->shape.end());
        size_t img_size = 1;
        for (auto d : img_shape) img_size *= d;

        // Create batch tensors
        std::vector<size_t> batch_img_shape = {{actual_batch}};
        batch_img_shape.insert(batch_img_shape.end(), img_shape.begin(), img_shape.end());

        auto batch_images = Tensor::create(batch_img_shape, false);
        auto batch_labels = Tensor::create({{actual_batch}}, false);

        for (size_t i = 0; i < actual_batch; i++) {{
            size_t idx = indices[current_idx + i];
            std::copy(
                dataset.images->data() + idx * img_size,
                dataset.images->data() + (idx + 1) * img_size,
                batch_images->data() + i * img_size
            );
            batch_labels->data()[i] = dataset.labels->data()[idx];
        }}

        current_idx += actual_batch;
        return {{batch_images, batch_labels}};
    }}

private:
    const CustomDataset& dataset;
    size_t batch_size;
    bool shuffle;
    size_t current_idx;
    std::vector<size_t> indices;
}};

float compute_accuracy(Sequential& model, CustomDataLoader& loader) {{
    NoGradGuard no_grad;
    model.eval();

    size_t correct = 0, total = 0;
    loader.reset();

    while (loader.has_next()) {{
        auto [images, labels] = loader.next_batch();
        auto output = model.forward(images);

        size_t batch_size = output->shape[0];
        size_t num_classes = output->shape[1];

        for (size_t i = 0; i < batch_size; i++) {{
            size_t predicted = 0;
            float max_val = output->data()[i * num_classes];
            for (size_t j = 1; j < num_classes; j++) {{
                if (output->data()[i * num_classes + j] > max_val) {{
                    max_val = output->data()[i * num_classes + j];
                    predicted = j;
                }}
            }}
            if (predicted == static_cast<size_t>(labels->data()[i])) correct++;
            total++;
        }}
    }}

    model.train();
    return static_cast<float>(correct) / total * 100.0f;
}}

int main(int argc, char* argv[]) {{
    if (argc < 3) {{
        printf("Usage: %s <data_dir> <output_model> [resume_weights] [start_epoch]\\n", argv[0]);
        return 1;
    }}

    std::string data_dir = argv[1];
    std::string output_path = argv[2];
    std::string resume_weights = (argc > 3) ? argv[3] : "";
    int start_epoch = (argc > 4) ? std::atoi(argv[4]) : 0;

    // Disable stdout buffering for real-time output
    setbuf(stdout, NULL);

    printf("Custom Model Training\\n");
    printf("=====================\\n\\n");

    printf("Loading dataset from '%s'...\\n", data_dir.c_str());
    auto train_data = load_dataset(data_dir, true);
    auto test_data = load_dataset(data_dir, false);

    printf("Train samples: %zu\\n", train_data.num_samples);
    printf("Test samples: %zu\\n\\n", test_data.num_samples);

    // Build model
    Sequential model({{
{layers_code}
    }});

    // Load weights if resuming
    if (!resume_weights.empty()) {{
        printf("Resuming from weights: %s (epoch %d)\\n", resume_weights.c_str(), start_epoch);
        load_model(&model, resume_weights);
    }}

    CrossEntropyLoss criterion;
    {optimizer_code}
    {scheduler_include}

    CustomDataLoader train_loader(train_data, {batch_size}, true);
    CustomDataLoader test_loader(test_data, {batch_size}, false);

    // Count parameters
    size_t total_params = 0;
    for (const auto& p : model.parameters()) total_params += p->size();
    printf("Total parameters: %zu\\n\\n", total_params);

    printf("Training for {epochs} epochs (starting from %d)...\\n", start_epoch);
    printf("------------------------------------------------------------------\\n");

    float best_acc = 0.0f;

    for (int epoch = start_epoch; epoch < {epochs}; epoch++) {{
        model.train();
        train_loader.reset();
        float total_loss = 0.0f;
        size_t num_batches = 0;

        while (train_loader.has_next()) {{
            auto [images, labels] = train_loader.next_batch();

            optimizer.zero_grad();
            auto output = model.forward(images);
            auto loss = criterion(output, labels);
            loss->backward();
            optimizer.step();

            total_loss += loss->data()[0];
            num_batches++;
        }}

        {scheduler_step}

        float avg_loss = total_loss / num_batches;
        float test_acc = compute_accuracy(model, test_loader);
        best_acc = std::max(best_acc, test_acc);

        printf("Epoch %2d | Loss: %.4f | Test Acc: %.2f%% | Best: %.2f%%\\n",
               epoch + 1, avg_loss, test_acc, best_acc);

        // Save checkpoint after each epoch
        save_model(&model, output_path);
    }}

    printf("------------------------------------------------------------------\\n");
    printf("Training complete! Best accuracy: %.2f%%\\n\\n", best_acc);

    printf("Saving model to '%s'...\\n", output_path.c_str());
    save_model(&model, output_path);

    return 0;
}}
'''
    return code
