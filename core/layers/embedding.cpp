#include "../layer.h"
#include <cassert>

Embedding::Embedding(size_t num_embeddings, size_t embedding_dim)
    : num_embeddings(num_embeddings), embedding_dim(embedding_dim) {
    weight = Tensor::randn({num_embeddings, embedding_dim}, true);
}

TensorPtr Embedding::forward(const TensorPtr& indices) {
    std::vector<size_t> out_shape;
    for (size_t dim : indices->shape) {
        out_shape.push_back(dim);
    }
    out_shape.push_back(embedding_dim);

    size_t num_indices = indices->size();
    bool track = weight->requires_grad && GradMode::is_enabled();
    auto result = Tensor::create(out_shape, track);

    for (size_t i = 0; i < num_indices; i++) {
        size_t idx = static_cast<size_t>(indices->data()[i]);
        assert(idx < num_embeddings);
        for (size_t j = 0; j < embedding_dim; j++) {
            result->data()[i * embedding_dim + j] = weight->data()[idx * embedding_dim + j];
        }
    }

    if (track) {
        auto weight_ptr = weight;
        auto indices_ptr = indices;
        result->parents = {weight_ptr};
        result->grad_fn = [weight_ptr, indices_ptr, result, num_indices, this]() {
            for (size_t i = 0; i < num_indices; i++) {
                size_t idx = static_cast<size_t>(indices_ptr->data()[i]);
                for (size_t j = 0; j < embedding_dim; j++) {
                    weight_ptr->grad()[idx * embedding_dim + j] += result->grad()[i * embedding_dim + j];
                }
            }
        };
    }

    return result;
}

std::vector<TensorPtr> Embedding::parameters() {
    return {weight};
}

std::string Embedding::extra_repr() const {
    return std::to_string(num_embeddings) + ", " + std::to_string(embedding_dim);
}

std::vector<size_t> Embedding::compute_output_shape(const std::vector<size_t>& input_shape) const {
    std::vector<size_t> output_shape = input_shape;
    output_shape.push_back(embedding_dim);
    return output_shape;
}
