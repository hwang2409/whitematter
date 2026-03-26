#include "../layer.h"
#include <cassert>
#include <cstring>

KVCache::KVCache(size_t max_seq_len, size_t num_heads, size_t head_dim)
    : current_len_(0),
      max_seq_len_(max_seq_len),
      embed_dim_(num_heads * head_dim),
      batch_size_(0) {
    // Cache tensors are lazily allocated on first append (when we know batch size)
}

void KVCache::append(const TensorPtr& new_keys, const TensorPtr& new_values) {
    assert(new_keys->shape.size() == 3 && "new_keys must be [batch, new_tokens, embed_dim]");
    assert(new_values->shape.size() == 3 && "new_values must be [batch, new_tokens, embed_dim]");
    assert(new_keys->shape[2] == embed_dim_ && "Key embed_dim mismatch");
    assert(new_values->shape[2] == embed_dim_ && "Value embed_dim mismatch");

    size_t batch = new_keys->shape[0];
    size_t new_tokens = new_keys->shape[1];
    assert(current_len_ + new_tokens <= max_seq_len_ && "KV cache overflow: exceeds max_seq_len");

    // Lazy allocation on first append
    if (!key_cache_) {
        batch_size_ = batch;
        key_cache_ = Tensor::zeros({batch_size_, max_seq_len_, embed_dim_}, false);
        value_cache_ = Tensor::zeros({batch_size_, max_seq_len_, embed_dim_}, false);
    }

    assert(batch == batch_size_ && "Batch size must be consistent across appends");

    // Copy new keys and values into the cache at current_len_ position
    for (size_t b = 0; b < batch; b++) {
        const float* src_k = new_keys->data() + b * new_tokens * embed_dim_;
        const float* src_v = new_values->data() + b * new_tokens * embed_dim_;
        float* dst_k = key_cache_->data() + b * max_seq_len_ * embed_dim_ + current_len_ * embed_dim_;
        float* dst_v = value_cache_->data() + b * max_seq_len_ * embed_dim_ + current_len_ * embed_dim_;
        std::memcpy(dst_k, src_k, new_tokens * embed_dim_ * sizeof(float));
        std::memcpy(dst_v, src_v, new_tokens * embed_dim_ * sizeof(float));
    }

    current_len_ += new_tokens;
}

TensorPtr KVCache::keys() const {
    assert(key_cache_ && "KV cache is empty — call append() first");

    auto result = Tensor::create({batch_size_, current_len_, embed_dim_}, false);
    for (size_t b = 0; b < batch_size_; b++) {
        const float* src = key_cache_->data() + b * max_seq_len_ * embed_dim_;
        float* dst = result->data() + b * current_len_ * embed_dim_;
        std::memcpy(dst, src, current_len_ * embed_dim_ * sizeof(float));
    }
    return result;
}

TensorPtr KVCache::values() const {
    assert(value_cache_ && "KV cache is empty — call append() first");

    auto result = Tensor::create({batch_size_, current_len_, embed_dim_}, false);
    for (size_t b = 0; b < batch_size_; b++) {
        const float* src = value_cache_->data() + b * max_seq_len_ * embed_dim_;
        float* dst = result->data() + b * current_len_ * embed_dim_;
        std::memcpy(dst, src, current_len_ * embed_dim_ * sizeof(float));
    }
    return result;
}

size_t KVCache::length() const {
    return current_len_;
}

void KVCache::clear() {
    current_len_ = 0;
    batch_size_ = 0;
    key_cache_ = nullptr;
    value_cache_ = nullptr;
}
