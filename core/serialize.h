#ifndef SERIALIZE_H
#define SERIALIZE_H

#include "tensor.h"
#include "layer.h"
#include "optimizer.h"
#include <string>
#include <fstream>

// File format magic numbers for validation
constexpr uint32_t TENSOR_MAGIC = 0x574D5400;  // "WMT\0" - WhiteMatter Tensor
constexpr uint32_t MODEL_MAGIC  = 0x574D4D00;  // "WMM\0" - WhiteMatter Model
constexpr uint32_t OPTIM_MAGIC  = 0x574D4F00;  // "WMO\0" - WhiteMatter Optimizer

// Tensor serialization
bool save_tensor(const TensorPtr& tensor, const std::string& path);
bool save_tensor(const TensorPtr& tensor, std::ofstream& out);
TensorPtr load_tensor(const std::string& path);
TensorPtr load_tensor(std::ifstream& in);

// Model serialization (saves/loads all parameters)
bool save_model(Module* module, const std::string& path);
bool load_model(Module* module, const std::string& path);

// Optimizer state serialization (for training checkpoints)
bool save_optimizer(Optimizer* optimizer, const std::string& path);
bool load_optimizer(Optimizer* optimizer, const std::string& path);

// Checkpoint: saves model + optimizer state together
struct Checkpoint {
    int epoch;
    float loss;
    float accuracy;
};

bool save_checkpoint(const std::string& path, Module* model, Optimizer* optimizer,
                     int epoch, float loss = 0.0f, float accuracy = 0.0f);
bool load_checkpoint(const std::string& path, Module* model, Optimizer* optimizer,
                     Checkpoint& info);

#endif
