// Stub when CUDA is not built (default or no nvcc).
#include "device.h"

namespace whitematter {

bool cuda_backend_available() { return false; }

}  // namespace whitematter
