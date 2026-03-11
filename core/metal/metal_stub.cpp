// Stub when Metal is not built (Linux/Windows or METAL=0).
#include "../device.h"

namespace whitematter {

bool metal_backend_available() { return false; }

}  // namespace whitematter
