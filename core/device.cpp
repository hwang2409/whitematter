#include "device.h"

namespace whitematter {

Device Device::cpu() { return Device(DeviceType::CPU); }
Device Device::metal() { return Device(DeviceType::METAL); }
Device Device::cuda() { return Device(DeviceType::CUDA); }

Device Device::auto_() {
#if defined(__APPLE__)
    if (metal_backend_available()) return Device(DeviceType::METAL);
#endif
    if (cuda_backend_available()) return Device(DeviceType::CUDA);
    return Device(DeviceType::CPU);
}

Device Device::default_device() { return auto_(); }

bool Device::is_available() const {
    if (type_ == DeviceType::CPU) return true;
    if (type_ == DeviceType::METAL) return metal_backend_available();
    if (type_ == DeviceType::CUDA) return cuda_backend_available();
    return false;
}

}  // namespace whitematter
