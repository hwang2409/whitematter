#include "device.h"

namespace whitematter {

Device Device::cpu() { return Device(DeviceType::CPU); }
Device Device::metal() { return Device(DeviceType::METAL); }
Device Device::default_device() { return cpu(); }

bool Device::is_available() const {
    if (type_ == DeviceType::CPU) return true;
    if (type_ == DeviceType::METAL) return metal_backend_available();
    return false;
}

}  // namespace whitematter
