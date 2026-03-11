#ifndef DEVICE_H
#define DEVICE_H

namespace whitematter {

enum class DeviceType { CPU, METAL, CUDA };

class Device {
public:
    static Device cpu();
    static Device metal();
    static Device cuda();
    static Device default_device();

    DeviceType type() const { return type_; }
    bool is_available() const;

private:
    explicit Device(DeviceType t) : type_(t) {}
    DeviceType type_;
};

// Implemented in metal_stub.cpp (false) or metal_backend.mm (runtime check) depending on build.
bool metal_backend_available();

// Implemented in cuda_stub.cpp (false) or cuda_backend (runtime check) depending on build.
bool cuda_backend_available();

}  // namespace whitematter

#endif
