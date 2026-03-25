#include "autograd.h"

bool GradMode::enabled_ = true;

bool GradMode::is_enabled() {
    return enabled_;
}

void GradMode::set_enabled(bool enabled) {
    enabled_ = enabled;
}

NoGradGuard::NoGradGuard() : prev_mode_(GradMode::is_enabled()) {
    GradMode::set_enabled(false);
}

NoGradGuard::~NoGradGuard() {
    GradMode::set_enabled(prev_mode_);
}
