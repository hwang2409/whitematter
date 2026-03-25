#ifndef AUTOGRAD_H
#define AUTOGRAD_H

class GradMode {
public:
    static bool is_enabled();
    static void set_enabled(bool enabled);
private:
    static bool enabled_;
};

class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();
private:
    bool prev_mode_;
};

#endif
