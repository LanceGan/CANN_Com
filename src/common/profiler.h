// f:/Projects/CANN_Com/src/common/profiler.h
#pragma once

#include <chrono>
#include <string>

namespace cann {

class ScopedTimer {
public:
    explicit ScopedTimer(double* output_ms)
        : output_(output_ms),
          start_(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start_);
        *output_ = duration.count();
    }

private:
    double* output_;
    std::chrono::high_resolution_clock::time_point start_;
};

} // namespace cann
