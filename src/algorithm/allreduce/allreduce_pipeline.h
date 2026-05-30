#pragma once

#include "algorithm/algorithm.h"

namespace cann {

class AllReducePipeline : public Algorithm {
public:
    explicit AllReducePipeline(int pipeline_depth = 4)
        : pipeline_depth_(pipeline_depth) {}

    Status Execute(void* sendbuf, void* recvbuf, size_t count,
                   HCCLDataType dtype, HCCLReduceOp op,
                   CommContext& ctx) override;

    const char* Name() const override { return "AllReducePipeline"; }

    int NumSteps(uint32_t nranks) const override {
        return 2 * static_cast<int>(nranks - 1);
    }

private:
    int pipeline_depth_;
};

} // namespace cann
