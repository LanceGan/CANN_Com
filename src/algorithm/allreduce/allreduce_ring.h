#pragma once

#include "algorithm/algorithm.h"

namespace cann {

class AllReduceRing : public Algorithm {
public:
    Status Execute(void* sendbuf, void* recvbuf, size_t count,
                   HCCLDataType dtype, HCCLReduceOp op,
                   CommContext& ctx) override;

    const char* Name() const override { return "AllReduceRing"; }

    int NumSteps(uint32_t nranks) const override {
        return 2 * static_cast<int>(nranks - 1);
    }
};

} // namespace cann
