// f:/Projects/CANN_Com/src/algorithm/allreduce/allreduce_rhd.h
#pragma once

#include "algorithm/algorithm.h"

namespace cann {

class AllReduceRHD : public Algorithm {
public:
    Status Execute(void* sendbuf, void* recvbuf, size_t count,
                   HCCLDataType dtype, HCCLReduceOp op,
                   CommContext& ctx) override;

    const char* Name() const override { return "AllReduceRHD"; }

    int NumSteps(uint32_t nranks) const override {
        if (nranks <= 1) return 0;
        int logN = 0;
        uint32_t n = nranks;
        while (n > 1) { n >>= 1; logN++; }
        return 2 * logN;
    }
};

} // namespace cann
