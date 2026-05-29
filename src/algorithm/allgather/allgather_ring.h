// f:/Projects/CANN_Com/src/algorithm/allgather/allgather_ring.h
#pragma once

#include "algorithm/algorithm.h"

namespace cann {

class AllGatherRing : public Algorithm {
public:
    Status Execute(void* sendbuf, void* recvbuf, size_t count,
                   HCCLDataType dtype, HCCLReduceOp op,
                   CommContext& ctx) override;

    const char* Name() const override { return "AllGatherRing"; }

    int NumSteps(uint32_t nranks) const override {
        return static_cast<int>(nranks - 1);
    }
};

} // namespace cann
