#pragma once

#include "algorithm/algorithm.h"
#include <memory>

namespace cann {

class AlgorithmSelector {
public:
    // Select the best algorithm for a given operation
    // TODO: implement in Task 6
    Algorithm* Select(HCCLReduceOp op, size_t count,
                      uint32_t nranks) const {
        return nullptr;
    }
};

} // namespace cann
