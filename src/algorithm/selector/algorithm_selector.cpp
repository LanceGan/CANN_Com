// f:/Projects/CANN_Com/src/algorithm/selector/algorithm_selector.cpp
#include "algorithm/selector/algorithm_selector.h"

namespace cann {

AlgorithmSelector::AlgorithmSelector() = default;

Algorithm* AlgorithmSelector::Select(PrimitiveType prim, size_t bytes,
                                      uint32_t nranks) {
    switch (prim) {
        case PrimitiveType::ALL_REDUCE:
            return &allreduce_ring_;
        case PrimitiveType::ALL_GATHER:
            return &allgather_ring_;
        case PrimitiveType::REDUCE_SCATTER:
            return &reduce_scatter_ring_;
        case PrimitiveType::ALL_TO_ALL:
            return &alltoall_direct_;
        default:
            return nullptr;
    }
}

std::vector<std::string> AlgorithmSelector::ListAlgorithms(PrimitiveType prim) const {
    switch (prim) {
        case PrimitiveType::ALL_REDUCE:
            return {"AllReduceRing"};
        case PrimitiveType::ALL_GATHER:
            return {"AllGatherRing"};
        case PrimitiveType::REDUCE_SCATTER:
            return {"ReduceScatterRing"};
        case PrimitiveType::ALL_TO_ALL:
            return {"AlltoAllDirect"};
        default:
            return {};
    }
}

} // namespace cann
