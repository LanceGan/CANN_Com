// f:/Projects/CANN_Com/src/algorithm/selector/algorithm_selector.cpp
#include "algorithm/selector/algorithm_selector.h"

namespace cann {

AlgorithmSelector::AlgorithmSelector() = default;

Algorithm* AlgorithmSelector::Select(PrimitiveType prim, size_t bytes,
                                      uint32_t nranks) {
    switch (prim) {
        case PrimitiveType::ALL_REDUCE:
            // RHD is better for large data on power-of-2 ranks
            if (bytes > 4 * 1024 * 1024 && (nranks & (nranks - 1)) == 0) {
                return &allreduce_rhd_;
            }
            return &allreduce_ring_;
        case PrimitiveType::ALL_GATHER:
            if (bytes > 4 * 1024 * 1024 && (nranks & (nranks - 1)) == 0) {
                return &allgather_butterfly_;
            }
            return &allgather_ring_;
        case PrimitiveType::REDUCE_SCATTER:
            if (bytes > 4 * 1024 * 1024 && (nranks & (nranks - 1)) == 0) {
                return &reduce_scatter_butterfly_;
            }
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
            return {"AllReduceRing", "AllReduceRHD"};
        case PrimitiveType::ALL_GATHER:
            return {"AllGatherRing", "AllGatherButterfly"};
        case PrimitiveType::REDUCE_SCATTER:
            return {"ReduceScatterRing", "ReduceScatterButterfly"};
        case PrimitiveType::ALL_TO_ALL:
            return {"AlltoAllDirect"};
        default:
            return {};
    }
}

} // namespace cann
