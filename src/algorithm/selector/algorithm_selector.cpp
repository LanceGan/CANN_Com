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

// Topology-aware algorithm selection.
// Selection logic:
//   - Small data (<=64KB): Always use Ring (lower latency)
//   - Single node (all HCCS): Use RHD for large data, Ring for small data
//   - Multi-node (HCCS + ROCE): Use Ring (better for heterogeneous bandwidth)
//   - Large data (>4MB): Use RHD if power-of-2 ranks, otherwise Ring
Algorithm* AlgorithmSelector::SelectWithTopology(PrimitiveType prim, size_t bytes,
                                                  uint32_t nranks,
                                                  const Topology& topo) {
    constexpr size_t kSmallThreshold = 64 * 1024;        // 64KB
    constexpr size_t kLargeThreshold = 4 * 1024 * 1024;  // 4MB

    bool is_single_node = (topo.numNodes() <= 1);
    bool is_small = (bytes <= kSmallThreshold);
    bool is_large = (bytes > kLargeThreshold);
    bool is_power_of_2 = (nranks > 0) && ((nranks & (nranks - 1)) == 0);

    // For ALL_TO_ALL, topology doesn't change the algorithm choice
    if (prim == PrimitiveType::ALL_TO_ALL) {
        return &alltoall_direct_;
    }

    // Small data always uses Ring — lowest latency regardless of topology
    if (is_small) {
        switch (prim) {
            case PrimitiveType::ALL_REDUCE:      return &allreduce_ring_;
            case PrimitiveType::ALL_GATHER:      return &allgather_ring_;
            case PrimitiveType::REDUCE_SCATTER:  return &reduce_scatter_ring_;
            default:                             return nullptr;
        }
    }

    // Multi-node: Ring is better for heterogeneous bandwidth (HCCS + ROCE)
    if (!is_single_node) {
        switch (prim) {
            case PrimitiveType::ALL_REDUCE:      return &allreduce_ring_;
            case PrimitiveType::ALL_GATHER:      return &allgather_ring_;
            case PrimitiveType::REDUCE_SCATTER:  return &reduce_scatter_ring_;
            default:                             return nullptr;
        }
    }

    // Single node, large data, power-of-2 ranks: use hierarchical algorithms
    if (is_large && is_power_of_2) {
        switch (prim) {
            case PrimitiveType::ALL_REDUCE:      return &allreduce_rhd_;
            case PrimitiveType::ALL_GATHER:      return &allgather_butterfly_;
            case PrimitiveType::REDUCE_SCATTER:  return &reduce_scatter_butterfly_;
            default:                             return nullptr;
        }
    }

    // Single node, medium data or non-power-of-2: use Ring
    switch (prim) {
        case PrimitiveType::ALL_REDUCE:      return &allreduce_ring_;
        case PrimitiveType::ALL_GATHER:      return &allgather_ring_;
        case PrimitiveType::REDUCE_SCATTER:  return &reduce_scatter_ring_;
        default:                             return nullptr;
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
