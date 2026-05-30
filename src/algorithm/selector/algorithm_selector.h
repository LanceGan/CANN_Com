// f:/Projects/CANN_Com/src/algorithm/selector/algorithm_selector.h
#pragma once

#include "algorithm/algorithm.h"
#include "algorithm/allreduce/allreduce_ring.h"
#include "algorithm/allreduce/allreduce_rhd.h"
#include "algorithm/allgather/allgather_ring.h"
#include "algorithm/allgather/allgather_butterfly.h"
#include "algorithm/reduce_scatter/reduce_scatter_ring.h"
#include "algorithm/reduce_scatter/reduce_scatter_butterfly.h"
#include "algorithm/alltoall/alltoall_direct.h"
#include "simulator/topology/topology.h"
#include <vector>
#include <string>
#include <memory>

namespace cann {

enum class PrimitiveType : uint8_t {
    ALL_REDUCE = 0,
    ALL_GATHER = 1,
    REDUCE_SCATTER = 2,
    ALL_TO_ALL = 3,
    BROADCAST = 4,
};

class AlgorithmSelector {
public:
    AlgorithmSelector();

    Algorithm* Select(PrimitiveType prim, size_t bytes, uint32_t nranks);

    Algorithm* SelectWithTopology(PrimitiveType prim, size_t bytes,
                                  uint32_t nranks, const Topology& topo);

    std::vector<std::string> ListAlgorithms(PrimitiveType prim) const;

private:
    AllReduceRing allreduce_ring_;
    AllReduceRHD allreduce_rhd_;
    AllGatherRing allgather_ring_;
    AllGatherButterfly allgather_butterfly_;
    ReduceScatterRing reduce_scatter_ring_;
    ReduceScatterButterfly reduce_scatter_butterfly_;
    AlltoAllDirect alltoall_direct_;
};

} // namespace cann
