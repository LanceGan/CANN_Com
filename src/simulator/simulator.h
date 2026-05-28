// f:/Projects/CANN_Com/src/simulator/simulator.h
#pragma once

#include "common/types.h"
#include "simulator/topology/topology.h"
#include "simulator/channel/pure_sim_channel.h"
#include <vector>
#include <memory>

namespace cann {

struct SimStats {
    double total_time_ms = 0.0;
    uint64_t total_sends = 0;
    uint64_t total_recvs = 0;
    double total_bytes_sent = 0;
};

class Simulator {
public:
    Simulator(const Topology& topo, SimMode mode);

    size_t numRanks() const { return topo_.numRanks(); }
    SimMode mode() const { return mode_; }
    const Topology& topology() const { return topo_; }

    IChannel& getChannel(uint32_t rank);

    void simulateSend(uint32_t src_rank, uint32_t dst_rank,
                      const void* data, size_t bytes);

    void simulateRecv(uint32_t dst_rank, uint32_t src_rank,
                      void* buffer, size_t bytes);

    SimStats getStats() const;

    void resetStats();

private:
    Topology topo_;
    SimMode mode_;
    std::vector<std::unique_ptr<IChannel>> channels_;
};

} // namespace cann
