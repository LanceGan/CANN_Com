// f:/Projects/CANN_Com/src/simulator/simulator.cpp
#include "simulator/simulator.h"
#include "common/error.h"

namespace cann {

Simulator::Simulator(const Topology& topo, SimMode mode)
    : topo_(topo), mode_(mode) {
    for (uint32_t r = 0; r < topo_.numRanks(); r++) {
        channels_.push_back(std::make_unique<PureSimChannel>(topo_, r));
    }
}

IChannel& Simulator::getChannel(uint32_t rank) {
    CANN_VALIDATE_PARAM(rank < channels_.size());
    return *channels_[rank];
}

void Simulator::simulateSend(uint32_t src_rank, uint32_t dst_rank,
                              const void* data, size_t bytes) {
    CANN_VALIDATE_PARAM(src_rank < channels_.size());
    channels_[src_rank]->send(data, bytes, dst_rank);
}

void Simulator::simulateRecv(uint32_t dst_rank, uint32_t src_rank,
                              void* buffer, size_t bytes) {
    CANN_VALIDATE_PARAM(dst_rank < channels_.size());
    channels_[dst_rank]->recv(buffer, bytes, src_rank);
}

SimStats Simulator::getStats() const {
    SimStats total;
    for (auto& ch : channels_) {
        auto s = ch->getStats();
        total.total_time_ms += s.total_send_time_ms + s.total_recv_time_ms;
        total.total_sends += s.num_sends;
        total.total_recvs += s.num_recvs;
        total.total_bytes_sent += s.total_bytes_sent;
    }
    return total;
}

void Simulator::resetStats() {
    for (auto& ch : channels_) {
        ch->resetStats();
    }
}

} // namespace cann
