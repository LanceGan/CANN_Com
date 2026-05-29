// f:/Projects/CANN_Com/src/simulator/channel/pure_sim_channel.h
#pragma once

#include "channel.h"
#include "simulator/topology/topology.h"
#include "simulator/network/link_model.h"
#include <vector>
#include <deque>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <cstdint>

namespace cann {

class PureSimChannel : public IChannel {
public:
    PureSimChannel(const Topology& topo, uint32_t rank);

    void send(const void* data, size_t bytes, uint32_t dst_rank) override;
    void recv(void* buffer, size_t bytes, uint32_t src_rank) override;
    void barrier() override;

    ChannelStats getStats() const override { return stats_; }
    void resetStats() override { stats_ = {}; }
    uint32_t rank() const override { return rank_; }

    // Clear the shared mailbox (useful for test teardown)
    static void clearMailbox();

private:
    const Topology& topo_;
    uint32_t rank_;
    ChannelStats stats_;

    LinkModel getLinkTo(uint32_t other_rank) const;

    // Shared mailbox for simulated data transfer between channel instances.
    // Key: (src_rank << 32 | dst_rank) — stores a FIFO queue of pending messages.
    // Using a queue ensures messages from different algorithm steps are not lost.
    static std::unordered_map<uint64_t, std::deque<std::vector<uint8_t>>>& mailbox();
    static std::mutex& mailboxMutex();
    static std::condition_variable& mailboxCV();
};

} // namespace cann
