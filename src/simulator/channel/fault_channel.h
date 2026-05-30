// f:/Projects/CANN_Com/src/simulator/channel/fault_channel.h
#pragma once

#include "channel.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <random>

namespace cann {

struct FaultConfig {
    double link_failure_rate = 0.0;
    double timeout_ms = 0.0;
    double data_corruption_rate = 0.0;
    uint32_t max_concurrent_sends = 0;  // 0 = unlimited
    uint32_t max_retries = 3;
};

struct FaultStats {
    uint64_t num_faults = 0;
    uint64_t num_timeouts = 0;
    uint64_t num_corruptions = 0;
    uint64_t num_retries = 0;
};

class FaultChannel : public IChannel {
public:
    FaultChannel(std::unique_ptr<IChannel> inner, const FaultConfig& config);

    void send(const void* data, size_t bytes, uint32_t dst_rank) override;
    void recv(void* buffer, size_t bytes, uint32_t src_rank) override;
    void barrier() override;

    ChannelStats getStats() const override { return inner_->getStats(); }
    void resetStats() override { inner_->resetStats(); }
    uint32_t rank() const override { return inner_->rank(); }

    FaultStats getFaultStats() const { return fault_stats_; }

private:
    std::unique_ptr<IChannel> inner_;
    FaultConfig config_;
    FaultStats fault_stats_;
    std::mt19937 rng_;
    mutable std::mutex rng_mutex_;
    std::atomic<uint32_t> active_sends_{0};

    bool shouldFail();
    bool shouldTimeout();
};

} // namespace cann
