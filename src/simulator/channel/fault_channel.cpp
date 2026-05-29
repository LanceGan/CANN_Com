// f:/Projects/CANN_Com/src/simulator/channel/fault_channel.cpp
#include "simulator/channel/fault_channel.h"
#include "common/error.h"

namespace cann {

FaultChannel::FaultChannel(std::unique_ptr<IChannel> inner, const FaultConfig& config)
    : inner_(std::move(inner)), config_(config), rng_(42) {}

void FaultChannel::send(const void* data, size_t bytes, uint32_t dst_rank) {
    if (shouldFail()) {
        fault_stats_.num_faults++;
        throw CannException("Simulated link failure on send");
    }
    inner_->send(data, bytes, dst_rank);
}

void FaultChannel::recv(void* buffer, size_t bytes, uint32_t src_rank) {
    if (shouldFail()) {
        fault_stats_.num_faults++;
        throw CannException("Simulated link failure on recv");
    }
    inner_->recv(buffer, bytes, src_rank);
}

void FaultChannel::barrier() {
    inner_->barrier();
}

bool FaultChannel::shouldFail() {
    if (config_.link_failure_rate <= 0.0) return false;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_) < config_.link_failure_rate;
}

} // namespace cann
