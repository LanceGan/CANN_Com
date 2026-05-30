// f:/Projects/CANN_Com/src/simulator/channel/fault_channel.cpp
#include "simulator/channel/fault_channel.h"
#include "common/error.h"

namespace cann {

FaultChannel::FaultChannel(std::unique_ptr<IChannel> inner, const FaultConfig& config)
    : inner_(std::move(inner)), config_(config), rng_(42) {}

void FaultChannel::send(const void* data, size_t bytes, uint32_t dst_rank) {
    // Flow control: wait until a slot is available
    if (config_.max_concurrent_sends > 0) {
        while (active_sends_.load() >= config_.max_concurrent_sends) {
            // Spin-wait for a free slot
        }
    }

    // Track active sends for flow control
    if (config_.max_concurrent_sends > 0) {
        active_sends_.fetch_add(1);
    }

    bool success = false;
    uint32_t retries = 0;

    try {
        for (uint32_t attempt = 0; attempt <= config_.max_retries; ++attempt) {
            if (shouldFail()) {
                fault_stats_.num_faults++;
                throw CannException("Simulated link failure on send");
            }
            if (config_.timeout_ms > 0.0 && shouldTimeout()) {
                fault_stats_.num_timeouts++;
                if (attempt < config_.max_retries) {
                    retries++;
                    fault_stats_.num_retries++;
                    continue;
                }
                throw CannException("Simulated timeout on send after retries");
            }
            inner_->send(data, bytes, dst_rank);
            success = true;
            break;
        }
    } catch (...) {
        if (config_.max_concurrent_sends > 0) {
            active_sends_.fetch_sub(1);
        }
        throw;
    }

    if (config_.max_concurrent_sends > 0) {
        active_sends_.fetch_sub(1);
    }

    if (!success) {
        throw CannException("Simulated timeout on send after retries");
    }
}

void FaultChannel::recv(void* buffer, size_t bytes, uint32_t src_rank) {
    if (shouldFail()) {
        fault_stats_.num_faults++;
        throw CannException("Simulated link failure on recv");
    }

    // Retransmission on timeout for recv
    if (config_.timeout_ms > 0.0) {
        for (uint32_t attempt = 0; attempt <= config_.max_retries; ++attempt) {
            if (shouldTimeout()) {
                fault_stats_.num_timeouts++;
                if (attempt < config_.max_retries) {
                    fault_stats_.num_retries++;
                    continue;
                }
                throw CannException("Simulated timeout on recv after retries");
            }
            inner_->recv(buffer, bytes, src_rank);
            return;
        }
    }

    inner_->recv(buffer, bytes, src_rank);
}

void FaultChannel::barrier() {
    inner_->barrier();
}

bool FaultChannel::shouldFail() {
    if (config_.link_failure_rate <= 0.0) return false;
    std::lock_guard<std::mutex> lock(rng_mutex_);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_) < config_.link_failure_rate;
}

bool FaultChannel::shouldTimeout() {
    if (config_.timeout_ms <= 0.0) return false;
    std::lock_guard<std::mutex> lock(rng_mutex_);
    // Use timeout_ms as a probability threshold for simulation.
    // E.g., timeout_ms=500 means 50% chance of timeout per attempt.
    double probability = std::min(config_.timeout_ms / 1000.0, 1.0);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_) < probability;
}

} // namespace cann
