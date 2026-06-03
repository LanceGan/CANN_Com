// f:/Projects/CANN_Com/src/simulator/channel/fault_channel.cpp
#include "simulator/channel/fault_channel.h"
#include "common/error.h"

namespace cann {

FaultChannel::FaultChannel(std::unique_ptr<IChannel> inner, const FaultConfig& config)
    : inner_(std::move(inner)), config_(config), rng_(42) {}

void FaultChannel::send(const void* data, size_t bytes, uint32_t dst_rank) {
    // Flow control: atomically acquire a send slot via CAS loop
    // (avoids TOCTOU race between load() and fetch_add())
    if (config_.max_concurrent_sends > 0) {
        while (true) {
            uint32_t current = active_sends_.load();
            if (current < config_.max_concurrent_sends) {
                if (active_sends_.compare_exchange_weak(current, current + 1)) {
                    break;
                }
            }
        }
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
            // Data corruption simulation
            if (config_.data_corruption_rate > 0.0) {
                std::lock_guard<std::mutex> lock(rng_mutex_);
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                if (dist(rng_) < config_.data_corruption_rate) {
                    fault_stats_.num_corruptions++;
                    // Note: In simulation, we track corruption but don't modify actual data
                    // as the mailbox is shared. The corruption is logged for analysis.
                }
            }
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
            // Data corruption simulation
            if (config_.data_corruption_rate > 0.0) {
                std::lock_guard<std::mutex> lock(rng_mutex_);
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                if (dist(rng_) < config_.data_corruption_rate) {
                    fault_stats_.num_corruptions++;
                }
            }
            return;
        }
    }

    inner_->recv(buffer, bytes, src_rank);
    // Data corruption simulation
    if (config_.data_corruption_rate > 0.0) {
        std::lock_guard<std::mutex> lock(rng_mutex_);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(rng_) < config_.data_corruption_rate) {
            fault_stats_.num_corruptions++;
        }
    }
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
