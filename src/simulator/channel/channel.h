// f:/Projects/CANN_Com/src/simulator/channel/channel.h
#pragma once

#include "common/types.h"
#include <cstddef>
#include <cstdint>

namespace cann {

struct ChannelStats {
    uint64_t num_sends = 0;
    uint64_t num_recvs = 0;
    uint64_t num_reduces = 0;
    uint64_t num_barriers = 0;
    double total_send_time_ms = 0.0;
    double total_recv_time_ms = 0.0;
    double total_bytes_sent = 0;
    double total_bytes_received = 0;
};

class IChannel {
public:
    virtual ~IChannel() = default;

    virtual void send(const void* data, size_t bytes, uint32_t dst_rank) = 0;
    virtual void recv(void* buffer, size_t bytes, uint32_t src_rank) = 0;
    virtual void barrier() = 0;

    virtual ChannelStats getStats() const = 0;
    virtual void resetStats() = 0;
    virtual uint32_t rank() const = 0;
};

} // namespace cann
