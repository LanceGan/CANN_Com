// f:/Projects/CANN_Com/src/simulator/channel/pure_sim_channel.cpp
#include "simulator/channel/pure_sim_channel.h"
#include "common/error.h"
#include <cstring>
#include <algorithm>

namespace cann {

// Static shared mailbox — represents the simulated network fabric.
// All PureSimChannel instances share this to allow data transfer between
// independently constructed sender/receiver channels.
std::unordered_map<uint64_t, std::vector<uint8_t>>& PureSimChannel::mailbox() {
    static std::unordered_map<uint64_t, std::vector<uint8_t>> mb;
    return mb;
}

std::mutex& PureSimChannel::mailboxMutex() {
    static std::mutex m;
    return m;
}

void PureSimChannel::clearMailbox() {
    std::lock_guard<std::mutex> lock(mailboxMutex());
    mailbox().clear();
}

PureSimChannel::PureSimChannel(const Topology& topo, uint32_t rank)
    : topo_(topo), rank_(rank) {
    CANN_VALIDATE_PARAM(rank < topo.numRanks());
}

void PureSimChannel::send(const void* data, size_t bytes, uint32_t dst_rank) {
    CANN_VALIDATE_PARAM(dst_rank < topo_.numRanks());
    CANN_VALIDATE_PARAM(dst_rank != rank_);

    LinkModel link = getLinkTo(dst_rank);
    double time_ms = link.transferTimeMs(bytes);

    // Buffer data in the shared mailbox so recv() on a separate
    // PureSimChannel instance can retrieve it.
    uint64_t key = (static_cast<uint64_t>(rank_) << 32) | dst_rank;
    {
        std::lock_guard<std::mutex> lock(mailboxMutex());
        auto& buf = mailbox()[key];
        buf.resize(bytes);
        std::memcpy(buf.data(), data, bytes);
    }

    stats_.num_sends++;
    stats_.total_send_time_ms += time_ms;
    stats_.total_bytes_sent += bytes;
}

void PureSimChannel::recv(void* buffer, size_t bytes, uint32_t src_rank) {
    CANN_VALIDATE_PARAM(src_rank < topo_.numRanks());
    CANN_VALIDATE_PARAM(src_rank != rank_);

    LinkModel link = getLinkTo(src_rank);
    double time_ms = link.transferTimeMs(bytes);

    // Retrieve data from the shared mailbox.
    uint64_t key = (static_cast<uint64_t>(src_rank) << 32) | rank_;
    {
        std::lock_guard<std::mutex> lock(mailboxMutex());
        auto it = mailbox().find(key);
        if (it != mailbox().end()) {
            size_t copy_bytes = std::min(bytes, it->second.size());
            std::memcpy(buffer, it->second.data(), copy_bytes);
            mailbox().erase(it);
        }
    }

    stats_.num_recvs++;
    stats_.total_recv_time_ms += time_ms;
    stats_.total_bytes_received += bytes;
}

void PureSimChannel::barrier() {
    stats_.num_barriers++;
}

LinkModel PureSimChannel::getLinkTo(uint32_t other_rank) const {
    uint32_t my_node = topo_.rankToNodeId(rank_);
    uint32_t other_node = topo_.rankToNodeId(other_rank);

    if (my_node == other_node) {
        // Intra-node: use HCCS link
        const auto& intra_links = topo_.getIntraNodeLinks(my_node);
        // For Full Mesh, any HCCS link has the same parameters
        return LinkModel(intra_links[0]);
    }

    // Inter-node: use ROCE link
    auto inter_links = topo_.getLinks(my_node, other_node);
    if (!inter_links.empty()) {
        return LinkModel(inter_links[0]);
    }

    throw CannException("No link found between ranks " +
                        std::to_string(rank_) + " and " +
                        std::to_string(other_rank));
}

} // namespace cann
