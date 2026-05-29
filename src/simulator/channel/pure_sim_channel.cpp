// f:/Projects/CANN_Com/src/simulator/channel/pure_sim_channel.cpp
#include "simulator/channel/pure_sim_channel.h"
#include "common/error.h"
#include <cstring>
#include <algorithm>
#include <thread>
#include <chrono>

namespace cann {

// Static shared mailbox — represents the simulated network fabric.
// All PureSimChannel instances share this to allow data transfer between
// independently constructed sender/receiver channels.
// Uses a FIFO queue per key so that messages from different algorithm steps
// are preserved even when a faster rank sends the next step's data before
// a slower rank has consumed the current step's message.
std::unordered_map<uint64_t, std::deque<std::vector<uint8_t>>>& PureSimChannel::mailbox() {
    static std::unordered_map<uint64_t, std::deque<std::vector<uint8_t>>> mb;
    return mb;
}

std::mutex& PureSimChannel::mailboxMutex() {
    static std::mutex m;
    return m;
}

std::condition_variable& PureSimChannel::mailboxCV() {
    static std::condition_variable cv;
    return cv;
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
        auto& queue = mailbox()[key];
        queue.emplace_back();
        auto& buf = queue.back();
        if (bytes > 0) {
            buf.resize(bytes);
            std::memcpy(buf.data(), data, bytes);
        }
        // For 0-byte sends, buf is default-constructed (empty) which
        // still counts as a valid message in the queue.
    }
    // Wake up any thread blocked in recv() waiting for this data.
    mailboxCV().notify_all();

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
    // Spin-wait until the sender has deposited data — this models the
    // blocking nature of a real recv in a collective communication.
    // Uses a short sleep to yield CPU and prevent starvation of the
    // sending thread (yield() alone is insufficient on Windows/MinGW).
    uint64_t key = (static_cast<uint64_t>(src_rank) << 32) | rank_;
    for (;;) {
        {
            std::lock_guard<std::mutex> lock(mailboxMutex());
            auto it = mailbox().find(key);
            if (it != mailbox().end() && !it->second.empty()) {
                auto& front = it->second.front();
                size_t copy_bytes = std::min(bytes, front.size());
                if (copy_bytes > 0) {
                    std::memcpy(buffer, front.data(), copy_bytes);
                }
                it->second.pop_front();
                if (it->second.empty()) {
                    mailbox().erase(it);
                }
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
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
