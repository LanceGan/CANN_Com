// f:/Projects/CANN_Com/src/simulator/topology/topology.h
#pragma once

#include "common/types.h"
#include "common/error.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace cann {

struct Link {
    LinkType type;
    double bandwidth_gbps;
    double latency_ms;
    double error_rate;
};

class NPUDevice {
public:
    NPUDevice(uint32_t id, NPUType type)
        : id_(id), type_(type) {}

    uint32_t id() const { return id_; }
    NPUType type() const { return type_; }

private:
    uint32_t id_;
    NPUType type_;
};

struct Node {
    std::string name;
    uint32_t node_id;
    std::vector<NPUDevice> devices;
    std::vector<Link> intra_links;
    // Hardware parameters
    size_t hbm_capacity_gb = 32;      // HBM capacity per device (GB)
    size_t ub_capacity_kb = 192;      // Unified Buffer capacity (KB)
    int numa_node_id = 0;             // NUMA node ID
    double pcie_bandwidth_gbps = 32.0; // PCIe bandwidth (GB/s)
};

class Topology {
public:
    Topology() = default;

    size_t numNodes() const { return nodes_.size(); }
    size_t numDevices() const {
        size_t total = 0;
        for (auto& n : nodes_) total += n.devices.size();
        return total;
    }
    size_t numRanks() const { return numDevices(); }

    const std::string& nodeName(uint32_t node_id) const {
        return nodes_.at(node_id).name;
    }

    const Node& node(uint32_t node_id) const {
        return nodes_.at(node_id);
    }

    const std::vector<Node>& nodes() const { return nodes_; }

    std::vector<Link> getLinks(uint32_t from_node, uint32_t to_node) const {
        auto key = makeLinkKey(from_node, to_node);
        auto it = inter_links_.find(key);
        if (it != inter_links_.end()) return it->second;
        return {};
    }

    const std::vector<Link>& getIntraNodeLinks(uint32_t node_id) const {
        return nodes_.at(node_id).intra_links;
    }

    uint32_t rankToNodeId(uint32_t rank) const {
        uint32_t offset = 0;
        for (auto& n : nodes_) {
            if (rank < offset + n.devices.size()) return n.node_id;
            offset += n.devices.size();
        }
        throw CannException("Invalid rank: " + std::to_string(rank));
    }

    uint32_t rankToLocalId(uint32_t rank) const {
        uint32_t offset = 0;
        for (auto& n : nodes_) {
            if (rank < offset + n.devices.size()) return rank - offset;
            offset += n.devices.size();
        }
        throw CannException("Invalid rank: " + std::to_string(rank));
    }

private:
    friend class TopologyBuilder;

    static uint64_t makeLinkKey(uint32_t a, uint32_t b) {
        return (static_cast<uint64_t>(a) << 32) | b;
    }

    std::vector<Node> nodes_;
    std::unordered_map<uint64_t, std::vector<Link>> inter_links_;
};

} // namespace cann
