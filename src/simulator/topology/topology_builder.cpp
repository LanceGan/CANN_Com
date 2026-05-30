// f:/Projects/CANN_Com/src/simulator/topology/topology_builder.cpp
#include "simulator/topology/topology_builder.h"
#include "common/error.h"

namespace cann {

TopologyBuilder& TopologyBuilder::addNode(const std::string& name, uint32_t num_devices,
                                           NPUType type, double intra_bw_gbps,
                                           double intra_latency_ms) {
    Node node;
    node.name = name;
    node.node_id = static_cast<uint32_t>(topo_.nodes_.size());

    for (uint32_t i = 0; i < num_devices; i++) {
        node.devices.emplace_back(i, type);
    }

    // Create Full Mesh intra-node links (HCCS)
    for (uint32_t i = 0; i < num_devices; i++) {
        for (uint32_t j = i + 1; j < num_devices; j++) {
            Link link;
            link.type = LinkType::HCCS;
            link.bandwidth_gbps = intra_bw_gbps;
            link.latency_ms = intra_latency_ms;
            link.error_rate = 0.0;
            node.intra_links.push_back(link);
        }
    }

    name_to_id_[name] = node.node_id;
    topo_.nodes_.push_back(std::move(node));
    return *this;
}

TopologyBuilder& TopologyBuilder::connectNodes(const std::string& from, const std::string& to,
                                                LinkType link_type, double bandwidth_gbps,
                                                double latency_ms, double error_rate) {
    auto it_from = name_to_id_.find(from);
    auto it_to = name_to_id_.find(to);
    CANN_VALIDATE_PARAM(it_from != name_to_id_.end());
    CANN_VALIDATE_PARAM(it_to != name_to_id_.end());

    Link link;
    link.type = link_type;
    link.bandwidth_gbps = bandwidth_gbps;
    link.latency_ms = latency_ms;
    link.error_rate = error_rate;

    uint32_t from_id = it_from->second;
    uint32_t to_id = it_to->second;

    auto key_fwd = Topology::makeLinkKey(from_id, to_id);
    auto key_rev = Topology::makeLinkKey(to_id, from_id);
    topo_.inter_links_[key_fwd].push_back(link);
    topo_.inter_links_[key_rev].push_back(link);

    return *this;
}

Topology TopologyBuilder::build() {
    return topo_;
}

} // namespace cann
