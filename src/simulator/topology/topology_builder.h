// f:/Projects/CANN_Com/src/simulator/topology/topology_builder.h
#pragma once

#include "topology.h"
#include <string>

namespace cann {

class TopologyBuilder {
public:
    TopologyBuilder() = default;

    TopologyBuilder& addNode(const std::string& name, uint32_t num_devices,
                             NPUType type, double intra_bw_gbps = 100.0,
                             double intra_latency_ms = 0.001);

    TopologyBuilder& connectNodes(const std::string& from, const std::string& to,
                                  LinkType link_type, double bandwidth_gbps,
                                  double latency_ms, double error_rate = 0.0);

    Topology build();

private:
    Topology topo_;
    std::unordered_map<std::string, uint32_t> name_to_id_;
};

} // namespace cann
