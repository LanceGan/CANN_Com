// f:/Projects/CANN_Com/src/simulator/network/link_model.h
#pragma once

#include "simulator/topology/topology.h"
#include <cstddef>

namespace cann {

class LinkModel {
public:
    explicit LinkModel(const Link& link);

    // Calculate transfer time in milliseconds for given data size
    // num_concurrent: number of transfers sharing this link (for congestion)
    double transferTimeMs(size_t bytes, uint32_t num_concurrent = 1) const;

    // Get effective bandwidth in GB/s (accounting for congestion)
    double effectiveBandwidthGbps(uint32_t num_concurrent = 1) const;

    const Link& link() const { return link_; }

private:
    Link link_;
};

} // namespace cann
