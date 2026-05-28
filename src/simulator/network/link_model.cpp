// f:/Projects/CANN_Com/src/simulator/network/link_model.cpp
#include "simulator/network/link_model.h"
#include <algorithm>

namespace cann {

LinkModel::LinkModel(const Link& link) : link_(link) {}

double LinkModel::transferTimeMs(size_t bytes, uint32_t num_concurrent) const {
    if (bytes == 0) return 0.0;

    double effective_bw = effectiveBandwidthGbps(num_concurrent);

    // Convert bytes to GB: bytes / (1024^3)
    double data_gb = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);

    // Transfer time = data / bandwidth (in seconds) + latency
    double transfer_sec = data_gb / effective_bw;
    double transfer_ms = transfer_sec * 1000.0;

    return transfer_ms + link_.latency_ms;
}

double LinkModel::effectiveBandwidthGbps(uint32_t num_concurrent) const {
    return link_.bandwidth_gbps / std::max(num_concurrent, 1u);
}

} // namespace cann
