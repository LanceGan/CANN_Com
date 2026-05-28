#include <gtest/gtest.h>
#include "simulator/network/link_model.h"
#include "simulator/topology/topology.h"

using namespace cann;

TEST(LinkModelTest, BasicTransferTime) {
    Link link;
    link.type = LinkType::HCCS;
    link.bandwidth_gbps = 100.0;   // 100 GB/s
    link.latency_ms = 0.001;       // 1 microsecond
    link.error_rate = 0.0;

    LinkModel model(link);

    // 1 GB data at 100 GB/s = 10ms transfer + 0.001ms latency
    double time_ms = model.transferTimeMs(1024 * 1024 * 1024ULL); // 1 GB
    EXPECT_NEAR(time_ms, 10.001, 0.01);
}

TEST(LinkModelTest, SmallDataTransfer) {
    Link link;
    link.type = LinkType::HCCS;
    link.bandwidth_gbps = 100.0;
    link.latency_ms = 0.001;
    link.error_rate = 0.0;

    LinkModel model(link);

    // 64KB data: 64*1024 / (100*1e9) * 1e3 ms + 0.001ms latency
    double time_ms = model.transferTimeMs(64 * 1024);
    EXPECT_NEAR(time_ms, 0.001 + 64.0 * 1024 / (100.0 * 1e9) * 1e3, 0.0001);
}

TEST(LinkModelTest, CongestionReducesBandwidth) {
    Link link;
    link.type = LinkType::ROCE;
    link.bandwidth_gbps = 100.0;
    link.latency_ms = 0.01;
    link.error_rate = 0.0;

    LinkModel model(link);

    // Without congestion
    double time_normal = model.transferTimeMs(1024 * 1024 * 1024ULL);

    // With 2 concurrent transfers sharing the link
    double time_congested = model.transferTimeMs(1024 * 1024 * 1024ULL, /*num_concurrent=*/2);

    // Congested should be roughly 2x slower
    EXPECT_GT(time_congested, time_normal * 1.5);
}

TEST(LinkModelTest, ROCEvsHCCSLatency) {
    Link hccs;
    hccs.type = LinkType::HCCS;
    hccs.bandwidth_gbps = 100.0;
    hccs.latency_ms = 0.001;
    hccs.error_rate = 0.0;

    Link roce;
    roce.type = LinkType::ROCE;
    roce.bandwidth_gbps = 100.0;
    roce.latency_ms = 0.01;
    roce.error_rate = 0.0;

    LinkModel hccs_model(hccs);
    LinkModel roce_model(roce);

    // Same data, ROCE should have higher latency
    size_t data_size = 1024 * 1024; // 1 MB
    EXPECT_LT(hccs_model.transferTimeMs(data_size), roce_model.transferTimeMs(data_size));
}
