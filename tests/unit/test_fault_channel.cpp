// f:/Projects/CANN_Com/tests/unit/test_fault_channel.cpp
#include <gtest/gtest.h>
#include "simulator/channel/fault_channel.h"
#include "simulator/channel/pure_sim_channel.h"
#include "simulator/topology/topology_builder.h"

using namespace cann;

class FaultChannelTest : public ::testing::Test {
protected:
    void SetUp() override { PureSimChannel::clearMailbox(); }
    void TearDown() override { PureSimChannel::clearMailbox(); }
};

TEST_F(FaultChannelTest, NormalOperation) {
    Topology topo = TopologyBuilder()
        .addNode("node0", 4, NPUType::ASCEND_910B)
        .build();

    auto inner = std::make_unique<PureSimChannel>(topo, 0);
    FaultChannel fault(std::move(inner), FaultConfig{});

    float send_buf[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    fault.send(send_buf, sizeof(send_buf), 1);

    auto stats = fault.getStats();
    EXPECT_EQ(stats.num_sends, 1u);
}

TEST_F(FaultChannelTest, LinkFailure) {
    Topology topo = TopologyBuilder()
        .addNode("node0", 4, NPUType::ASCEND_910B)
        .build();

    FaultConfig config;
    config.link_failure_rate = 1.0;

    auto inner = std::make_unique<PureSimChannel>(topo, 0);
    FaultChannel fault(std::move(inner), config);

    float send_buf[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_THROW(fault.send(send_buf, sizeof(send_buf), 1), CannException);
}

TEST_F(FaultChannelTest, StatsTrackFaults) {
    Topology topo = TopologyBuilder()
        .addNode("node0", 4, NPUType::ASCEND_910B)
        .build();

    FaultConfig config;
    config.link_failure_rate = 0.5;

    auto inner = std::make_unique<PureSimChannel>(topo, 0);
    FaultChannel fault(std::move(inner), config);

    int failures = 0;
    for (int i = 0; i < 100; i++) {
        float buf[1] = {1.0f};
        try {
            fault.send(buf, sizeof(buf), 1);
        } catch (const CannException&) {
            failures++;
        }
    }

    auto stats = fault.getFaultStats();
    EXPECT_GT(stats.num_faults, 0u);
}
