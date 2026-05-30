// f:/Projects/CANN_Com/tests/unit/test_fault_channel.cpp
#include <gtest/gtest.h>
#include "simulator/channel/fault_channel.h"
#include "simulator/channel/pure_sim_channel.h"
#include "simulator/topology/topology_builder.h"
#include <thread>
#include <vector>

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

TEST_F(FaultChannelTest, FlowControlTest) {
    // Use a single-node topology so all intra-node sends succeed
    Topology topo = TopologyBuilder()
        .addNode("node0", 8, NPUType::ASCEND_910B)
        .build();

    FaultConfig config;
    config.max_concurrent_sends = 2;  // Limit to 2 concurrent sends

    // Create sender channel at rank 0
    auto inner_sender = std::make_unique<PureSimChannel>(topo, 0);
    FaultChannel sender(std::move(inner_sender), config);

    // Launch multiple threads that send to different destinations
    auto send_task = [&](uint32_t dst_rank) {
        float buf[1] = {1.0f};
        sender.send(buf, sizeof(buf), dst_rank);
    };

    // Spawn threads targeting ranks 1-7 (valid intra-node dst)
    std::vector<std::thread> threads;
    for (uint32_t i = 1; i < 8; ++i) {
        threads.emplace_back(send_task, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    // Verify that all 7 sends completed successfully
    auto stats = sender.getStats();
    EXPECT_EQ(stats.num_sends, 7u);
}

TEST_F(FaultChannelTest, FlowControlRespectsMaxConcurrent) {
    // Test that flow control actually limits concurrent sends
    Topology topo = TopologyBuilder()
        .addNode("node0", 8, NPUType::ASCEND_910B)
        .build();

    FaultConfig config;
    config.max_concurrent_sends = 1;  // Only 1 concurrent send at a time

    auto inner = std::make_unique<PureSimChannel>(topo, 0);
    FaultChannel fault(std::move(inner), config);

    // Sequential sends should all succeed
    for (uint32_t dst = 1; dst < 8; ++dst) {
        float buf[1] = {1.0f};
        fault.send(buf, sizeof(buf), dst);
    }

    auto stats = fault.getStats();
    EXPECT_EQ(stats.num_sends, 7u);
}

TEST_F(FaultChannelTest, RetransmissionTest) {
    Topology topo = TopologyBuilder()
        .addNode("node0", 4, NPUType::ASCEND_910B)
        .build();

    FaultConfig config;
    config.timeout_ms = 800.0;  // 80% timeout probability
    config.max_retries = 3;

    auto inner = std::make_unique<PureSimChannel>(topo, 0);
    FaultChannel fault(std::move(inner), config);

    int timeouts = 0;
    int successes = 0;

    // Run multiple attempts to get a mix of successes and timeouts
    for (int i = 0; i < 50; i++) {
        float buf[1] = {1.0f};
        try {
            fault.send(buf, sizeof(buf), 1);
            successes++;
        } catch (const CannException& e) {
            if (std::string(e.what()).find("timeout") != std::string::npos) {
                timeouts++;
            }
        }
    }

    auto stats = fault.getFaultStats();
    // With 80% timeout rate and 3 retries, we should see retries and timeouts
    EXPECT_GT(stats.num_timeouts, 0u);
    EXPECT_GT(stats.num_retries, 0u);
}

TEST_F(FaultChannelTest, RetransmissionSuccessAfterRetry) {
    // Test that a send can succeed after a few retries
    Topology topo = TopologyBuilder()
        .addNode("node0", 4, NPUType::ASCEND_910B)
        .build();

    FaultConfig config;
    config.timeout_ms = 500.0;   // 50% timeout probability
    config.max_retries = 10;     // Many retries so most attempts should succeed

    auto inner = std::make_unique<PureSimChannel>(topo, 0);
    FaultChannel fault(std::move(inner), config);

    int successes = 0;
    for (int i = 0; i < 20; i++) {
        float buf[1] = {1.0f};
        try {
            fault.send(buf, sizeof(buf), 1);
            successes++;
        } catch (const CannException&) {
            // Timeout after all retries
        }
    }

    // With 50% timeout rate and 10 retries, most sends should succeed
    EXPECT_GT(successes, 10);

    auto stats = fault.getFaultStats();
    EXPECT_GT(stats.num_retries, 0u);
}
