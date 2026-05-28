#include <gtest/gtest.h>
#include "simulator/simulator.h"
#include "simulator/topology/topology_builder.h"

using namespace cann;

class SimulatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        topo_ = TopologyBuilder()
            .addNode("node0", 4, NPUType::ASCEND_910B)
            .build();
        sim_ = std::make_unique<Simulator>(topo_, SimMode::PureSim);
    }

    Topology topo_;
    std::unique_ptr<Simulator> sim_;
};

TEST_F(SimulatorTest, Creation) {
    EXPECT_EQ(sim_->numRanks(), 4u);
    EXPECT_EQ(sim_->mode(), SimMode::PureSim);
}

TEST_F(SimulatorTest, GetChannel) {
    auto& ch = sim_->getChannel(0);
    EXPECT_EQ(ch.rank(), 0u);
}

TEST_F(SimulatorTest, SimulateSendRecv) {
    float send_buf[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float recv_buf[4] = {0.0f};

    sim_->simulateSend(0, 1, send_buf, sizeof(send_buf));
    sim_->simulateRecv(1, 0, recv_buf, sizeof(recv_buf));

    auto stats = sim_->getStats();
    EXPECT_GT(stats.total_time_ms, 0.0);
}

TEST_F(SimulatorTest, ResetClearsStats) {
    float buf[4] = {1.0f};
    sim_->simulateSend(0, 1, buf, sizeof(buf));

    auto stats1 = sim_->getStats();
    EXPECT_GT(stats1.total_time_ms, 0.0);

    sim_->resetStats();
    auto stats2 = sim_->getStats();
    EXPECT_EQ(stats2.total_time_ms, 0.0);
}

TEST_F(SimulatorTest, TwoNodeTopology) {
    auto big_topo = TopologyBuilder()
        .addNode("node0", 8, NPUType::ASCEND_910B)
        .addNode("node1", 8, NPUType::ASCEND_910B)
        .connectNodes("node0", "node1", LinkType::ROCE, 100.0, 0.01)
        .build();

    Simulator big_sim(big_topo, SimMode::PureSim);
    EXPECT_EQ(big_sim.numRanks(), 16u);

    // Cross-node send should use ROCE (higher latency)
    float buf[1024] = {1.0f};
    big_sim.simulateSend(0, 8, buf, sizeof(buf)); // rank 0 -> rank 8 (cross-node)

    auto stats = big_sim.getStats();
    EXPECT_GT(stats.total_time_ms, 0.0);
}
