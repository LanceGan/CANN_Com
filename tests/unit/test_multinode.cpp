// f:/Projects/CANN_Com/tests/unit/test_multinode.cpp
#include <gtest/gtest.h>
#include "algorithm/allreduce/allreduce_ring.h"
#include "algorithm/allreduce/allreduce_rhd.h"
#include "algorithm/allgather/allgather_ring.h"
#include "algorithm/algorithm.h"
#include "simulator/simulator.h"
#include "simulator/topology/topology_builder.h"
#include <vector>
#include <thread>

using namespace cann;

class MultiNodeTest : public ::testing::Test {
protected:
    void SetUp() override { PureSimChannel::clearMailbox(); }
    void TearDown() override { PureSimChannel::clearMailbox(); }
};

TEST_F(MultiNodeTest, TwoNodeAllReduceRing) {
    uint32_t ranks_per_node = 8;
    uint32_t total_ranks = ranks_per_node * 2;

    Topology topo = TopologyBuilder()
        .addNode("node0", ranks_per_node, NPUType::ASCEND_910B)
        .addNode("node1", ranks_per_node, NPUType::ASCEND_910B)
        .connectNodes("node0", "node1", LinkType::ROCE, 100.0, 0.01)
        .build();

    Simulator sim(topo, SimMode::PureSim);
    AllReduceRing algo;

    float expected = 0.0f;
    for (uint32_t i = 0; i < total_ranks; i++) expected += static_cast<float>(i);

    std::vector<std::vector<float>> outputs(total_ranks, std::vector<float>(1));
    std::vector<std::thread> threads;

    for (uint32_t r = 0; r < total_ranks; r++) {
        threads.emplace_back([&, r]() {
            float input = static_cast<float>(r);
            CommContext ctx(r, total_ranks, sim.getChannel(r));
            algo.Execute(&input, outputs[r].data(), 1,
                         HCCLDataType::FLOAT32, HCCLReduceOp::SUM, ctx);
        });
    }
    for (auto& t : threads) t.join();

    for (uint32_t r = 0; r < total_ranks; r++) {
        EXPECT_FLOAT_EQ(outputs[r][0], expected) << "Rank " << r;
    }
}

TEST_F(MultiNodeTest, TwoNodeAllReduceRHD) {
    uint32_t ranks_per_node = 8;
    uint32_t total_ranks = ranks_per_node * 2;

    Topology topo = TopologyBuilder()
        .addNode("node0", ranks_per_node, NPUType::ASCEND_910B)
        .addNode("node1", ranks_per_node, NPUType::ASCEND_910B)
        .connectNodes("node0", "node1", LinkType::ROCE, 100.0, 0.01)
        .build();

    Simulator sim(topo, SimMode::PureSim);
    AllReduceRHD algo;

    float expected = 0.0f;
    for (uint32_t i = 0; i < total_ranks; i++) expected += static_cast<float>(i);

    std::vector<std::vector<float>> outputs(total_ranks, std::vector<float>(1));
    std::vector<std::thread> threads;

    for (uint32_t r = 0; r < total_ranks; r++) {
        threads.emplace_back([&, r]() {
            float input = static_cast<float>(r);
            CommContext ctx(r, total_ranks, sim.getChannel(r));
            algo.Execute(&input, outputs[r].data(), 1,
                         HCCLDataType::FLOAT32, HCCLReduceOp::SUM, ctx);
        });
    }
    for (auto& t : threads) t.join();

    for (uint32_t r = 0; r < total_ranks; r++) {
        EXPECT_FLOAT_EQ(outputs[r][0], expected) << "Rank " << r;
    }
}

TEST_F(MultiNodeTest, TwoNodeAllGather) {
    uint32_t ranks_per_node = 4;
    uint32_t total_ranks = ranks_per_node * 2;

    Topology topo = TopologyBuilder()
        .addNode("node0", ranks_per_node, NPUType::ASCEND_910B)
        .addNode("node1", ranks_per_node, NPUType::ASCEND_910B)
        .connectNodes("node0", "node1", LinkType::ROCE, 100.0, 0.01)
        .build();

    Simulator sim(topo, SimMode::PureSim);
    AllGatherRing algo;

    std::vector<std::vector<float>> outputs(total_ranks, std::vector<float>(total_ranks));
    std::vector<std::thread> threads;

    for (uint32_t r = 0; r < total_ranks; r++) {
        threads.emplace_back([&, r]() {
            float input = static_cast<float>(r) * 10.0f;
            CommContext ctx(r, total_ranks, sim.getChannel(r));
            algo.Execute(&input, outputs[r].data(), 1,
                         HCCLDataType::FLOAT32, HCCLReduceOp::SUM, ctx);
        });
    }
    for (auto& t : threads) t.join();

    for (uint32_t r = 0; r < total_ranks; r++) {
        for (uint32_t i = 0; i < total_ranks; i++) {
            EXPECT_FLOAT_EQ(outputs[r][i], static_cast<float>(i) * 10.0f)
                << "Rank " << r << " element " << i;
        }
    }
}
