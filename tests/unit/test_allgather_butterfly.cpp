#include <gtest/gtest.h>
#include "algorithm/allgather/allgather_butterfly.h"
#include "algorithm/algorithm.h"
#include "simulator/simulator.h"
#include "simulator/topology/topology_builder.h"
#include <vector>
#include <thread>

using namespace cann;

class AllGatherButterflyTest : public ::testing::Test {
protected:
    void SetUp() override { PureSimChannel::clearMailbox(); }
    void TearDown() override { PureSimChannel::clearMailbox(); }
};

TEST_F(AllGatherButterflyTest, FourRanks) {
    uint32_t nranks = 4;
    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();
    Simulator sim(topo, SimMode::PureSim);
    AllGatherButterfly algo;

    // Each rank has one float: rank_id * 10.0f
    // After AllGather, each rank should have [0, 10, 20, 30]
    std::vector<std::vector<float>> outputs(nranks, std::vector<float>(nranks));
    std::vector<std::thread> threads;

    for (uint32_t r = 0; r < nranks; r++) {
        threads.emplace_back([&, r]() {
            float input = static_cast<float>(r) * 10.0f;
            CommContext ctx(r, nranks, sim.getChannel(r));
            Status s = algo.Execute(&input, outputs[r].data(), 1,
                                    HCCLDataType::FLOAT32, HCCLReduceOp::SUM, ctx);
            EXPECT_EQ(s, Status::SUCCESS);
        });
    }
    for (auto& t : threads) t.join();

    for (uint32_t r = 0; r < nranks; r++) {
        for (uint32_t i = 0; i < nranks; i++) {
            EXPECT_FLOAT_EQ(outputs[r][i], static_cast<float>(i) * 10.0f)
                << "Rank " << r << " element " << i;
        }
    }
}

TEST_F(AllGatherButterflyTest, EightRanks) {
    uint32_t nranks = 8;
    size_t count = 128;

    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();
    Simulator sim(topo, SimMode::PureSim);
    AllGatherButterfly algo;

    std::vector<std::vector<float>> outputs(nranks, std::vector<float>(nranks * count));
    std::vector<std::thread> threads;

    for (uint32_t r = 0; r < nranks; r++) {
        threads.emplace_back([&, r]() {
            std::vector<float> input(count, static_cast<float>(r));
            CommContext ctx(r, nranks, sim.getChannel(r));
            algo.Execute(input.data(), outputs[r].data(), count,
                         HCCLDataType::FLOAT32, HCCLReduceOp::SUM, ctx);
        });
    }
    for (auto& t : threads) t.join();

    for (uint32_t r = 0; r < nranks; r++) {
        for (uint32_t src = 0; src < nranks; src++) {
            for (size_t i = 0; i < count; i++) {
                EXPECT_FLOAT_EQ(outputs[r][src * count + i], static_cast<float>(src))
                    << "Rank " << r << " src " << src << " elem " << i;
            }
        }
    }
}

TEST_F(AllGatherButterflyTest, AlgorithmName) {
    AllGatherButterfly algo;
    EXPECT_STREQ(algo.Name(), "AllGatherButterfly");
}

TEST_F(AllGatherButterflyTest, NumSteps) {
    AllGatherButterfly algo;
    EXPECT_EQ(algo.NumSteps(4), 2);
    EXPECT_EQ(algo.NumSteps(8), 3);
}
