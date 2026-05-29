#include <gtest/gtest.h>
#include "algorithm/alltoall/alltoall_direct.h"
#include "algorithm/algorithm.h"
#include "simulator/simulator.h"
#include "simulator/topology/topology_builder.h"
#include <vector>
#include <thread>

using namespace cann;

class AlltoAllTest : public ::testing::Test {
protected:
    void SetUp() override { PureSimChannel::clearMailbox(); }
    void TearDown() override { PureSimChannel::clearMailbox(); }
};

TEST_F(AlltoAllTest, FourRanks) {
    uint32_t nranks = 4;

    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();
    Simulator sim(topo, SimMode::PureSim);
    AlltoAllDirect algo;

    // Each rank r has [r*100+0, r*100+1, r*100+2, r*100+3]
    // After AlltoAll, rank r receives element r from each rank
    // Rank 0 gets [0*100+0, 1*100+0, 2*100+0, 3*100+0] = [0, 100, 200, 300]
    std::vector<std::vector<float>> outputs(nranks, std::vector<float>(nranks));
    std::vector<std::thread> threads;

    for (uint32_t r = 0; r < nranks; r++) {
        threads.emplace_back([&, r]() {
            std::vector<float> input(nranks);
            for (size_t i = 0; i < nranks; i++) {
                input[i] = static_cast<float>(r * 100 + i);
            }
            CommContext ctx(r, nranks, sim.getChannel(r));
            algo.Execute(input.data(), outputs[r].data(), nranks,
                         HCCLDataType::FLOAT32, HCCLReduceOp::SUM, ctx);
        });
    }
    for (auto& t : threads) t.join();

    for (uint32_t r = 0; r < nranks; r++) {
        for (uint32_t src = 0; src < nranks; src++) {
            EXPECT_FLOAT_EQ(outputs[r][src], static_cast<float>(src * 100 + r))
                << "Rank " << r << " from src " << src;
        }
    }
}

TEST_F(AlltoAllTest, EightRanks) {
    uint32_t nranks = 8;

    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();
    Simulator sim(topo, SimMode::PureSim);
    AlltoAllDirect algo;

    std::vector<std::vector<float>> outputs(nranks, std::vector<float>(nranks));
    std::vector<std::thread> threads;

    for (uint32_t r = 0; r < nranks; r++) {
        threads.emplace_back([&, r]() {
            std::vector<float> input(nranks, static_cast<float>(r));
            CommContext ctx(r, nranks, sim.getChannel(r));
            algo.Execute(input.data(), outputs[r].data(), nranks,
                         HCCLDataType::FLOAT32, HCCLReduceOp::SUM, ctx);
        });
    }
    for (auto& t : threads) t.join();

    for (uint32_t r = 0; r < nranks; r++) {
        for (uint32_t src = 0; src < nranks; src++) {
            EXPECT_FLOAT_EQ(outputs[r][src], static_cast<float>(src))
                << "Rank " << r << " from src " << src;
        }
    }
}

TEST_F(AlltoAllTest, AlgorithmName) {
    AlltoAllDirect algo;
    EXPECT_STREQ(algo.Name(), "AlltoAllDirect");
}

TEST_F(AlltoAllTest, NumSteps) {
    AlltoAllDirect algo;
    EXPECT_EQ(algo.NumSteps(4), 3);
    EXPECT_EQ(algo.NumSteps(8), 7);
}
