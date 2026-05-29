#include <gtest/gtest.h>
#include "algorithm/allreduce/allreduce_rhd.h"
#include "algorithm/algorithm.h"
#include "simulator/simulator.h"
#include "simulator/topology/topology_builder.h"
#include <vector>
#include <thread>

using namespace cann;

class AllReduceRHDTest : public ::testing::Test {
protected:
    void SetUp() override { PureSimChannel::clearMailbox(); }
    void TearDown() override { PureSimChannel::clearMailbox(); }

    void TestAllReduceSum(uint32_t nranks) {
        Topology topo = TopologyBuilder()
            .addNode("node0", nranks, NPUType::ASCEND_910B)
            .build();

        Simulator sim(topo, SimMode::PureSim);
        AllReduceRHD algo;

        float expected = 0.0f;
        for (uint32_t i = 0; i < nranks; i++) expected += static_cast<float>(i);

        std::vector<std::vector<float>> outputs(nranks, std::vector<float>(1));
        std::vector<std::thread> threads;

        for (uint32_t r = 0; r < nranks; r++) {
            threads.emplace_back([&, r]() {
                float input = static_cast<float>(r);
                CommContext ctx(r, nranks, sim.getChannel(r));
                algo.Execute(&input, outputs[r].data(), 1,
                             HCCLDataType::FLOAT32, HCCLReduceOp::SUM, ctx);
            });
        }
        for (auto& t : threads) t.join();

        for (uint32_t r = 0; r < nranks; r++) {
            EXPECT_FLOAT_EQ(outputs[r][0], expected)
                << "Rank " << r << " got " << outputs[r][0]
                << " expected " << expected;
        }
    }
};

TEST_F(AllReduceRHDTest, TwoRanks) { TestAllReduceSum(2); }
TEST_F(AllReduceRHDTest, FourRanks) { TestAllReduceSum(4); }
TEST_F(AllReduceRHDTest, EightRanks) { TestAllReduceSum(8); }

TEST_F(AllReduceRHDTest, LargeData) {
    uint32_t nranks = 4;
    size_t count = 1024;

    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();

    Simulator sim(topo, SimMode::PureSim);
    AllReduceRHD algo;

    float expected_val = 0.0f;
    for (uint32_t i = 0; i < nranks; i++) expected_val += static_cast<float>(i);

    std::vector<std::vector<float>> outputs(nranks, std::vector<float>(count));
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
        for (size_t i = 0; i < count; i++) {
            EXPECT_FLOAT_EQ(outputs[r][i], expected_val)
                << "Rank " << r << " element " << i;
        }
    }
}

TEST_F(AllReduceRHDTest, AlgorithmName) {
    AllReduceRHD algo;
    EXPECT_STREQ(algo.Name(), "AllReduceRHD");
}

TEST_F(AllReduceRHDTest, NumSteps) {
    AllReduceRHD algo;
    EXPECT_EQ(algo.NumSteps(4), 4);   // 2*log2(4) = 4
    EXPECT_EQ(algo.NumSteps(8), 6);   // 2*log2(8) = 6
}
