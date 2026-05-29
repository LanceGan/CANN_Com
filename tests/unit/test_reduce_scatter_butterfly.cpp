#include <gtest/gtest.h>
#include "algorithm/reduce_scatter/reduce_scatter_butterfly.h"
#include "algorithm/algorithm.h"
#include "simulator/simulator.h"
#include "simulator/topology/topology_builder.h"
#include <vector>
#include <thread>

using namespace cann;

class ReduceScatterButterflyTest : public ::testing::Test {
protected:
    void SetUp() override { PureSimChannel::clearMailbox(); }
    void TearDown() override { PureSimChannel::clearMailbox(); }
};

TEST_F(ReduceScatterButterflyTest, FourRanks) {
    uint32_t nranks = 4;
    size_t count = nranks;  // 4 elements, 1 per rank after scatter

    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();
    Simulator sim(topo, SimMode::PureSim);
    ReduceScatterButterfly algo;

    // Each rank has [rank*10+0, rank*10+1, rank*10+2, rank*10+3]
    // After ReduceScatter SUM, rank r gets sum of all ranks' element r
    std::vector<std::vector<float>> outputs(nranks, std::vector<float>(1));
    std::vector<std::thread> threads;

    for (uint32_t r = 0; r < nranks; r++) {
        threads.emplace_back([&, r]() {
            std::vector<float> input(count);
            for (size_t i = 0; i < count; i++) {
                input[i] = static_cast<float>(r * 10 + i);
            }
            CommContext ctx(r, nranks, sim.getChannel(r));
            algo.Execute(input.data(), outputs[r].data(), count,
                         HCCLDataType::FLOAT32, HCCLReduceOp::SUM, ctx);
        });
    }
    for (auto& t : threads) t.join();

    for (uint32_t r = 0; r < nranks; r++) {
        float expected = 0.0f;
        for (uint32_t k = 0; k < nranks; k++) {
            expected += static_cast<float>(k * 10 + r);
        }
        EXPECT_FLOAT_EQ(outputs[r][0], expected) << "Rank " << r;
    }
}

TEST_F(ReduceScatterButterflyTest, EightRanks) {
    uint32_t nranks = 8;
    size_t count = 16;

    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();
    Simulator sim(topo, SimMode::PureSim);
    ReduceScatterButterfly algo;

    // Each rank has all elements set to its rank value
    // After ReduceScatter SUM, each output element = sum of all ranks = 0+1+...+7 = 28
    float expected_sum = 0.0f;
    for (uint32_t i = 0; i < nranks; i++) expected_sum += static_cast<float>(i);

    size_t out_count = count / nranks;
    std::vector<std::vector<float>> outputs(nranks, std::vector<float>(out_count));
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
        for (size_t i = 0; i < out_count; i++) {
            EXPECT_FLOAT_EQ(outputs[r][i], expected_sum)
                << "Rank " << r << " element " << i;
        }
    }
}

TEST_F(ReduceScatterButterflyTest, AlgorithmName) {
    ReduceScatterButterfly algo;
    EXPECT_STREQ(algo.Name(), "ReduceScatterButterfly");
}

TEST_F(ReduceScatterButterflyTest, NumSteps) {
    ReduceScatterButterfly algo;
    EXPECT_EQ(algo.NumSteps(1), 0);
    EXPECT_EQ(algo.NumSteps(2), 1);
    EXPECT_EQ(algo.NumSteps(4), 2);
    EXPECT_EQ(algo.NumSteps(8), 3);
    EXPECT_EQ(algo.NumSteps(16), 4);
}
