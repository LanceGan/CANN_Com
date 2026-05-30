#include <gtest/gtest.h>
#include "algorithm/allreduce/allreduce_pipeline.h"
#include "algorithm/algorithm.h"
#include "simulator/simulator.h"
#include "simulator/topology/topology_builder.h"
#include <vector>
#include <thread>

using namespace cann;

class AllReducePipelineTest : public ::testing::Test {
protected:
    void SetUp() override { PureSimChannel::clearMailbox(); }
    void TearDown() override { PureSimChannel::clearMailbox(); }

    void TestAllReduceSum(uint32_t nranks, int pipeline_depth = 4) {
        Topology topo = TopologyBuilder()
            .addNode("node0", nranks, NPUType::ASCEND_910B)
            .build();

        Simulator sim(topo, SimMode::PureSim);
        AllReducePipeline algo(pipeline_depth);

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

TEST_F(AllReducePipelineTest, FourRanks) {
    TestAllReduceSum(4);
}

TEST_F(AllReducePipelineTest, EightRanks) {
    TestAllReduceSum(8);
}

TEST_F(AllReducePipelineTest, LargeData) {
    uint32_t nranks = 4;
    size_t count = 1024;

    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();

    Simulator sim(topo, SimMode::PureSim);
    AllReducePipeline algo(4);

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

TEST_F(AllReducePipelineTest, PipelineDepth) {
    // Test with different pipeline depths
    for (int depth : {2, 4, 8}) {
        uint32_t nranks = 4;
        size_t count = 256;

        Topology topo = TopologyBuilder()
            .addNode("node0", nranks, NPUType::ASCEND_910B)
            .build();

        Simulator sim(topo, SimMode::PureSim);
        AllReducePipeline algo(depth);

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
                    << "Rank " << r << " element " << i
                    << " depth " << depth;
            }
        }
    }
}

TEST_F(AllReducePipelineTest, NumSteps) {
    AllReducePipeline algo(4);
    EXPECT_EQ(algo.NumSteps(4), 6);   // 2 * (4 - 1) = 6
    EXPECT_EQ(algo.NumSteps(8), 14);  // 2 * (8 - 1) = 14
}

TEST_F(AllReducePipelineTest, AlgorithmName) {
    AllReducePipeline algo(4);
    EXPECT_STREQ(algo.Name(), "AllReducePipeline");
}
