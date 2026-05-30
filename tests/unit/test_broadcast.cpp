#include <gtest/gtest.h>
#include "algorithm/broadcast/broadcast_ring.h"
#include "algorithm/algorithm.h"
#include "simulator/simulator.h"
#include "simulator/topology/topology_builder.h"
#include <vector>
#include <thread>

using namespace cann;

class BroadcastTest : public ::testing::Test {
protected:
    void SetUp() override { PureSimChannel::clearMailbox(); }
    void TearDown() override { PureSimChannel::clearMailbox(); }
};

TEST_F(BroadcastTest, FourRanks) {
    uint32_t nranks = 4;
    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();
    Simulator sim(topo, SimMode::PureSim);
    BroadcastRing algo;

    // Root (rank 0) has data: [1.0, 2.0, 3.0, 4.0]
    // After broadcast, all ranks should have the same data
    std::vector<float> root_data = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t count = root_data.size();

    std::vector<std::vector<float>> outputs(nranks, std::vector<float>(count));
    std::vector<std::thread> threads;

    for (uint32_t r = 0; r < nranks; r++) {
        threads.emplace_back([&, r]() {
            void* sendbuf = (r == 0) ? root_data.data() : nullptr;
            CommContext ctx(r, nranks, sim.getChannel(r));
            Status s = algo.Execute(sendbuf, outputs[r].data(), count,
                                    HCCLDataType::FLOAT32, HCCLReduceOp::SUM, ctx);
            EXPECT_EQ(s, Status::SUCCESS);
        });
    }
    for (auto& t : threads) t.join();

    // Verify all ranks have root's data
    for (uint32_t r = 0; r < nranks; r++) {
        for (size_t i = 0; i < count; i++) {
            EXPECT_FLOAT_EQ(outputs[r][i], root_data[i])
                << "Rank " << r << " element " << i;
        }
    }
}

TEST_F(BroadcastTest, EightRanks) {
    uint32_t nranks = 8;
    size_t count = 16;

    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();
    Simulator sim(topo, SimMode::PureSim);
    BroadcastRing algo;

    // Root (rank 0) has data: [42.0, 42.0, ..., 42.0]
    std::vector<float> root_data(count, 42.0f);

    std::vector<std::vector<float>> outputs(nranks, std::vector<float>(count));
    std::vector<std::thread> threads;

    for (uint32_t r = 0; r < nranks; r++) {
        threads.emplace_back([&, r]() {
            void* sendbuf = (r == 0) ? root_data.data() : nullptr;
            CommContext ctx(r, nranks, sim.getChannel(r));
            Status s = algo.Execute(sendbuf, outputs[r].data(), count,
                                    HCCLDataType::FLOAT32, HCCLReduceOp::SUM, ctx);
            EXPECT_EQ(s, Status::SUCCESS);
        });
    }
    for (auto& t : threads) t.join();

    // Verify all ranks have root's data
    for (uint32_t r = 0; r < nranks; r++) {
        for (size_t i = 0; i < count; i++) {
            EXPECT_FLOAT_EQ(outputs[r][i], 42.0f)
                << "Rank " << r << " element " << i;
        }
    }
}

TEST_F(BroadcastTest, AlgorithmName) {
    BroadcastRing algo;
    EXPECT_STREQ(algo.Name(), "BroadcastRing");
}

TEST_F(BroadcastTest, NumSteps) {
    BroadcastRing algo;
    EXPECT_EQ(algo.NumSteps(4), 3);
    EXPECT_EQ(algo.NumSteps(8), 7);
}
