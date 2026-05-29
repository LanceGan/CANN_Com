// f:/Projects/CANN_Com/tests/fault/test_reliability.cpp
#include <gtest/gtest.h>
#include "algorithm/allreduce/allreduce_ring.h"
#include "simulator/simulator.h"
#include "simulator/topology/topology_builder.h"
#include "simulator/channel/fault_channel.h"
#include "common/error.h"
#include <vector>
#include <thread>
#include <atomic>

using namespace cann;

TEST(ReliabilityTest, AllReduceRingWithLinkFailure) {
    uint32_t nranks = 4;
    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();

    Simulator sim(topo, SimMode::PureSim);
    AllReduceRing algo;

    std::atomic<int> fault_count{0};
    std::vector<std::thread> threads;

    for (uint32_t r = 0; r < nranks; r++) {
        threads.emplace_back([&, r]() {
            float input = static_cast<float>(r);
            float output = 0.0f;
            CommContext ctx(r, nranks, sim.getChannel(r));
            try {
                algo.Execute(&input, &output, 1,
                             HCCLDataType::FLOAT32, HCCLReduceOp::SUM, ctx);
            } catch (const CannException&) {
                fault_count++;
            }
        });
    }
    for (auto& t : threads) t.join();
}

TEST(ReliabilityTest, CorrectnessWithoutFaults) {
    uint32_t nranks = 4;
    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();

    Simulator sim(topo, SimMode::PureSim);
    PureSimChannel::clearMailbox();

    AllReduceRing algo;
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
        EXPECT_FLOAT_EQ(outputs[r][0], expected);
    }
}

TEST(ReliabilityTest, FaultChannelStatsAccumulate) {
    Topology topo = TopologyBuilder()
        .addNode("node0", 4, NPUType::ASCEND_910B)
        .build();

    FaultConfig config;
    config.link_failure_rate = 1.0;

    auto inner = std::make_unique<PureSimChannel>(topo, 0);
    FaultChannel fault(std::move(inner), config);

    int failures = 0;
    for (int i = 0; i < 10; i++) {
        float buf[1] = {1.0f};
        try {
            fault.send(buf, sizeof(buf), 1);
        } catch (const CannException&) {
            failures++;
        }
    }

    EXPECT_EQ(failures, 10);
    EXPECT_EQ(fault.getFaultStats().num_faults, 10u);
}
