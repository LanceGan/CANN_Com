#include <gtest/gtest.h>
#include <algorithm>
#include "algorithm/selector/algorithm_selector.h"
#include "simulator/topology/topology_builder.h"

using namespace cann;

TEST(AlgorithmSelectorTest, SelectAllReduceSmallData) {
    AlgorithmSelector selector;
    auto* algo = selector.Select(PrimitiveType::ALL_REDUCE, 1024, 8);
    EXPECT_NE(algo, nullptr);
    EXPECT_STREQ(algo->Name(), "AllReduceRing");
}

TEST(AlgorithmSelectorTest, SelectAllGather) {
    AlgorithmSelector selector;
    auto* algo = selector.Select(PrimitiveType::ALL_GATHER, 4096, 4);
    EXPECT_NE(algo, nullptr);
    EXPECT_STREQ(algo->Name(), "AllGatherRing");
}

TEST(AlgorithmSelectorTest, SelectReduceScatter) {
    AlgorithmSelector selector;
    auto* algo = selector.Select(PrimitiveType::REDUCE_SCATTER, 8192, 8);
    EXPECT_NE(algo, nullptr);
    EXPECT_STREQ(algo->Name(), "ReduceScatterRing");
}

TEST(AlgorithmSelectorTest, SelectAlltoAll) {
    AlgorithmSelector selector;
    auto* algo = selector.Select(PrimitiveType::ALL_TO_ALL, 256, 4);
    EXPECT_NE(algo, nullptr);
    EXPECT_STREQ(algo->Name(), "AlltoAllDirect");
}

TEST(AlgorithmSelectorTest, ReturnsNullForUnknown) {
    AlgorithmSelector selector;
    auto* algo = selector.Select(PrimitiveType::BROADCAST, 1024, 4);
    EXPECT_EQ(algo, nullptr);
}

TEST(AlgorithmSelectorTest, SelectAllReduceLargeDataRHD) {
    AlgorithmSelector selector;
    auto* algo = selector.Select(PrimitiveType::ALL_REDUCE, 16 * 1024 * 1024, 8);
    EXPECT_NE(algo, nullptr);
    EXPECT_STREQ(algo->Name(), "AllReduceRHD");
}

TEST(AlgorithmSelectorTest, SelectAllReduceSmallDataRing) {
    AlgorithmSelector selector;
    auto* algo = selector.Select(PrimitiveType::ALL_REDUCE, 1024, 8);
    EXPECT_NE(algo, nullptr);
    EXPECT_STREQ(algo->Name(), "AllReduceRing");
}

TEST(AlgorithmSelectorTest, ListAlgorithms) {
    AlgorithmSelector selector;
    auto names = selector.ListAlgorithms(PrimitiveType::ALL_REDUCE);
    EXPECT_FALSE(names.empty());
    EXPECT_NE(std::find(names.begin(), names.end(), "AllReduceRing"), names.end());
    EXPECT_NE(std::find(names.begin(), names.end(), "AllReduceRHD"), names.end());
}

// --- Topology-aware selection tests ---

TEST(AlgorithmSelectorTest, TopoSingleNodeLargeDataRHD) {
    // Single node with 8 devices (power-of-2), large data -> RHD
    auto topo = TopologyBuilder()
        .addNode("node0", 8, NPUType::ASCEND_910B)
        .build();

    AlgorithmSelector selector;
    size_t large_bytes = 16 * 1024 * 1024;  // 16MB
    auto* algo = selector.SelectWithTopology(PrimitiveType::ALL_REDUCE,
                                              large_bytes, 8, topo);
    EXPECT_NE(algo, nullptr);
    EXPECT_STREQ(algo->Name(), "AllReduceRHD");
}

TEST(AlgorithmSelectorTest, TopoSingleNodeSmallDataRing) {
    // Single node, small data -> Ring (lower latency)
    auto topo = TopologyBuilder()
        .addNode("node0", 8, NPUType::ASCEND_910B)
        .build();

    AlgorithmSelector selector;
    size_t small_bytes = 1024;  // 1KB
    auto* algo = selector.SelectWithTopology(PrimitiveType::ALL_REDUCE,
                                              small_bytes, 8, topo);
    EXPECT_NE(algo, nullptr);
    EXPECT_STREQ(algo->Name(), "AllReduceRing");
}

TEST(AlgorithmSelectorTest, TopoMultiNodeUsesRing) {
    // Multi-node topology: even with large data, multi-node -> Ring
    auto topo = TopologyBuilder()
        .addNode("node0", 8, NPUType::ASCEND_910B)
        .addNode("node1", 8, NPUType::ASCEND_910B)
        .connectNodes("node0", "node1", LinkType::ROCE, 100.0, 0.001)
        .build();

    AlgorithmSelector selector;
    size_t large_bytes = 16 * 1024 * 1024;  // 16MB
    auto* algo = selector.SelectWithTopology(PrimitiveType::ALL_REDUCE,
                                              large_bytes, 16, topo);
    EXPECT_NE(algo, nullptr);
    EXPECT_STREQ(algo->Name(), "AllReduceRing");
}

TEST(AlgorithmSelectorTest, TopoNonPowerOf2RanksRing) {
    // Single node, 6 devices (non-power-of-2), large data -> Ring
    auto topo = TopologyBuilder()
        .addNode("node0", 6, NPUType::ASCEND_910B)
        .build();

    AlgorithmSelector selector;
    size_t large_bytes = 16 * 1024 * 1024;  // 16MB
    auto* algo = selector.SelectWithTopology(PrimitiveType::ALL_REDUCE,
                                              large_bytes, 6, topo);
    EXPECT_NE(algo, nullptr);
    EXPECT_STREQ(algo->Name(), "AllReduceRing");
}

TEST(AlgorithmSelectorTest, TopoSingleAllGatherLargeDataButterfly) {
    // Single node, AllGather, large data, power-of-2 -> Butterfly
    auto topo = TopologyBuilder()
        .addNode("node0", 8, NPUType::ASCEND_910B)
        .build();

    AlgorithmSelector selector;
    size_t large_bytes = 16 * 1024 * 1024;
    auto* algo = selector.SelectWithTopology(PrimitiveType::ALL_GATHER,
                                              large_bytes, 8, topo);
    EXPECT_NE(algo, nullptr);
    EXPECT_STREQ(algo->Name(), "AllGatherButterfly");
}

TEST(AlgorithmSelectorTest, TopoSingleReduceScatterLargeDataButterfly) {
    // Single node, ReduceScatter, large data, power-of-2 -> Butterfly
    auto topo = TopologyBuilder()
        .addNode("node0", 8, NPUType::ASCEND_910B)
        .build();

    AlgorithmSelector selector;
    size_t large_bytes = 16 * 1024 * 1024;
    auto* algo = selector.SelectWithTopology(PrimitiveType::REDUCE_SCATTER,
                                              large_bytes, 8, topo);
    EXPECT_NE(algo, nullptr);
    EXPECT_STREQ(algo->Name(), "ReduceScatterButterfly");
}

TEST(AlgorithmSelectorTest, TopoAlltoAllUnchanged) {
    // AlltoAll always returns Direct regardless of topology
    auto topo = TopologyBuilder()
        .addNode("node0", 4, NPUType::ASCEND_910B)
        .addNode("node1", 4, NPUType::ASCEND_910B)
        .connectNodes("node0", "node1", LinkType::ROCE, 100.0, 0.001)
        .build();

    AlgorithmSelector selector;
    auto* algo = selector.SelectWithTopology(PrimitiveType::ALL_TO_ALL,
                                              4096, 8, topo);
    EXPECT_NE(algo, nullptr);
    EXPECT_STREQ(algo->Name(), "AlltoAllDirect");
}

TEST(AlgorithmSelectorTest, TopoSmallDataThresholdAt64KB) {
    // Exactly 64KB -> still small -> Ring
    auto topo = TopologyBuilder()
        .addNode("node0", 8, NPUType::ASCEND_910B)
        .build();

    AlgorithmSelector selector;
    auto* algo = selector.SelectWithTopology(PrimitiveType::ALL_REDUCE,
                                              64 * 1024, 8, topo);
    EXPECT_NE(algo, nullptr);
    EXPECT_STREQ(algo->Name(), "AllReduceRing");

    // 64KB + 1 -> medium/large -> RHD (single node, power-of-2, >4MB would be RHD)
    // but 64KB+1 is still under 4MB so Ring
    algo = selector.SelectWithTopology(PrimitiveType::ALL_REDUCE,
                                        64 * 1024 + 1, 8, topo);
    EXPECT_NE(algo, nullptr);
    EXPECT_STREQ(algo->Name(), "AllReduceRing");
}
