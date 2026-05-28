// f:/Projects/CANN_Com/tests/unit/test_topology.cpp
#include <gtest/gtest.h>
#include "simulator/topology/topology.h"
#include "simulator/topology/topology_builder.h"

using namespace cann;

TEST(NPUDeviceTest, Creation) {
    NPUDevice dev(0, NPUType::ASCEND_910B);
    EXPECT_EQ(dev.id(), 0);
    EXPECT_EQ(dev.type(), NPUType::ASCEND_910B);
}

TEST(TopologyBuilderTest, SingleNode) {
    auto topo = TopologyBuilder()
        .addNode("node0", 8, NPUType::ASCEND_910B)
        .build();

    EXPECT_EQ(topo.numNodes(), 1u);
    EXPECT_EQ(topo.numDevices(), 8u);
    EXPECT_EQ(topo.numRanks(), 8u);
    EXPECT_EQ(topo.nodeName(0), "node0");
}

TEST(TopologyBuilderTest, TwoNodesWithLink) {
    auto topo = TopologyBuilder()
        .addNode("node0", 8, NPUType::ASCEND_910B)
        .addNode("node1", 8, NPUType::ASCEND_910B)
        .connectNodes("node0", "node1", LinkType::ROCE, 100.0, 0.001)
        .build();

    EXPECT_EQ(topo.numNodes(), 2u);
    EXPECT_EQ(topo.numDevices(), 16u);
    EXPECT_EQ(topo.numRanks(), 16u);

    // Verify inter-node link exists
    auto links = topo.getLinks(0, 1); // node0 -> node1
    EXPECT_FALSE(links.empty());
    EXPECT_EQ(links[0].type, LinkType::ROCE);
    EXPECT_DOUBLE_EQ(links[0].bandwidth_gbps, 100.0);
}

TEST(TopologyBuilderTest, DeviceIdToRank) {
    auto topo = TopologyBuilder()
        .addNode("node0", 4, NPUType::ASCEND_910B)
        .addNode("node1", 4, NPUType::ASCEND_910B)
        .build();

    // node0: ranks 0-3, node1: ranks 4-7
    EXPECT_EQ(topo.rankToNodeId(0), 0u);
    EXPECT_EQ(topo.rankToNodeId(3), 0u);
    EXPECT_EQ(topo.rankToNodeId(4), 1u);
    EXPECT_EQ(topo.rankToNodeId(7), 1u);
}

TEST(TopologyBuilderTest, IntraNodeLinks) {
    auto topo = TopologyBuilder()
        .addNode("node0", 8, NPUType::ASCEND_910B)
        .build();

    // Intra-node links should exist (HCCS, Full Mesh)
    auto links = topo.getIntraNodeLinks(0);
    // 8 devices, Full Mesh: 8*7/2 = 28 links
    EXPECT_EQ(links.size(), 28u);
    EXPECT_EQ(links[0].type, LinkType::HCCS);
}
