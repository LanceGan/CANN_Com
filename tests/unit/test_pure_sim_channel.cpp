#include <gtest/gtest.h>
#include "simulator/channel/pure_sim_channel.h"
#include "simulator/topology/topology_builder.h"

using namespace cann;

class PureSimChannelTest : public ::testing::Test {
protected:
    void SetUp() override {
        topo_ = TopologyBuilder()
            .addNode("node0", 4, NPUType::ASCEND_910B)
            .build();
        channel_ = std::make_unique<PureSimChannel>(topo_, /*rank=*/0);
    }

    Topology topo_;
    std::unique_ptr<PureSimChannel> channel_;
};

TEST_F(PureSimChannelTest, SendRecvData) {
    float send_buf[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float recv_buf[4] = {0.0f};

    // Send from rank 0 to rank 1
    channel_->send(send_buf, sizeof(send_buf), /*dst_rank=*/1);
    // Receive on rank 1 side (create a channel for rank 1)
    PureSimChannel recv_channel(topo_, 1);
    recv_channel.recv(recv_buf, sizeof(recv_buf), /*src_rank=*/0);

    // Data should be copied correctly
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(recv_buf[i], send_buf[i]);
    }
}

TEST_F(PureSimChannelTest, TransferTimeTracked) {
    float buf[1024] = {1.0f};

    channel_->send(buf, sizeof(buf), /*dst_rank=*/1);

    auto stats = channel_->getStats();
    EXPECT_GT(stats.total_send_time_ms, 0.0);
    EXPECT_EQ(stats.num_sends, 1u);
}

TEST_F(PureSimChannelTest, BarrierSync) {
    channel_->barrier();
    auto stats = channel_->getStats();
    EXPECT_EQ(stats.num_barriers, 1u);
}

TEST_F(PureSimChannelTest, LargeDataTransfer) {
    // 1 GB buffer
    size_t size = 1024 * 1024 * 1024ULL;
    std::vector<float> buf(size / sizeof(float), 1.0f);

    channel_->send(buf.data(), size, /*dst_rank=*/1);

    auto stats = channel_->getStats();
    // Should take measurable time (at least 10ms for 1GB at 100GB/s)
    EXPECT_GT(stats.total_send_time_ms, 5.0);
}
