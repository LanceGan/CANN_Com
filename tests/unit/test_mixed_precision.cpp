#include <gtest/gtest.h>
#include "common/types.h"
#include "algorithm/algorithm.h"
#include "algorithm/allreduce/allreduce_ring.h"
#include "simulator/simulator.h"
#include "simulator/topology/topology_builder.h"
#include <vector>
#include <thread>
#include <cmath>

using namespace cann;

TEST(FP16Test, ConversionRoundTrip) {
    // Test that float -> fp16 -> float preserves value within precision
    float values[] = {1.0f, -1.0f, 0.5f, 100.0f, 0.001f, 65504.0f};
    for (float v : values) {
        uint16_t h = float_to_fp16(v);
        float back = fp16_to_float(h);
        EXPECT_NEAR(back, v, std::abs(v) * 0.001f) << "Failed for " << v;
    }
}

TEST(FP16Test, ZeroAndInfinity) {
    uint16_t pos_zero = float_to_fp16(0.0f);
    EXPECT_EQ(fp16_to_float(pos_zero), 0.0f);

    uint16_t neg_zero = float_to_fp16(-0.0f);
    EXPECT_EQ(fp16_to_float(neg_zero), -0.0f);
}

TEST(FP16Test, ReduceBufferSum) {
    // Test FP16 reduce with SUM
    uint16_t a[4] = {float_to_fp16(1.0f), float_to_fp16(2.0f),
                     float_to_fp16(3.0f), float_to_fp16(4.0f)};
    uint16_t b[4] = {float_to_fp16(10.0f), float_to_fp16(20.0f),
                     float_to_fp16(30.0f), float_to_fp16(40.0f)};

    ReduceBuffer(a, b, 4, HCCLDataType::FLOAT16, HCCLReduceOp::SUM);

    EXPECT_NEAR(fp16_to_float(a[0]), 11.0f, 0.01f);
    EXPECT_NEAR(fp16_to_float(a[1]), 22.0f, 0.01f);
    EXPECT_NEAR(fp16_to_float(a[2]), 33.0f, 0.01f);
    EXPECT_NEAR(fp16_to_float(a[3]), 44.0f, 0.01f);
}

TEST(BF16Test, ConversionRoundTrip) {
    float values[] = {1.0f, -1.0f, 0.5f, 100.0f, 0.001f};
    for (float v : values) {
        uint16_t b = float_to_bf16(v);
        float back = bf16_to_float(b);
        EXPECT_NEAR(back, v, std::abs(v) * 0.01f) << "Failed for " << v;
    }
}

TEST(BF16Test, ReduceBufferSum) {
    uint16_t a[4] = {float_to_bf16(1.0f), float_to_bf16(2.0f),
                     float_to_bf16(3.0f), float_to_bf16(4.0f)};
    uint16_t b[4] = {float_to_bf16(10.0f), float_to_bf16(20.0f),
                     float_to_bf16(30.0f), float_to_bf16(40.0f)};

    ReduceBuffer(a, b, 4, HCCLDataType::BFLOAT16, HCCLReduceOp::SUM);

    EXPECT_NEAR(bf16_to_float(a[0]), 11.0f, 0.1f);
    EXPECT_NEAR(bf16_to_float(a[1]), 22.0f, 0.1f);
    EXPECT_NEAR(bf16_to_float(a[2]), 33.0f, 0.1f);
    EXPECT_NEAR(bf16_to_float(a[3]), 44.0f, 0.1f);
}

TEST(MixedPrecisionTest, AllReduceFP16) {
    uint32_t nranks = 4;
    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();

    Simulator sim(topo, SimMode::PureSim);
    PureSimChannel::clearMailbox();
    AllReduceRing algo;

    // Each rank has rank_id as FP16
    std::vector<std::vector<uint16_t>> outputs(nranks, std::vector<uint16_t>(1));
    std::vector<std::thread> threads;

    for (uint32_t r = 0; r < nranks; r++) {
        threads.emplace_back([&, r]() {
            uint16_t input = float_to_fp16(static_cast<float>(r));
            CommContext ctx(r, nranks, sim.getChannel(r));
            algo.Execute(&input, outputs[r].data(), 1,
                         HCCLDataType::FLOAT16, HCCLReduceOp::SUM, ctx);
        });
    }
    for (auto& t : threads) t.join();

    // Expected: 0+1+2+3 = 6.0
    float result = fp16_to_float(outputs[0][0]);
    EXPECT_NEAR(result, 6.0f, 0.01f);
}

TEST(MixedPrecisionTest, AllReduceBF16) {
    uint32_t nranks = 4;
    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();

    Simulator sim(topo, SimMode::PureSim);
    PureSimChannel::clearMailbox();
    AllReduceRing algo;

    std::vector<std::vector<uint16_t>> outputs(nranks, std::vector<uint16_t>(1));
    std::vector<std::thread> threads;

    for (uint32_t r = 0; r < nranks; r++) {
        threads.emplace_back([&, r]() {
            uint16_t input = float_to_bf16(static_cast<float>(r));
            CommContext ctx(r, nranks, sim.getChannel(r));
            algo.Execute(&input, outputs[r].data(), 1,
                         HCCLDataType::BFLOAT16, HCCLReduceOp::SUM, ctx);
        });
    }
    for (auto& t : threads) t.join();

    float result = bf16_to_float(outputs[0][0]);
    EXPECT_NEAR(result, 6.0f, 0.1f);
}
