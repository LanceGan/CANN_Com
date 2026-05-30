// f:/Projects/CANN_Com/tests/unit/test_correctness_rigorous.cpp
// Rigorous correctness tests: boundary conditions, precision drift, randomized testing.

#include <gtest/gtest.h>
#include "algorithm/allreduce/allreduce_ring.h"
#include "algorithm/allreduce/allreduce_rhd.h"
#include "algorithm/allgather/allgather_ring.h"
#include "algorithm/algorithm.h"
#include "simulator/simulator.h"
#include "simulator/topology/topology_builder.h"
#include "common/types.h"
#include <vector>
#include <thread>
#include <random>
#include <cmath>
#include <cstdint>

using namespace cann;

class RigorousCorrectnessTest : public ::testing::Test {
protected:
    void SetUp() override { PureSimChannel::clearMailbox(); }
    void TearDown() override { PureSimChannel::clearMailbox(); }
};

// ---------------------------------------------------------------------------
// 1. Boundary: count=1  -- AllReduce with 1 element across 4 and 8 ranks
// ---------------------------------------------------------------------------

TEST_F(RigorousCorrectnessTest, AllReduceSingleElement4Ranks) {
    uint32_t nranks = 4;
    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();
    Simulator sim(topo, SimMode::PureSim);
    AllReduceRing algo;

    std::vector<std::vector<float>> outputs(nranks, std::vector<float>(1));
    std::vector<std::thread> threads;
    for (uint32_t r = 0; r < nranks; r++) {
        threads.emplace_back([&, r]() {
            float input = static_cast<float>(r + 1);
            CommContext ctx(r, nranks, sim.getChannel(r));
            algo.Execute(&input, outputs[r].data(), 1,
                         HCCLDataType::FLOAT32, HCCLReduceOp::SUM, ctx);
        });
    }
    for (auto& t : threads) t.join();

    float expected = 0.0f;
    for (uint32_t i = 0; i < nranks; i++) expected += static_cast<float>(i + 1);
    for (uint32_t r = 0; r < nranks; r++) {
        EXPECT_FLOAT_EQ(outputs[r][0], expected)
            << "Rank " << r << " got " << outputs[r][0];
    }
}

TEST_F(RigorousCorrectnessTest, AllReduceSingleElement8Ranks) {
    uint32_t nranks = 8;
    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();
    Simulator sim(topo, SimMode::PureSim);
    AllReduceRing algo;

    std::vector<std::vector<float>> outputs(nranks, std::vector<float>(1));
    std::vector<std::thread> threads;
    for (uint32_t r = 0; r < nranks; r++) {
        threads.emplace_back([&, r]() {
            float input = static_cast<float>(r + 1);
            CommContext ctx(r, nranks, sim.getChannel(r));
            algo.Execute(&input, outputs[r].data(), 1,
                         HCCLDataType::FLOAT32, HCCLReduceOp::SUM, ctx);
        });
    }
    for (auto& t : threads) t.join();

    float expected = 0.0f;
    for (uint32_t i = 0; i < nranks; i++) expected += static_cast<float>(i + 1);
    for (uint32_t r = 0; r < nranks; r++) {
        EXPECT_FLOAT_EQ(outputs[r][0], expected)
            << "Rank " << r << " got " << outputs[r][0];
    }
}

// ---------------------------------------------------------------------------
// 2. Boundary: count < nranks  -- AllReduce with fewer elements than ranks
// ---------------------------------------------------------------------------

TEST_F(RigorousCorrectnessTest, AllReduceCountLessThanRanks) {
    uint32_t nranks = 8;
    size_t count = 2;
    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();
    Simulator sim(topo, SimMode::PureSim);
    AllReduceRing algo;

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

    float expected = 0.0f;
    for (uint32_t i = 0; i < nranks; i++) expected += static_cast<float>(i);
    for (uint32_t r = 0; r < nranks; r++) {
        for (size_t i = 0; i < count; i++) {
            EXPECT_FLOAT_EQ(outputs[r][i], expected)
                << "Rank " << r << " element " << i;
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Boundary: large data -- AllReduce with 1 MB of floats
// ---------------------------------------------------------------------------

TEST_F(RigorousCorrectnessTest, AllReduceLargeData1MB) {
    uint32_t nranks = 4;
    // 1 MB = 1 048 576 bytes = 262 144 floats
    size_t count = (1024 * 1024) / sizeof(float);

    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();
    Simulator sim(topo, SimMode::PureSim);
    AllReduceRing algo;

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

    // Spot-check a subset of elements to keep the test fast
    for (uint32_t r = 0; r < nranks; r++) {
        for (size_t i = 0; i < count; i += 1000) {
            EXPECT_FLOAT_EQ(outputs[r][i], expected_val)
                << "Rank " << r << " element " << i;
        }
        // Always check last element
        EXPECT_FLOAT_EQ(outputs[r][count - 1], expected_val)
            << "Rank " << r << " last element";
    }
}

// ---------------------------------------------------------------------------
// 4. Precision drift FLOAT32 -- Run AllReduce 100 times with fixed input,
//    verify output is deterministic (no bit-level drift across runs).
// ---------------------------------------------------------------------------

TEST_F(RigorousCorrectnessTest, AllReducePrecisionDriftFLOAT32) {
    uint32_t nranks = 4;
    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();

    float input_val = 3.14159f;
    float reference = 0.0f;  // set on the first iteration

    for (int iter = 0; iter < 100; iter++) {
        Simulator sim(topo, SimMode::PureSim);
        AllReduceRing algo;

        std::vector<std::vector<float>> outputs(nranks, std::vector<float>(1));
        std::vector<std::thread> threads;
        for (uint32_t r = 0; r < nranks; r++) {
            threads.emplace_back([&, r]() {
                CommContext ctx(r, nranks, sim.getChannel(r));
                algo.Execute(&input_val, outputs[r].data(), 1,
                             HCCLDataType::FLOAT32, HCCLReduceOp::SUM, ctx);
            });
        }
        for (auto& t : threads) t.join();

        // Expected: 4 * input_val
        if (iter == 0) {
            reference = outputs[0][0];
            EXPECT_FLOAT_EQ(reference, input_val * static_cast<float>(nranks));
        } else {
            EXPECT_FLOAT_EQ(outputs[0][0], reference)
                << "Output drifted at iteration " << iter;
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Precision drift FP16 -- Run AllReduce 100 times with fixed FP16 input,
//    verify output is deterministic across runs.
// ---------------------------------------------------------------------------

TEST_F(RigorousCorrectnessTest, AllReducePrecisionDriftFP16) {
    uint32_t nranks = 4;
    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();

    uint16_t input_val = float_to_fp16(3.14f);
    uint16_t reference = 0;

    for (int iter = 0; iter < 100; iter++) {
        Simulator sim(topo, SimMode::PureSim);
        AllReduceRing algo;

        std::vector<std::vector<uint16_t>> outputs(nranks, std::vector<uint16_t>(1));
        std::vector<std::thread> threads;
        for (uint32_t r = 0; r < nranks; r++) {
            threads.emplace_back([&, r]() {
                CommContext ctx(r, nranks, sim.getChannel(r));
                algo.Execute(&input_val, outputs[r].data(), 1,
                             HCCLDataType::FLOAT16, HCCLReduceOp::SUM, ctx);
            });
        }
        for (auto& t : threads) t.join();

        if (iter == 0) {
            reference = outputs[0][0];
            float expected_f = fp16_to_float(input_val) * static_cast<float>(nranks);
            EXPECT_NEAR(fp16_to_float(reference), expected_f, 0.05f);
        } else {
            EXPECT_EQ(outputs[0][0], reference)
                << "FP16 output drifted at iteration " << iter;
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Randomized AllReduce -- random inputs, verify sum matches expected
// ---------------------------------------------------------------------------

TEST_F(RigorousCorrectnessTest, RandomizedAllReduce) {
    uint32_t nranks = 4;
    size_t count = 100;
    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();
    Simulator sim(topo, SimMode::PureSim);
    AllReduceRing algo;

    // Generate random inputs
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    std::vector<std::vector<float>> inputs(nranks, std::vector<float>(count));
    for (uint32_t r = 0; r < nranks; r++) {
        for (size_t i = 0; i < count; i++) {
            inputs[r][i] = dist(rng);
        }
    }

    // Compute expected sum
    std::vector<float> expected(count, 0.0f);
    for (uint32_t r = 0; r < nranks; r++) {
        for (size_t i = 0; i < count; i++) {
            expected[i] += inputs[r][i];
        }
    }

    std::vector<std::vector<float>> outputs(nranks, std::vector<float>(count));
    std::vector<std::thread> threads;
    for (uint32_t r = 0; r < nranks; r++) {
        threads.emplace_back([&, r]() {
            CommContext ctx(r, nranks, sim.getChannel(r));
            algo.Execute(inputs[r].data(), outputs[r].data(), count,
                         HCCLDataType::FLOAT32, HCCLReduceOp::SUM, ctx);
        });
    }
    for (auto& t : threads) t.join();

    for (uint32_t r = 0; r < nranks; r++) {
        for (size_t i = 0; i < count; i++) {
            EXPECT_NEAR(outputs[r][i], expected[i], std::abs(expected[i]) * 1e-5f)
                << "Rank " << r << " element " << i;
        }
    }
}

// ---------------------------------------------------------------------------
// 7. Randomized AllGather -- random inputs, verify gathered output matches
// ---------------------------------------------------------------------------

TEST_F(RigorousCorrectnessTest, RandomizedAllGather) {
    uint32_t nranks = 4;
    size_t count = 50;
    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();
    Simulator sim(topo, SimMode::PureSim);
    AllGatherRing algo;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    std::vector<std::vector<float>> inputs(nranks, std::vector<float>(count));
    for (uint32_t r = 0; r < nranks; r++) {
        for (size_t i = 0; i < count; i++) {
            inputs[r][i] = dist(rng);
        }
    }

    std::vector<std::vector<float>> outputs(nranks, std::vector<float>(nranks * count));
    std::vector<std::thread> threads;
    for (uint32_t r = 0; r < nranks; r++) {
        threads.emplace_back([&, r]() {
            CommContext ctx(r, nranks, sim.getChannel(r));
            algo.Execute(inputs[r].data(), outputs[r].data(), count,
                         HCCLDataType::FLOAT32, HCCLReduceOp::SUM, ctx);
        });
    }
    for (auto& t : threads) t.join();

    for (uint32_t r = 0; r < nranks; r++) {
        for (uint32_t src = 0; src < nranks; src++) {
            for (size_t i = 0; i < count; i++) {
                EXPECT_FLOAT_EQ(outputs[r][src * count + i], inputs[src][i])
                    << "Rank " << r << " src " << src << " elem " << i;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 8. Mixed data types -- Test FLOAT32, FP16, BF16, INT32
// ---------------------------------------------------------------------------

TEST_F(RigorousCorrectnessTest, MixedDataTypesFLOAT32) {
    uint32_t nranks = 4;
    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();
    Simulator sim(topo, SimMode::PureSim);
    AllReduceRing algo;

    std::vector<std::vector<float>> outputs(nranks, std::vector<float>(1));
    std::vector<std::thread> threads;
    for (uint32_t r = 0; r < nranks; r++) {
        threads.emplace_back([&, r]() {
            float input = static_cast<float>(r + 1);
            CommContext ctx(r, nranks, sim.getChannel(r));
            algo.Execute(&input, outputs[r].data(), 1,
                         HCCLDataType::FLOAT32, HCCLReduceOp::SUM, ctx);
        });
    }
    for (auto& t : threads) t.join();

    float expected = 1.0f + 2.0f + 3.0f + 4.0f; // 10.0
    for (uint32_t r = 0; r < nranks; r++) {
        EXPECT_FLOAT_EQ(outputs[r][0], expected);
    }
}

TEST_F(RigorousCorrectnessTest, MixedDataTypesFP16) {
    uint32_t nranks = 4;
    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();
    Simulator sim(topo, SimMode::PureSim);
    AllReduceRing algo;

    std::vector<std::vector<uint16_t>> outputs(nranks, std::vector<uint16_t>(1));
    std::vector<std::thread> threads;
    for (uint32_t r = 0; r < nranks; r++) {
        threads.emplace_back([&, r]() {
            uint16_t input = float_to_fp16(static_cast<float>(r + 1));
            CommContext ctx(r, nranks, sim.getChannel(r));
            algo.Execute(&input, outputs[r].data(), 1,
                         HCCLDataType::FLOAT16, HCCLReduceOp::SUM, ctx);
        });
    }
    for (auto& t : threads) t.join();

    float expected = 10.0f; // 1+2+3+4
    for (uint32_t r = 0; r < nranks; r++) {
        float result = fp16_to_float(outputs[r][0]);
        EXPECT_NEAR(result, expected, 0.05f)
            << "Rank " << r << " FP16 result " << result;
    }
}

TEST_F(RigorousCorrectnessTest, MixedDataTypesBF16) {
    uint32_t nranks = 4;
    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();
    Simulator sim(topo, SimMode::PureSim);
    AllReduceRing algo;

    std::vector<std::vector<uint16_t>> outputs(nranks, std::vector<uint16_t>(1));
    std::vector<std::thread> threads;
    for (uint32_t r = 0; r < nranks; r++) {
        threads.emplace_back([&, r]() {
            uint16_t input = float_to_bf16(static_cast<float>(r + 1));
            CommContext ctx(r, nranks, sim.getChannel(r));
            algo.Execute(&input, outputs[r].data(), 1,
                         HCCLDataType::BFLOAT16, HCCLReduceOp::SUM, ctx);
        });
    }
    for (auto& t : threads) t.join();

    float expected = 10.0f; // 1+2+3+4
    for (uint32_t r = 0; r < nranks; r++) {
        float result = bf16_to_float(outputs[r][0]);
        EXPECT_NEAR(result, expected, 0.1f)
            << "Rank " << r << " BF16 result " << result;
    }
}

TEST_F(RigorousCorrectnessTest, MixedDataTypesINT32) {
    uint32_t nranks = 4;
    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();
    Simulator sim(topo, SimMode::PureSim);
    AllReduceRing algo;

    std::vector<std::vector<int32_t>> outputs(nranks, std::vector<int32_t>(1));
    std::vector<std::thread> threads;
    for (uint32_t r = 0; r < nranks; r++) {
        threads.emplace_back([&, r]() {
            int32_t input = static_cast<int32_t>(r + 1) * 100;
            CommContext ctx(r, nranks, sim.getChannel(r));
            algo.Execute(&input, outputs[r].data(), 1,
                         HCCLDataType::INT32, HCCLReduceOp::SUM, ctx);
        });
    }
    for (auto& t : threads) t.join();

    int32_t expected = 100 + 200 + 300 + 400; // 1000
    for (uint32_t r = 0; r < nranks; r++) {
        EXPECT_EQ(outputs[r][0], expected)
            << "Rank " << r << " INT32 result " << outputs[r][0];
    }
}
