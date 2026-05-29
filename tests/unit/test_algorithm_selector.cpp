#include <gtest/gtest.h>
#include <algorithm>
#include "algorithm/selector/algorithm_selector.h"

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
