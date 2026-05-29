#include <gtest/gtest.h>
#include "algorithm/selector/algorithm_selector.h"

using namespace cann;

TEST(AlgorithmSelectorTest, Placeholder) {
    AlgorithmSelector selector;
    // Selector returns nullptr for now (stub)
    EXPECT_EQ(selector.Select(HCCLReduceOp::SUM, 1024, 4), nullptr);
}
