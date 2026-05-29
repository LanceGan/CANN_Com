#include <gtest/gtest.h>
#include "algorithm/reduce_scatter/reduce_scatter_ring.h"

using namespace cann;

TEST(ReduceScatterTest, Placeholder) {
    ReduceScatterRing algo;
    EXPECT_STREQ(algo.Name(), "ReduceScatterRing");
}
