#include <gtest/gtest.h>
#include "algorithm/allgather/allgather_ring.h"

using namespace cann;

TEST(AllGatherTest, Placeholder) {
    AllGatherRing algo;
    EXPECT_STREQ(algo.Name(), "AllGatherRing");
}
