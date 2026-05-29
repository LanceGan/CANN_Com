#include <gtest/gtest.h>
#include "algorithm/alltoall/alltoall_direct.h"

using namespace cann;

TEST(AlltoAllTest, Placeholder) {
    AlltoAllDirect algo;
    EXPECT_STREQ(algo.Name(), "AlltoAllDirect");
}
