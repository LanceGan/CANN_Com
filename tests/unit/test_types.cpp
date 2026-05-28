#include <gtest/gtest.h>
#include "common/types.h"

using namespace cann;

TEST(HCCLDataTypeTest, SizeInBytes) {
    EXPECT_EQ(GetDataTypeSize(HCCLDataType::FLOAT16), 2u);
    EXPECT_EQ(GetDataTypeSize(HCCLDataType::FLOAT32), 4u);
    EXPECT_EQ(GetDataTypeSize(HCCLDataType::INT32), 4u);
    EXPECT_EQ(GetDataTypeSize(HCCLDataType::BFLOAT16), 2u);
}

TEST(HCCLReduceOpTest, ReduceOperations) {
    float a = 3.0f, b = 5.0f;
    EXPECT_FLOAT_EQ(ApplyReduceOp(HCCLReduceOp::SUM, a, b), 8.0f);
    EXPECT_FLOAT_EQ(ApplyReduceOp(HCCLReduceOp::PROD, a, b), 15.0f);
    EXPECT_FLOAT_EQ(ApplyReduceOp(HCCLReduceOp::MAX, a, b), 5.0f);
    EXPECT_FLOAT_EQ(ApplyReduceOp(HCCLReduceOp::MIN, a, b), 3.0f);
}
