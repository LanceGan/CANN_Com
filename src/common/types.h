#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace cann {

enum class HCCLDataType : uint8_t {
    FLOAT16 = 0,
    FLOAT32 = 1,
    INT32   = 2,
    INT8    = 3,
    UINT8   = 4,
    BFLOAT16 = 5,
};

enum class HCCLReduceOp : uint8_t {
    SUM  = 0,
    PROD = 1,
    MAX  = 2,
    MIN  = 3,
};

inline size_t GetDataTypeSize(HCCLDataType dtype) {
    switch (dtype) {
        case HCCLDataType::FLOAT16:   return 2;
        case HCCLDataType::FLOAT32:   return 4;
        case HCCLDataType::INT32:     return 4;
        case HCCLDataType::INT8:      return 1;
        case HCCLDataType::UINT8:     return 1;
        case HCCLDataType::BFLOAT16:  return 2;
        default:
            throw std::invalid_argument("Unknown HCCLDataType");
    }
}

inline float ApplyReduceOp(HCCLReduceOp op, float a, float b) {
    switch (op) {
        case HCCLReduceOp::SUM:  return a + b;
        case HCCLReduceOp::PROD: return a * b;
        case HCCLReduceOp::MAX:  return (a > b) ? a : b;
        case HCCLReduceOp::MIN:  return (a < b) ? a : b;
        default:
            throw std::invalid_argument("Unknown HCCLReduceOp");
    }
}

enum class LinkType : uint8_t {
    HCCS  = 0,
    PCIE  = 1,
    ROCE  = 2,
};

enum class NPUType : uint8_t {
    ASCEND_910A2 = 0,
    ASCEND_910A3 = 1,
    ASCEND_910B  = 2,
    ASCEND_910C  = 3,
};

enum class SimMode : uint8_t {
    PureSim    = 0,
    HCCLPlugin = 1,
};

} // namespace cann
