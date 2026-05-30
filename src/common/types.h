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

// FP16 (IEEE 754 half-precision): 1 sign + 5 exponent + 10 mantissa
inline float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    if (exp == 0) {
        if (mantissa == 0) return sign ? -0.0f : 0.0f;
        // Denormalized
        float val = static_cast<float>(mantissa) / 1024.0f / 16.0f;
        return sign ? -val : val;
    }
    if (exp == 31) {
        if (mantissa == 0) return sign ? -__builtin_huge_valf() : __builtin_huge_valf();
        // NaN: preserve payload in float mantissa
        uint32_t f = (sign << 31) | 0x7F800000 | (mantissa << 13);
        float result;
        __builtin_memcpy(&result, &f, 4);
        return result;
    }

    uint32_t f = (sign << 31) | ((exp + 112) << 23) | (mantissa << 13);
    float result;
    __builtin_memcpy(&result, &f, 4);
    return result;
}

inline uint16_t float_to_fp16(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 31) & 1;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127;
    uint32_t mantissa = bits & 0x7FFFFF;

    if (exp > 15) return static_cast<uint16_t>((sign << 15) | 0x7C00); // Inf
    if (exp < -24) return static_cast<uint16_t>(sign << 15); // Zero
    if (exp < -14) {
        // Denormalized
        mantissa = (mantissa | 0x800000) >> (-exp - 14 + 13);
        return static_cast<uint16_t>((sign << 15) | mantissa);
    }
    return static_cast<uint16_t>((sign << 15) | ((exp + 15) << 10) | (mantissa >> 13));
}

// BF16 (Brain Floating Point): 1 sign + 8 exponent + 7 mantissa
inline float bf16_to_float(uint16_t b) {
    uint32_t f = static_cast<uint32_t>(b) << 16;
    float result;
    __builtin_memcpy(&result, &f, 4);
    return result;
}

inline uint16_t float_to_bf16(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, 4);
    // Round to nearest even
    uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
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
