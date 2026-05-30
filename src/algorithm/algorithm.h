#pragma once

#include "common/types.h"
#include "common/error.h"
#include "simulator/channel/channel.h"
#include "simulator/topology/topology.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <cstring>
#include <functional>

namespace cann {

// Communication context — provides rank, nranks, and channel access
class CommContext {
public:
    CommContext(uint32_t rank, uint32_t nranks, IChannel& channel)
        : rank_(rank), nranks_(nranks), channel_(channel) {}

    uint32_t rank() const { return rank_; }
    uint32_t nranks() const { return nranks_; }
    IChannel& channel() { return channel_; }

    void send(const void* data, size_t bytes, uint32_t dst_rank) {
        channel_.send(data, bytes, dst_rank);
    }

    void recv(void* buffer, size_t bytes, uint32_t src_rank) {
        channel_.recv(buffer, bytes, src_rank);
    }

    void barrier() {
        channel_.barrier();
    }

private:
    uint32_t rank_;
    uint32_t nranks_;
    IChannel& channel_;
};

// Performance profile for an algorithm run
struct AlgorithmProfile {
    std::string name;
    size_t bytes_transferred;
    double time_ms;
    int num_steps;
    double bandwidth_gbps;
};

// Algorithm base class
class Algorithm {
public:
    virtual ~Algorithm() = default;

    virtual Status Execute(void* sendbuf, void* recvbuf, size_t count,
                           HCCLDataType dtype, HCCLReduceOp op,
                           CommContext& ctx) = 0;

    virtual const char* Name() const = 0;

    virtual int NumSteps(uint32_t nranks) const = 0;
};

// Helper: apply reduce operation on raw bytes (supports FLOAT32, FLOAT16, BF16, INT32)
inline void ReduceBuffer(void* dst, const void* src, size_t count,
                         HCCLDataType dtype, HCCLReduceOp op) {
    if (dtype == HCCLDataType::FLOAT32) {
        float* d = static_cast<float*>(dst);
        const float* s = static_cast<const float*>(src);
        for (size_t i = 0; i < count; i++) {
            d[i] = ApplyReduceOp(op, d[i], s[i]);
        }
    } else if (dtype == HCCLDataType::FLOAT16) {
        uint16_t* d = static_cast<uint16_t*>(dst);
        const uint16_t* s = static_cast<const uint16_t*>(src);
        for (size_t i = 0; i < count; i++) {
            float a = fp16_to_float(d[i]);
            float b = fp16_to_float(s[i]);
            d[i] = float_to_fp16(ApplyReduceOp(op, a, b));
        }
    } else if (dtype == HCCLDataType::BFLOAT16) {
        uint16_t* d = static_cast<uint16_t*>(dst);
        const uint16_t* s = static_cast<const uint16_t*>(src);
        for (size_t i = 0; i < count; i++) {
            float a = bf16_to_float(d[i]);
            float b = bf16_to_float(s[i]);
            d[i] = float_to_bf16(ApplyReduceOp(op, a, b));
        }
    } else if (dtype == HCCLDataType::INT32) {
        int32_t* d = static_cast<int32_t*>(dst);
        const int32_t* s = static_cast<const int32_t*>(src);
        for (size_t i = 0; i < count; i++) {
            switch (op) {
                case HCCLReduceOp::SUM:  d[i] += s[i]; break;
                case HCCLReduceOp::PROD: d[i] *= s[i]; break;
                case HCCLReduceOp::MAX:  d[i] = (d[i] > s[i]) ? d[i] : s[i]; break;
                case HCCLReduceOp::MIN:  d[i] = (d[i] < s[i]) ? d[i] : s[i]; break;
            }
        }
    } else if (dtype == HCCLDataType::INT8) {
        int8_t* d = static_cast<int8_t*>(dst);
        const int8_t* s = static_cast<const int8_t*>(src);
        for (size_t i = 0; i < count; i++) {
            switch (op) {
                case HCCLReduceOp::SUM:  d[i] += s[i]; break;
                case HCCLReduceOp::PROD: d[i] *= s[i]; break;
                case HCCLReduceOp::MAX:  d[i] = (d[i] > s[i]) ? d[i] : s[i]; break;
                case HCCLReduceOp::MIN:  d[i] = (d[i] < s[i]) ? d[i] : s[i]; break;
            }
        }
    } else if (dtype == HCCLDataType::UINT8) {
        uint8_t* d = static_cast<uint8_t*>(dst);
        const uint8_t* s = static_cast<const uint8_t*>(src);
        for (size_t i = 0; i < count; i++) {
            switch (op) {
                case HCCLReduceOp::SUM:  d[i] += s[i]; break;
                case HCCLReduceOp::PROD: d[i] *= s[i]; break;
                case HCCLReduceOp::MAX:  d[i] = (d[i] > s[i]) ? d[i] : s[i]; break;
                case HCCLReduceOp::MIN:  d[i] = (d[i] < s[i]) ? d[i] : s[i]; break;
            }
        }
    }
}

inline size_t DataSize(size_t count, HCCLDataType dtype) {
    return count * GetDataTypeSize(dtype);
}

} // namespace cann
