// f:/Projects/CANN_Com/src/algorithm/allreduce/allreduce_rhd.cpp
#include "algorithm/allreduce/allreduce_rhd.h"
#include <cstring>
#include <vector>

namespace cann {

Status AllReduceRHD::Execute(void* sendbuf, void* recvbuf, size_t count,
                              HCCLDataType dtype, HCCLReduceOp op,
                              CommContext& ctx) {
    uint32_t rank = ctx.rank();
    uint32_t nranks = ctx.nranks();
    size_t elem_size = GetDataTypeSize(dtype);
    size_t total_bytes = count * elem_size;

    if (nranks <= 1) {
        if (sendbuf != recvbuf) std::memcpy(recvbuf, sendbuf, total_bytes);
        return Status::SUCCESS;
    }
    if (count == 0) return Status::SUCCESS;

    // Copy sendbuf to recvbuf (we operate on recvbuf)
    if (sendbuf != recvbuf) {
        std::memcpy(recvbuf, sendbuf, total_bytes);
    }

    int logN = 0;
    {
        uint32_t n = nranks;
        while (n > 1) { n >>= 1; logN++; }
    }

    // Fallback for non-power-of-2 or when data is too small to chunk
    if ((1u << logN) != nranks || count < nranks) {
        // Save original data before recvbuf gets modified by reductions
        std::vector<uint8_t> orig(total_bytes);
        std::memcpy(orig.data(), sendbuf, total_bytes);
        std::vector<uint8_t> tmp(total_bytes);
        for (uint32_t other = 0; other < nranks; other++) {
            if (other == rank) continue;
            ctx.send(orig.data(), total_bytes, other);
            ctx.recv(tmp.data(), total_bytes, other);
            ReduceBuffer(recvbuf, tmp.data(), count, dtype, op);
        }
        return Status::SUCCESS;
    }

    std::vector<uint8_t> tmp(total_bytes);
    size_t chunk_size = count / nranks;

    // Phase 1: Reduce-Scatter via Recursive Halving
    // At each step, paired ranks exchange data and reduce.
    // Each rank keeps one half of its current range and sends the other.
    // Ranks with the distance bit CLEAR keep the LOWER half.
    // Ranks with the distance bit SET keep the UPPER half.
    // Each rank reduces its KEPT half with the partner's SAME half.
    for (int step = 0; step < logN; step++) {
        uint32_t distance = nranks >> (step + 1);
        uint32_t partner = rank ^ distance;

        size_t half = distance * chunk_size * elem_size;
        if (half == 0) continue;

        // Offset into buffer for this rank's current range.
        // At step k, the buffer is divided into 2^k groups of (nranks/2^k) chunks.
        // This rank's group determines the base offset.
        size_t offset = 0;
        if (step > 0) {
            uint32_t group_bits = rank >> (logN - step);
            size_t group_chunks = nranks >> step;
            offset = static_cast<size_t>(group_bits) * group_chunks * chunk_size * elem_size;
        }

        uint8_t* buf = static_cast<uint8_t*>(recvbuf) + offset;

        if (rank & distance) {
            // Bit set: keeps upper half, sends lower half
            ctx.send(buf, half, partner);
            ctx.recv(tmp.data(), half, partner);
            ReduceBuffer(buf + half, tmp.data(), half / elem_size, dtype, op);
        } else {
            // Bit clear: keeps lower half, sends upper half
            ctx.send(buf + half, half, partner);
            ctx.recv(tmp.data(), half, partner);
            ReduceBuffer(buf, tmp.data(), half / elem_size, dtype, op);
        }
    }

    // Phase 2: AllGather via Recursive Doubling
    // Reverse the reduce-scatter communication pattern.
    // Each rank sends its reduced portion and receives the partner's reduced portion.
    for (int step = logN - 1; step >= 0; step--) {
        uint32_t distance = nranks >> (step + 1);
        uint32_t partner = rank ^ distance;

        size_t half = distance * chunk_size * elem_size;
        if (half == 0) continue;

        size_t offset = 0;
        if (step > 0) {
            uint32_t group_bits = rank >> (logN - step);
            size_t group_chunks = nranks >> step;
            offset = static_cast<size_t>(group_bits) * group_chunks * chunk_size * elem_size;
        }

        uint8_t* buf = static_cast<uint8_t*>(recvbuf) + offset;

        if (rank & distance) {
            // Bit set: sends upper half (its reduced data), receives into lower half
            ctx.send(buf + half, half, partner);
            ctx.recv(buf, half, partner);
        } else {
            // Bit clear: sends lower half (its reduced data), receives into upper half
            ctx.send(buf, half, partner);
            ctx.recv(buf + half, half, partner);
        }
    }

    return Status::SUCCESS;
}

} // namespace cann
