#include "algorithm/allreduce/allreduce_pipeline.h"
#include <cstring>
#include <vector>
#include <algorithm>

namespace cann {

Status AllReducePipeline::Execute(void* sendbuf, void* recvbuf, size_t count,
                                   HCCLDataType dtype, HCCLReduceOp op,
                                   CommContext& ctx) {
    uint32_t rank = ctx.rank();
    uint32_t nranks = ctx.nranks();
    size_t elem_size = GetDataTypeSize(dtype);
    size_t total_bytes = count * elem_size;

    if (nranks <= 1) {
        if (sendbuf != recvbuf) {
            std::memcpy(recvbuf, sendbuf, total_bytes);
        }
        return Status::SUCCESS;
    }

    if (count == 0) {
        return Status::SUCCESS;
    }

    // Copy sendbuf to recvbuf (we operate on recvbuf)
    if (sendbuf != recvbuf) {
        std::memcpy(recvbuf, sendbuf, total_bytes);
    }

    // When count < nranks, the ring pattern's chunking produces empty chunks.
    // Fall back to simple all-to-all reduce.
    if (count < nranks) {
        std::vector<uint8_t> tmp(total_bytes);
        for (uint32_t other = 0; other < nranks; other++) {
            if (other == rank) continue;
            ctx.send(static_cast<const uint8_t*>(sendbuf), total_bytes, other);
            ctx.recv(tmp.data(), total_bytes, other);
            ReduceBuffer(recvbuf, tmp.data(), count, dtype, op);
        }
        return Status::SUCCESS;
    }

    // Ensure pipeline_depth does not exceed the number of elements
    int depth = std::min(pipeline_depth_, static_cast<int>(count));

    // Divide data into pipeline_depth sub-chunks.
    // Each sub-chunk is independently processed through a full ring allreduce
    // (Reduce-Scatter + AllGather). The pipeline benefit comes from overlapping
    // communication of one sub-chunk with computation of the next.
    size_t chunk_elems = count / depth;
    size_t last_chunk_elems = count - chunk_elems * (depth - 1);

    auto chunkInfo = [&](int idx) -> std::pair<size_t, size_t> {
        if (idx < depth - 1) {
            return {static_cast<size_t>(idx) * chunk_elems, chunk_elems};
        }
        return {static_cast<size_t>(idx) * chunk_elems, last_chunk_elems};
    };

    // Temporary buffer for receives (reused across sub-chunks)
    std::vector<uint8_t> tmp(total_bytes);

    uint8_t* buf = static_cast<uint8_t*>(recvbuf);

    // Ring neighbor indices
    uint32_t send_to = (rank + 1) % nranks;
    uint32_t recv_from = (rank - 1 + nranks) % nranks;

    // === Pipeline: process each sub-chunk through a full ring allreduce ===
    //
    // Each sub-chunk is divided into nranks mini-chunks. The ring reduce-scatter
    // reduces the mini-chunks so each rank holds one fully-reduced mini-chunk.
    // The ring allgather then distributes the reduced mini-chunks to all ranks.

    for (int c = 0; c < depth; c++) {
        auto [sub_off, sub_len] = chunkInfo(c);

        // Check if this sub-chunk is large enough for ring pattern
        if (sub_len < nranks) {
            // Sub-chunk too small for ring: use all-to-all reduce
            std::vector<uint8_t> orig(sub_len * elem_size);
            std::memcpy(orig.data(), buf + sub_off * elem_size, sub_len * elem_size);
            for (uint32_t other = 0; other < nranks; other++) {
                if (other == rank) continue;
                ctx.send(orig.data(), sub_len * elem_size, other);
                ctx.recv(tmp.data(), sub_len * elem_size, other);
                ReduceBuffer(buf + sub_off * elem_size, tmp.data(), sub_len, dtype, op);
            }
            continue;
        }

        // Divide sub-chunk into nranks mini-chunks
        size_t mini_elems = sub_len / nranks;
        size_t last_mini_elems = sub_len - mini_elems * (nranks - 1);

        auto miniInfo = [&](int idx) -> std::pair<size_t, size_t> {
            if (idx < static_cast<int>(nranks - 1)) {
                return {sub_off + static_cast<size_t>(idx) * mini_elems, mini_elems};
            }
            return {sub_off + static_cast<size_t>(idx) * mini_elems, last_mini_elems};
        };

        // === Phase 1: Reduce-Scatter for sub-chunk c ===
        // In step s, rank r sends mini-chunk (r-s) and receives mini-chunk (r-s-1),
        // reducing the received data into its local buffer.
        // After nranks-1 steps, each rank has one fully-reduced mini-chunk.
        for (uint32_t step = 0; step < nranks - 1; step++) {
            int send_mc = (static_cast<int>(rank) - static_cast<int>(step)
                          + static_cast<int>(nranks)) % static_cast<int>(nranks);
            int recv_mc = (static_cast<int>(rank) - static_cast<int>(step) - 1
                          + static_cast<int>(nranks)) % static_cast<int>(nranks);

            auto [s_off, s_len] = miniInfo(send_mc);
            auto [r_off, r_len] = miniInfo(recv_mc);

            ctx.send(buf + s_off * elem_size, s_len * elem_size, send_to);
            ctx.recv(tmp.data(), r_len * elem_size, recv_from);
            ReduceBuffer(buf + r_off * elem_size, tmp.data(), r_len, dtype, op);
        }

        // === Phase 2: AllGather for sub-chunk c ===
        // After reduce-scatter, rank r has fully-reduced mini-chunk (r+1)%nranks.
        // Propagate the reduced mini-chunks around the ring.
        // After nranks-1 steps, all ranks have all mini-chunks fully reduced.
        for (uint32_t step = 0; step < nranks - 1; step++) {
            int send_mc = (static_cast<int>(rank) + 1 - static_cast<int>(step)
                          + static_cast<int>(nranks)) % static_cast<int>(nranks);
            int recv_mc = (static_cast<int>(rank) - static_cast<int>(step)
                          + static_cast<int>(nranks)) % static_cast<int>(nranks);

            auto [s_off, s_len] = miniInfo(send_mc);
            auto [r_off, r_len] = miniInfo(recv_mc);

            ctx.send(buf + s_off * elem_size, s_len * elem_size, send_to);
            ctx.recv(buf + r_off * elem_size, r_len * elem_size, recv_from);
        }
    }

    return Status::SUCCESS;
}

} // namespace cann
