#include "algorithm/allreduce/allreduce_ring.h"
#include <cstring>
#include <vector>

namespace cann {

Status AllReduceRing::Execute(void* sendbuf, void* recvbuf, size_t count,
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

    // When count < nranks, the ring algorithm's chunking produces empty
    // chunks that prevent correct reduction. Fall back to a simple
    // all-to-all reduce: each rank sends its ORIGINAL data (not the
    // intermediate reduced value) to all other ranks.
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

    // Standard ring allreduce for count >= nranks
    std::vector<uint8_t> tmp(total_bytes);

    // Chunk size — divide count evenly among nranks.
    // The last chunk gets any remainder.
    size_t chunk_elems = count / nranks;
    size_t last_chunk_elems = count - chunk_elems * (nranks - 1);

    auto chunkInfo = [&](int chunk_idx) -> std::pair<size_t, size_t> {
        if (chunk_idx < static_cast<int>(nranks - 1)) {
            return {chunk_idx * chunk_elems, chunk_elems};
        }
        return {chunk_idx * chunk_elems, last_chunk_elems};
    };

    // === Phase 1: Reduce-Scatter ===
    // In step s, rank r sends chunk (r-s) and receives chunk (r-s-1).
    // After nranks-1 steps, rank r has chunk (r+1)%nranks fully reduced.
    for (uint32_t step = 0; step < nranks - 1; step++) {
        int send_chunk = (rank - step + nranks) % nranks;
        int recv_chunk = (rank - step - 1 + nranks) % nranks;

        auto [send_off, send_len] = chunkInfo(send_chunk);
        auto [recv_off, recv_len] = chunkInfo(recv_chunk);

        uint32_t send_to = (rank + 1) % nranks;
        uint32_t recv_from = (rank - 1 + nranks) % nranks;

        uint8_t* buf = static_cast<uint8_t*>(recvbuf);

        ctx.send(buf + send_off * elem_size, send_len * elem_size, send_to);
        ctx.recv(tmp.data(), recv_len * elem_size, recv_from);

        if (recv_len > 0) {
            ReduceBuffer(buf + recv_off * elem_size, tmp.data(), recv_len, dtype, op);
        }
    }

    // === Phase 2: AllGather ===
    // After reduce-scatter, rank r has chunk (r+1)%nranks fully reduced.
    // In step s, rank r sends chunk (r+1-s) and receives chunk (r-s),
    // propagating the fully reduced chunks around the ring.
    for (uint32_t step = 0; step < nranks - 1; step++) {
        int send_chunk = (rank + 1 - step + nranks) % nranks;
        int recv_chunk = (rank - step + nranks) % nranks;

        auto [send_off, send_len] = chunkInfo(send_chunk);
        auto [recv_off, recv_len] = chunkInfo(recv_chunk);

        uint32_t send_to = (rank + 1) % nranks;
        uint32_t recv_from = (rank - 1 + nranks) % nranks;

        uint8_t* buf = static_cast<uint8_t*>(recvbuf);

        ctx.send(buf + send_off * elem_size, send_len * elem_size, send_to);
        ctx.recv(buf + recv_off * elem_size, recv_len * elem_size, recv_from);
    }

    return Status::SUCCESS;
}

} // namespace cann
