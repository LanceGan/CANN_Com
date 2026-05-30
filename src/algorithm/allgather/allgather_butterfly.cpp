// f:/Projects/CANN_Com/src/algorithm/allgather/allgather_butterfly.cpp
#include "algorithm/allgather/allgather_butterfly.h"
#include "algorithm/allgather/allgather_ring.h"
#include <cstring>
#include <vector>

namespace cann {

Status AllGatherButterfly::Execute(void* sendbuf, void* recvbuf, size_t count,
                                    HCCLDataType dtype, HCCLReduceOp op,
                                    CommContext& ctx) {
    uint32_t rank = ctx.rank();
    uint32_t nranks = ctx.nranks();
    size_t elem_size = GetDataTypeSize(dtype);
    size_t chunk_bytes = count * elem_size;

    if (nranks <= 1) {
        std::memcpy(recvbuf, sendbuf, chunk_bytes);
        return Status::SUCCESS;
    }

    if (count == 0) {
        return Status::SUCCESS;
    }

    // For non-power-of-2 nranks, fall back to ring allgather
    if ((nranks & (nranks - 1)) != 0) {
        AllGatherRing ring;
        return ring.Execute(sendbuf, recvbuf, count, dtype, op, ctx);
    }

    // Place own data into recvbuf at the correct offset
    uint8_t* out = static_cast<uint8_t*>(recvbuf);
    std::memcpy(out + rank * chunk_bytes, sendbuf, chunk_bytes);

    // Butterfly AllGather: log2(N) steps
    // At step k, each rank sends only the 2^k chunks it has accumulated
    // to its partner rank ^ (1 << k), receiving 2^k new chunks in return.
    // This yields O(N) total bandwidth per rank instead of O(N log N).
    int num_steps = NumSteps(nranks);

    // Track which chunk indices we have accumulated so far
    std::vector<uint32_t> my_chunks;
    my_chunks.push_back(rank);

    // Send buffer sized for the largest transfer (nranks/2 chunks at the final step)
    std::vector<uint8_t> send_buf((nranks / 2) * chunk_bytes);

    for (int step = 0; step < num_steps; step++) {
        uint32_t partner = rank ^ (1 << step);
        uint32_t num_my = static_cast<uint32_t>(my_chunks.size());
        size_t transfer_bytes = num_my * chunk_bytes;

        // Pack our accumulated chunks into the send buffer
        for (uint32_t i = 0; i < num_my; i++) {
            std::memcpy(send_buf.data() + i * chunk_bytes,
                         out + my_chunks[i] * chunk_bytes, chunk_bytes);
        }

        // Send only our accumulated chunks (not the full nranks buffer)
        ctx.send(send_buf.data(), transfer_bytes, partner);

        // Receive partner's accumulated chunks (same count as ours)
        std::vector<uint8_t> recv_tmp(transfer_bytes);
        ctx.recv(recv_tmp.data(), transfer_bytes, partner);

        // Unpack: partner's i-th chunk corresponds to my_chunks[i] ^ (1 << step)
        std::vector<uint32_t> new_chunks;
        for (uint32_t i = 0; i < num_my; i++) {
            uint32_t chunk_idx = my_chunks[i] ^ (1 << step);
            std::memcpy(out + chunk_idx * chunk_bytes,
                         recv_tmp.data() + i * chunk_bytes, chunk_bytes);
            new_chunks.push_back(chunk_idx);
        }

        // Merge newly received chunk indices into our list
        my_chunks.insert(my_chunks.end(), new_chunks.begin(), new_chunks.end());
    }

    return Status::SUCCESS;
}

} // namespace cann
