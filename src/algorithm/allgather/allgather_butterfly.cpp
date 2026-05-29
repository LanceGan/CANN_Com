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
    // In step k, each rank exchanges all accumulated data with partner = rank ^ (1 << k).
    // After each step, each rank doubles the number of chunks it has.
    int num_steps = NumSteps(nranks);
    std::vector<uint8_t> send_buf(nranks * chunk_bytes);

    for (int step = 0; step < num_steps; step++) {
        uint32_t partner = rank ^ (1 << step);

        // Copy current accumulated data to send buffer
        std::memcpy(send_buf.data(), out, nranks * chunk_bytes);

        // Send all accumulated data to partner
        ctx.send(send_buf.data(), nranks * chunk_bytes, partner);

        // Receive partner's accumulated data
        std::vector<uint8_t> recv_tmp(nranks * chunk_bytes);
        ctx.recv(recv_tmp.data(), nranks * chunk_bytes, partner);

        // Merge: copy chunks from partner's buffer that we don't have
        for (uint32_t i = 0; i < nranks; i++) {
            // Check if this chunk is present in partner's buffer but not in ours
            // After step k, rank has chunks where bits 0..k match its own bits 0..k
            // Partner differs in bit step, so partner has chunks where bit step is different
            // We need chunks where bit step matches partner's bit (i.e., differs from ours)
            if (((i >> step) & 1) != ((rank >> step) & 1)) {
                std::memcpy(out + i * chunk_bytes, recv_tmp.data() + i * chunk_bytes, chunk_bytes);
            }
        }
    }

    return Status::SUCCESS;
}

} // namespace cann
