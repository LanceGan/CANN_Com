// f:/Projects/CANN_Com/src/algorithm/allgather/allgather_ring.cpp
#include "algorithm/allgather/allgather_ring.h"
#include <cstring>
#include <vector>

namespace cann {

Status AllGatherRing::Execute(void* sendbuf, void* recvbuf, size_t count,
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

    // Place own data into recvbuf at the correct offset
    uint8_t* out = static_cast<uint8_t*>(recvbuf);
    std::memcpy(out + rank * chunk_bytes, sendbuf, chunk_bytes);

    // Ring AllGather: N-1 steps
    // In step s, rank r sends chunk (r-s) mod N and receives chunk (r-s-1) mod N.
    // This propagates each rank's data around the ring.
    for (uint32_t step = 0; step < nranks - 1; step++) {
        int send_chunk = (rank - step + nranks) % nranks;
        int recv_chunk = (rank - step - 1 + nranks) % nranks;

        uint32_t send_to = (rank + 1) % nranks;
        uint32_t recv_from = (rank - 1 + nranks) % nranks;

        ctx.send(out + send_chunk * chunk_bytes, chunk_bytes, send_to);
        ctx.recv(out + recv_chunk * chunk_bytes, chunk_bytes, recv_from);
    }

    return Status::SUCCESS;
}

} // namespace cann
