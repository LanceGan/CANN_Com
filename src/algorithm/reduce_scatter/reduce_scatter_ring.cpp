#include "algorithm/reduce_scatter/reduce_scatter_ring.h"
#include <cstring>
#include <vector>

namespace cann {

Status ReduceScatterRing::Execute(void* sendbuf, void* recvbuf, size_t count,
                                   HCCLDataType dtype, HCCLReduceOp op,
                                   CommContext& ctx) {
    uint32_t rank = ctx.rank();
    uint32_t nranks = ctx.nranks();
    size_t elem_size = GetDataTypeSize(dtype);

    if (nranks <= 1) {
        std::memcpy(recvbuf, sendbuf, count * elem_size);
        return Status::SUCCESS;
    }

    if (count == 0) {
        return Status::SUCCESS;
    }

    size_t chunk_elems = count / nranks;
    if (chunk_elems == 0) {
        // Fallback for count < nranks
        std::memcpy(recvbuf, sendbuf, count * elem_size);
        return Status::SUCCESS;
    }

    size_t chunk_bytes = chunk_elems * elem_size;

    // Working buffer: copy sendbuf
    std::vector<uint8_t> work(count * elem_size);
    std::memcpy(work.data(), sendbuf, count * elem_size);

    std::vector<uint8_t> tmp(chunk_bytes);

    // ReduceScatter Ring: N-1 steps
    // In step s, rank r sends chunk (r-1-s) mod N and receives chunk (r-2-s) mod N,
    // reducing received data into its working buffer.
    // After N-1 steps, rank r owns chunk r fully reduced.
    for (uint32_t step = 0; step < nranks - 1; step++) {
        int send_chunk = (rank - 1 - step + nranks) % nranks;
        int recv_chunk = (rank - 2 - step + nranks) % nranks;

        uint32_t send_to = (rank + 1) % nranks;
        uint32_t recv_from = (rank - 1 + nranks) % nranks;

        ctx.send(work.data() + send_chunk * chunk_bytes, chunk_bytes, send_to);
        ctx.recv(tmp.data(), chunk_bytes, recv_from);

        ReduceBuffer(work.data() + recv_chunk * chunk_bytes, tmp.data(),
                     chunk_elems, dtype, op);
    }

    // After N-1 steps, rank r owns chunk r fully reduced.
    std::memcpy(recvbuf, work.data() + rank * chunk_bytes, chunk_bytes);

    return Status::SUCCESS;
}

} // namespace cann
