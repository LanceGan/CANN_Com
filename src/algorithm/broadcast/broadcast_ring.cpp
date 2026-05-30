// f:/Projects/CANN_Com/src/algorithm/broadcast/broadcast_ring.cpp
#include "algorithm/broadcast/broadcast_ring.h"
#include <cstring>

namespace cann {

Status BroadcastRing::Execute(void* sendbuf, void* recvbuf, size_t count,
                               HCCLDataType dtype, HCCLReduceOp op,
                               CommContext& ctx) {
    uint32_t rank = ctx.rank();
    uint32_t nranks = ctx.nranks();
    size_t elem_size = GetDataTypeSize(dtype);
    size_t bytes = count * elem_size;

    if (nranks <= 1) {
        if (sendbuf != recvbuf) {
            std::memcpy(recvbuf, sendbuf, bytes);
        }
        return Status::SUCCESS;
    }

    if (count == 0) {
        return Status::SUCCESS;
    }

    uint32_t root = 0;
    uint32_t prev = (rank - 1 + nranks) % nranks;
    uint32_t next = (rank + 1) % nranks;

    if (rank == root) {
        // Root: copy own data to recvbuf, then send to next rank
        if (sendbuf != recvbuf) {
            std::memcpy(recvbuf, sendbuf, bytes);
        }
        ctx.send(sendbuf, bytes, next);
    } else {
        // Non-root: receive from previous rank first (avoids deadlock),
        // then forward to next rank unless this is the last rank in the ring
        ctx.recv(recvbuf, bytes, prev);
        if (rank != nranks - 1) {
            ctx.send(recvbuf, bytes, next);
        }
    }

    return Status::SUCCESS;
}

} // namespace cann
