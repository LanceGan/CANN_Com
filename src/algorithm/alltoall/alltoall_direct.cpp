// f:/Projects/CANN_Com/src/algorithm/alltoall/alltoall_direct.cpp
#include "algorithm/alltoall/alltoall_direct.h"
#include <cstring>
#include <vector>

namespace cann {

Status AlltoAllDirect::Execute(void* sendbuf, void* recvbuf, size_t count,
                                HCCLDataType dtype, HCCLReduceOp op,
                                CommContext& ctx) {
    uint32_t rank = ctx.rank();
    uint32_t nranks = ctx.nranks();
    size_t elem_size = GetDataTypeSize(dtype);

    // AlltoAll: total buffer has `count` elements divided into `nranks` chunks.
    // Each rank sends chunk `i` to rank `i`, and receives chunk `rank` from each peer.
    size_t chunk_count = count / nranks;
    size_t chunk_bytes = chunk_count * elem_size;
    size_t total_bytes = count * elem_size;

    if (nranks <= 1) {
        std::memcpy(recvbuf, sendbuf, total_bytes);
        return Status::SUCCESS;
    }

    if (count == 0) {
        return Status::SUCCESS;
    }

    const uint8_t* send = static_cast<const uint8_t*>(sendbuf);
    uint8_t* recv = static_cast<uint8_t*>(recvbuf);

    // Direct exchange: send each chunk to its destination, receive from sources.
    // Send to all other ranks first, then receive from all.
    for (uint32_t peer = 0; peer < nranks; peer++) {
        if (peer == rank) continue;
        ctx.send(send + peer * chunk_bytes, chunk_bytes, peer);
    }

    for (uint32_t peer = 0; peer < nranks; peer++) {
        if (peer == rank) {
            std::memcpy(recv + peer * chunk_bytes, send + peer * chunk_bytes, chunk_bytes);
        } else {
            ctx.recv(recv + peer * chunk_bytes, chunk_bytes, peer);
        }
    }

    return Status::SUCCESS;
}

} // namespace cann
