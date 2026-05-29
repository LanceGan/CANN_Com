#include "algorithm/reduce_scatter/reduce_scatter_butterfly.h"
#include <cstring>
#include <vector>

namespace cann {

Status ReduceScatterButterfly::Execute(void* sendbuf, void* recvbuf, size_t count,
                                       HCCLDataType dtype, HCCLReduceOp op,
                                       CommContext& ctx) {
    uint32_t rank = ctx.rank();
    uint32_t nranks = ctx.nranks();
    size_t elem_size = GetDataTypeSize(dtype);

    // Single rank: just copy
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

    // Check if nranks is a power of 2
    if ((nranks & (nranks - 1)) != 0) {
        // Non-power-of-2 fallback: simple chunk copy
        // Each rank just gets its own chunk unreduced
        size_t chunk_bytes = chunk_elems * elem_size;
        std::memcpy(recvbuf, static_cast<uint8_t*>(sendbuf) + rank * chunk_bytes, chunk_bytes);
        return Status::SUCCESS;
    }

    size_t chunk_bytes = chunk_elems * elem_size;
    size_t total_bytes = count * elem_size;

    // Working buffer: copy sendbuf
    std::vector<uint8_t> work(total_bytes);
    std::memcpy(work.data(), sendbuf, total_bytes);

    // tmp must hold the largest half-exchange: nranks/2 chunks at step 0
    std::vector<uint8_t> tmp((nranks / 2) * chunk_bytes);

    uint32_t logN = 0;
    {
        uint32_t n = nranks;
        while (n > 1) { n >>= 1; logN++; }
    }

    // Butterfly ReduceScatter: log2(N) steps.
    // We process bits from MSB to LSB so that rank r ends up owning chunk r.
    // At each step, the working region is split into two halves. Based on the
    // current bit, each rank keeps one half and sends the other to its partner,
    // then reduces the received data into its kept half.
    size_t offset = 0;          // start of working region (in chunks)
    size_t wrk_size = nranks;   // working region size (in chunks)

    for (uint32_t k = 0; k < logN; k++) {
        uint32_t bit_idx = logN - 1 - k;    // process bits MSB first
        uint32_t partner = rank ^ (1 << bit_idx);
        uint32_t bit = (rank >> bit_idx) & 1;
        size_t half_size = wrk_size / 2;     // chunks in each half

        size_t send_off, recv_off;
        if (bit == 0) {
            // Keep lower half, send upper half
            send_off = offset + half_size;
            recv_off = offset;
        } else {
            // Keep upper half, send lower half
            send_off = offset;
            recv_off = offset + half_size;
        }

        // Exchange half the data with partner
        ctx.send(work.data() + send_off * chunk_bytes, half_size * chunk_bytes, partner);
        ctx.recv(tmp.data(), half_size * chunk_bytes, partner);

        // Reduce received data into kept half
        ReduceBuffer(work.data() + recv_off * chunk_bytes, tmp.data(),
                     half_size * chunk_elems, dtype, op);

        // Update working region
        if (bit == 1) {
            offset = offset + half_size;
        }
        wrk_size = half_size;
    }

    // After logN steps, rank r owns chunk r (at offset, which equals rank).
    std::memcpy(recvbuf, work.data() + offset * chunk_bytes, chunk_bytes);

    return Status::SUCCESS;
}

} // namespace cann
