# Example: Pipeline AllReduce Design

## Problem
Design AllReduce for 8 ranks on a single node, using pipelining to overlap Reduce-Scatter and AllGather phases for maximum throughput.

## Design

### Algorithm Name: AllReducePipeline
### Algorithm Type: Pipeline (Ring with overlapped phases)

### Step-by-step Description

**Key Idea: Overlap Reduce-Scatter and AllGather**
Standard Ring AllReduce completes Reduce-Scatter before starting AllGather.
Pipeline AllReduce overlaps them: as soon as a chunk is reduced in Reduce-Scatter, it starts the AllGather for that chunk.

**Phase: Pipelined Ring (2*(N-1) steps, but overlapped)**
1. Divide input data into K chunks (K >> N, e.g., K = 4*N or more)
2. For step s in 0 to 2*(N-1)-1:
   - **Reduce-Scatter work**: If s < N-1, send and receive chunk for Reduce-Scatter
   - **AllGather work**: If s >= N-1, send and receive chunk for AllGather
   - Both operations happen in the same step, using different buffers
3. Each step does one send and one receive (full-duplex link utilization)

**Pipeline Stages:**
```
Step:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14
RS:   [c0][c1][c2][c3][c4][c5][c6]                                 (steps 0-6)
AG:            [c0][c1][c2][c3][c4][c5][c6]                         (steps 2-8, overlapped)
```

### Complexity Analysis
- Steps: 2*(N-1) total, but overlapped
- Effective latency: N-1 + 1 steps (first RS finishes, last AG finishes)
- Data per step: K/N elements per chunk
- Total data moved per rank: 2*(N-1)/N * input_size (same as Ring)
- Key benefit: **latency hiding** -- computation/communication overlap

### Pseudocode
```
AllReducePipeline(sendbuf, recvbuf, count, nranks, rank):
    // Split into many small chunks for pipelining
    num_chunks = nranks * 4  // 4x oversubscription for smooth pipeline
    chunk_size = count / num_chunks

    copy sendbuf to recvbuf

    for step in 0 to 2 * (nranks - 1) - 1:
        // Reduce-Scatter phase (first N-1 steps)
        if step < nranks - 1:
            rs_send_chunk = (rank - step) % nranks
            rs_recv_chunk = (rank - step - 1) % nranks
            send recvbuf[rs_send_chunk * chunk_size : (rs_send_chunk+1) * chunk_size] to (rank+1) % nranks
            recv tmp from (rank-1) % nranks
            recvbuf[rs_recv_chunk * chunk_size : (rs_recv_chunk+1) * chunk_size] += tmp

        // AllGather phase (last N-1 steps, overlapped)
        if step >= nranks - 1:
            ag_step = step - (nranks - 1)
            ag_send_chunk = (rank + 1 - ag_step) % nranks
            ag_recv_chunk = (rank - ag_step) % nranks
            send recvbuf[ag_send_chunk * chunk_size : (ag_send_chunk+1) * chunk_size] to (rank+1) % nranks
            recv recvbuf[ag_recv_chunk * chunk_size : (ag_recv_chunk+1) * chunk_size] from (rank-1) % nranks
```

### Expected Performance
- For 1GB data on 8 ranks: ~18ms (vs ~20ms for non-pipelined Ring)
- Improvement: 10-15% for large messages (latency hiding)
- Best improvement when computation can be interleaved with communication

### When to Use Pipeline AllReduce
- **Large messages**: Pipeline overhead is amortized over many chunks
- **Computation overlap**: When reduce operations can run concurrently with communication
- **Full-duplex links**: HCCS supports simultaneous send/receive
- **Avoid when**: Very small messages (overhead of many chunks hurts), or ranks are not contiguous in the ring

### Chunk Size Strategy
- **Too few chunks**: Poor overlap, pipeline stalls
- **Too many chunks**: Per-chunk overhead dominates, cache thrashing
- **Sweet spot**: 4*N to 16*N chunks, where N = nranks
- **Adaptive**: Start with coarse chunks, refine based on message size

### Comparison with Standard Ring
| Metric | Ring | Pipeline Ring |
|--------|------|---------------|
| Steps | 2*(N-1) | 2*(N-1) (overlapped) |
| Effective latency | 2*(N-1)*T_msg | (N-1)*T_msg + T_msg |
| Bandwidth | Same | Same |
| Latency hiding | None | Good for large msgs |
| Implementation | Simple | Moderate |
