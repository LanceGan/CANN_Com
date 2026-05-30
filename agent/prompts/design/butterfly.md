# Example: Butterfly AllGather Design

## Problem
Design AllGather for 8 ranks using a Butterfly (recursive doubling) pattern on a single node (Full Mesh HCCS).

## Design

### Algorithm Name: AllGatherButterfly
### Algorithm Type: Butterfly (Recursive Doubling)

### Step-by-step Description

**Phase 1: Recursive Doubling (log2(N) steps)**
1. Divide input data into N equal chunks, each rank starts with one chunk
2. In step s (0 to log2(N)-1):
   - Distance d = 2^s
   - Rank r exchanges chunk (r XOR d) with rank (r XOR d)
   - Rank r sends its current data for chunk (r XOR d) to partner
   - Rank r receives chunk (r XOR d) from partner
3. After log2(N) steps, all ranks have the complete gathered result

**Key Property: Butterfly Pattern**
- Each rank communicates with a partner at distance 2^s in step s
- Total data exchanged doubles each step (chunk size grows)
- All ranks participate in every step (no idle ranks)

### Complexity Analysis
- Steps: log2(N) = 3 for 8 ranks
- Step 0: exchange N/2 chunks, each of size count/N
- Step 1: exchange N/2 chunks, each of size 2*count/N
- Step 2: exchange N/2 chunks, each of size 4*count/N
- Total data moved per rank: 2 * (N-1)/N * input_size
- Latency: O(log N) messages (vs O(N) for Ring)

### Pseudocode
```
AllGatherButterfly(sendbuf, recvbuf, count, nranks, rank):
    chunk_size = count / nranks
    copy sendbuf to recvbuf[rank * chunk_size]

    // Recursive Doubling
    for step in 0 to log2(nranks) - 1:
        d = 1 << step
        partner = rank ^ d
        send_offset = (rank ^ d) * chunk_size
        recv_offset = (rank ^ d) * chunk_size

        // Each step doubles the amount of data exchanged
        exchange_size = chunk_size * d
        send_offset = ((rank >> step) << step) * chunk_size
        recv_offset = ((partner >> step) << step) * chunk_size

        send recvbuf[send_offset:send_offset+exchange_size] to partner
        recv tmp[0:exchange_size] from partner
        recvbuf[recv_offset:recv_offset+exchange_size] = tmp
```

### Expected Performance
- For 1GB data on 8 ranks: ~12ms (at 100 GB/s HCCS)
- Latency: 3 steps (vs 7 for Ring) -- significant advantage for small messages
- Bandwidth: ~87.5% of peak (same total data as Ring, but fewer steps)

### When to Use Butterfly
- **Small to medium messages**: log(N) latency steps dominate
- **Low-latency networks**: HCCS intra-node where latency matters
- **Power-of-2 ranks**: Butterfly works best with 2^k ranks
- **Avoid when**: Large messages on bandwidth-limited links (Ring is simpler)

### Comparison with Ring AllGather
| Metric | Ring | Butterfly |
|--------|------|-----------|
| Steps | N-1 | log2(N) |
| Data per step | count/N | grows (doubles) |
| Total data moved | (N-1)/N * size | (N-1)/N * size |
| Latency (msgs) | N-1 | log2(N) |
| Implementation | Simple | Moderate |
