# Example: Ring AllReduce Design

## Problem
Design AllReduce for 8 ranks on a single node (Full Mesh HCCS).

## Design

### Algorithm Name: AllReduceRing
### Algorithm Type: Ring

### Step-by-step Description

**Phase 1: Reduce-Scatter (N-1 steps)**
1. Divide input data into N equal chunks
2. In step s (0 to N-2):
   - Rank r sends chunk (r-s) mod N to rank (r+1) mod N
   - Rank r receives chunk (r-s-1) mod N from rank (r-1) mod N
   - Rank r reduces received data into its local chunk
3. After N-1 steps, each rank owns one fully-reduced chunk

**Phase 2: AllGather (N-1 steps)**
4. In step s (0 to N-2):
   - Rank r sends chunk (r+1-s) mod N to rank (r+1) mod N
   - Rank r receives chunk (r-s) mod N from rank (r-1) mod N
5. After N-1 steps, all ranks have the complete reduced result

### Complexity Analysis
- Steps: 2*(N-1)
- Data per step: count/N elements
- Total data moved per rank: 2*(N-1)/N * input_size
- Bandwidth efficiency: (N-1)/N (approaches 1.0 for large N)

### Pseudocode
```
AllReduceRing(sendbuf, recvbuf, count, nranks, rank):
    copy sendbuf to recvbuf
    chunk_size = count / nranks

    // Phase 1: Reduce-Scatter
    for step in 0 to nranks-2:
        send_chunk = (rank - step) % nranks
        recv_chunk = (rank - step - 1) % nranks
        send recvbuf[send_chunk] to (rank+1) % nranks
        recv tmp from (rank-1) % nranks
        recvbuf[recv_chunk] += tmp

    // Phase 2: AllGather
    for step in 0 to nranks-2:
        send_chunk = (rank + 1 - step) % nranks
        recv_chunk = (rank - step) % nranks
        send recvbuf[send_chunk] to (rank+1) % nranks
        recv recvbuf[recv_chunk] from (rank-1) % nranks
```

### Expected Performance
- For 1GB data on 8 ranks: ~20ms (at 100 GB/s HCCS)
- Bandwidth: ~87.5% of peak (7/8 efficiency)
