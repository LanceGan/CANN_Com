# Optimization Patterns for Distributed Communication

## Overview
This document catalogs proven optimization patterns for NCCL/HCCL collective communication algorithms. Each pattern includes when to use it, implementation guidance, and expected impact.

---

## Pattern 1: Pipeline Communication

### Description
Overlap Reduce-Scatter and AllGather phases by pipelining chunks. As soon as a chunk is reduced, begin its AllGather while continuing to reduce other chunks.

### When to Use
- Large messages (>1MB) where latency hiding provides benefit
- Full-duplex links (HCCS, NVLink) that support simultaneous send/receive
- When computation can be interleaved with communication

### Implementation
```
1. Split message into K chunks (K = 4*N to 16*N)
2. For each step:
   - If in Reduce-Scatter phase: send/receive RS chunk
   - If in AllGather phase: send/receive AG chunk
   - Both can happen in the same step (different buffers)
```

### Expected Impact
- Latency reduction: 10-30% for large messages
- Bandwidth: unchanged (same total data movement)
- Overhead: minimal additional memory for pipeline buffers

### Example
```
Standard Ring:  [RS0][RS1][RS2][RS3][RS4][RS5][RS6] -> [AG0][AG1][AG2][AG3][AG4][AG5][AG6]
Pipeline Ring:  [RS0][RS1][RS2][RS3][RS4][RS5][RS6]
                      [AG0][AG1][AG2][AG3][AG4][AG5][AG6]  (overlapped)
```

---

## Pattern 2: Hierarchical Algorithms

### Description
Use a two-level algorithm: fast intra-node communication (HCCS) followed by slower inter-node communication (ROCE). This minimizes cross-node traffic.

### When to Use
- Multi-node setups with heterogeneous link speeds
- When intra-node bandwidth >> inter-node bandwidth (e.g., HCCS 100GB/s vs ROCE 25GB/s)
- For AllReduce, AllGather, ReduceScatter

### Implementation
```
For AllReduce:
1. Intra-node Reduce-Scatter (fast, local)
2. Inter-node AllReduce (one rank per node participates)
3. Intra-node Broadcast (fast, local)

For AllGather:
1. Intra-node AllGather (fast, local)
2. Inter-node AllGather (one rank per node)
3. Intra-node Broadcast (fast, local)
```

### Expected Impact
- Inter-node traffic: reduced by factor of N_nodes (only one rank per node communicates)
- Latency: may increase slightly due to extra coordination
- Bandwidth: much better utilization of fast HCCS links

### Example (8 nodes, 8 NPUs per node)
```
Step 1: Intra-node Reduce-Scatter (HCCS, fast)
        Each node: 8 ranks reduce to 1 result
Step 2: Inter-node AllReduce (ROCE, slower)
        8 node-representatives exchange data
Step 3: Intra-node Broadcast (HCCS, fast)
        Each node broadcasts to all 8 ranks
```

---

## Pattern 3: Topology-Aware Algorithm Selection

### Description
Select the communication algorithm based on the hardware topology, message size, and number of ranks. No single algorithm is optimal for all scenarios.

### Decision Matrix

| Scenario | Best Algorithm | Reason |
|----------|---------------|--------|
| Small msg, few ranks | Tree/Binary | Low latency, log(N) steps |
| Small msg, many ranks | Butterfly | Logarithmic steps |
| Large msg, single node | Ring | Bandwidth-optimal |
| Large msg, multi-node | Hierarchical Ring | Minimize cross-node |
| Latency-bound | Butterfly/Tree | Fewer steps |
| Bandwidth-bound | Ring/Pipeline | Maximum bandwidth utilization |

### Implementation
```
def select_algorithm(primitive, nranks, data_size, topology):
    if data_size < 1024:  # Small message
        if nranks <= 8:
            return "Tree"
        else:
            return "Butterfly"
    else:  # Large message
        if topology == "SingleNode":
            return "Ring"
        else:
            return "HierarchicalRing"
```

### Expected Impact
- 20-50% performance improvement over using a single algorithm for all cases
- Requires accurate topology detection and message size estimation

---

## Pattern 4: Chunk Size Optimization

### Description
Choose the optimal chunk size for splitting messages. Too small = per-chunk overhead dominates. Too large = poor latency hiding, cache thrashing.

### Guidelines

| Message Size | Chunk Size | Rationale |
|--------------|------------|-----------|
| < 1KB | No split | Overhead not worth it |
| 1KB - 1MB | 4KB - 64KB | Balance overhead vs latency hiding |
| 1MB - 100MB | 64KB - 1MB | Good pipelining |
| > 100MB | 1MB - 4MB | Maximize bandwidth, limit overhead |

### Implementation
```
def optimal_chunk_size(data_size, nranks):
    if data_size < 1024:
        return data_size  # No split
    elif data_size < 1024 * 1024:
        return max(4096, data_size // (nranks * 4))
    else:
        return min(4 * 1024 * 1024, data_size // (nranks * 8))
```

### Expected Impact
- 5-20% improvement from avoiding suboptimal chunk sizes
- Critical for pipeline algorithms (too few chunks = poor overlap)

---

## Pattern 5: ReduceScatter Butterfly

### Description
Butterfly ReduceScatter: logarithmic steps where each rank reduces data with a partner at distance 2^s. Complementary to AllGather Butterfly for full AllReduce.

### When to Use
- Small to medium messages where latency matters
- When paired with AllGatherButterfly for AllReduce
- Power-of-2 rank counts

### Implementation
```
ReduceScatterButterfly(sendbuf, recvbuf, count, nranks, rank):
    chunk_size = count / nranks
    for step in 0 to log2(nranks) - 1:
        d = 1 << step
        partner = rank ^ d
        exchange_size = chunk_size * d
        send sendbuf[local_chunk] to partner
        recv tmp from partner
        sendbuf[local_chunk] += tmp
    recvbuf = sendbuf[local_chunk]  // Scatter result
```

### Expected Impact
- Latency: O(log N) vs O(N) for Ring ReduceScatter
- Bandwidth: same total data movement
- Best for: messages < 1MB on HCCS

---

## Pattern 6: Broadcast Ring

### Description
Ring-based Broadcast: root sends data to next rank, which forwards to the next, etc. Simple and bandwidth-efficient for large messages.

### When to Use
- Single-node broadcast where simplicity matters
- Large messages (bandwidth-bound)
- When HCCS provides good ring connectivity

### Implementation
```
BroadcastRing(sendbuf, recvbuf, count, nranks, rank, root):
    if rank == root:
        send sendbuf to (rank+1) % nranks
    else:
        recv recvbuf from (rank-1) % nranks
        if rank != root:
            send recvbuf to (rank+1) % nranks
```

### Expected Impact
- Latency: O(N) messages (not ideal for small messages)
- Bandwidth: 100% utilization (each link used once)
- Simplicity: easy to implement and debug

---

## Pattern Summary

| Pattern | Best For | Latency | Bandwidth | Complexity |
|---------|----------|---------|-----------|------------|
| Pipeline | Large msgs | Low (hidden) | High | Medium |
| Hierarchical | Multi-node | Medium | High | Medium |
| Topology-Aware | All scenarios | Optimal | Optimal | High |
| Chunk Optimization | Pipeline algos | Varies | Improved | Low |
| Butterfly RS | Small msgs | Low | Medium | Medium |
| Broadcast Ring | Large msgs | High | High | Low |
