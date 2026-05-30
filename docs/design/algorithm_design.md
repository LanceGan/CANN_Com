# CANN 分布式通信算法设计说明书

## 1. 概述
本项目实现了面向昇腾 NPU 集群的分布式集合通信算法，覆盖 5 种核心通信原语、9 种算法实现。

## 2. 已实现算法

### 2.1 AllReduce（梯度同步）
| 算法 | 步数 | 复杂度 | 适用场景 |
|------|------|--------|---------|
| Ring | 2*(N-1) | O(N) | 中等数据，单节点 |
| RHD | 2*log2(N) | O(logN) | 大数据，2的幂次节点数 |
| Pipeline | 2*(N-1) | O(N) | 大数据，流水线重叠 |

**Ring AllReduce**：分两阶段执行——Reduce-Scatter（N-1步）+ AllGather（N-1步）。每步每个 rank 向右邻居发送一个 chunk，从左邻居接收一个 chunk。带宽效率为 (N-1)/N。

**Recursive Halving-Doubling（RHD）**：分两阶段——Reduce-Scatter via Recursive Halving（log2(N)步）+ AllGather via Recursive Doubling（log2(N)步）。使用 XOR 计算通信伙伴：`partner = rank ^ distance`。步数更少但每次通信量更大。

**Pipeline AllReduce**：将数据分成多个 sub-chunk，每个 sub-chunk 独立执行 Ring AllReduce。通过 chunk 级流水线重叠通信与计算，提升大数据场景吞吐量。流水线深度可配置（默认 4）。

### 2.2 AllGather（参数聚合）
| 算法 | 步数 | 复杂度 | 适用场景 |
|------|------|--------|---------|
| Ring | N-1 | O(N) | 中等数据 |
| Butterfly | log2(N) | O(logN) | 大数据，2的幂次节点数 |

**Ring AllGather**：N-1 步，每步每个 rank 发送一个 chunk 并接收一个 chunk，数据沿环传播。

**Butterfly AllGather**：log2(N) 步，每步每个 rank 与 XOR 伙伴交换所有已累积数据。步数更少，适合延迟敏感场景。

### 2.3 ReduceScatter（分片归约）
| 算法 | 步数 | 复杂度 | 适用场景 |
|------|------|--------|---------|
| Ring | N-1 | O(N) | 中等数据 |
| Butterfly | log2(N) | O(logN) | 大数据，2的幂次节点数 |

**Ring ReduceScatter**：N-1 步，每步发送一个 chunk 并接收归约后的 chunk。

**Butterfly ReduceScatter**：log2(N) 步，从 MSB 到 LSB 处理 rank 位，每步与 XOR 伙伴交换并归约一半数据。

### 2.4 AlltoAll（全局交换）
| 算法 | 步数 | 复杂度 | 适用场景 |
|------|------|--------|---------|
| Direct | N-1 | O(N) | 小数据，低延迟网络 |

**Direct AlltoAll**：每个 rank 直接向所有其他 rank 发送数据。

### 2.5 Broadcast（参数广播）
| 算法 | 步数 | 复杂度 | 适用场景 |
|------|------|--------|---------|
| Ring | N-1 | O(N) | 所有场景 |

**Ring Broadcast**：Root rank 发送数据到下一个 rank，每个接收到数据的 rank 转发给下一个，N-1 步后所有 rank 拥有数据。使用 recv-first 顺序避免死锁。

## 3. 拓扑适配策略

### 3.1 单节点（Full Mesh HCCS）
- 所有算法均支持
- HCCS 带宽 ~100 GB/s，延迟 ~0.001ms
- Ring 算法在单节点上带宽效率接近理论峰值

### 3.2 多节点（ROCE 互联）
- 节点内使用 HCCS，节点间使用 ROCE
- 算法选择器根据拓扑自动选择最优算法

## 4. 算法选择策略

### 4.1 基础选择（Select）
- 数据 ≤ 4MB：选择 Ring 算法
- 数据 > 4MB 且节点数为 2 的幂次：选择 RHD/Butterfly 算法
- 其他情况：选择 Ring 算法

### 4.2 拓扑感知选择（SelectWithTopology）
- **小数据（≤64KB）**：始终选择 Ring（最低延迟）
- **多节点**：始终选择 Ring（适应 HCCS+ROCE 异构带宽）
- **单节点 + 大数据 + 2的幂次**：选择 RHD/Butterfly
- **单节点 + 非2的幂次**：选择 Ring

## 5. 混合精度支持

支持 FLOAT32、FLOAT16、BFLOAT16、INT32 数据类型：
- FP16：IEEE 754 半精度（1+5+10 位）
- BF16：Brain Floating Point（1+8+7 位）
- 归约操作时自动转换为 float 进行计算，再转回目标格式

## 6. 可靠性机制

- **故障注入**：FaultChannel 装饰器支持链路故障、超时、数据损坏模拟
- **流量控制**：并发发送数限制，防止网络拥塞
- **超时重传**：发送/接收超时后自动重试，最大重试次数可配置
- **错误恢复**：链路故障抛出异常，上层可捕获并处理
