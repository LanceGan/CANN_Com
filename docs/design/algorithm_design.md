# CANN 分布式通信算法设计说明书

## 1. 概述
本项目实现了面向昇腾 NPU 集群的分布式集合通信算法，覆盖 5 种核心通信原语。

## 2. 已实现算法

### 2.1 AllReduce（梯度同步）
| 算法 | 步数 | 复杂度 | 适用场景 |
|------|------|--------|---------|
| Ring | 2*(N-1) | O(N) | 中等数据，单节点 |
| RHD | 2*log2(N) | O(logN) | 大数据，2的幂次节点数 |

**Ring AllReduce**：分两阶段执行——Reduce-Scatter（N-1步）+ AllGather（N-1步）。每步每个 rank 向右邻居发送一个 chunk，从左邻居接收一个 chunk。带宽效率为 (N-1)/N。

**Recursive Halving-Doubling**：分两阶段——Reduce-Scatter via Recursive Halving（log2(N)步）+ AllGather via Recursive Doubling（log2(N)步）。使用 XOR 计算通信伙伴：`partner = rank ^ distance`。步数更少但每次通信量更大。

### 2.2 AllGather（参数聚合）
| 算法 | 步数 | 复杂度 | 适用场景 |
|------|------|--------|---------|
| Ring | N-1 | O(N) | 中等数据 |

**Ring AllGather**：N-1 步，每步每个 rank 发送一个 chunk 并接收一个 chunk，数据沿环传播。

### 2.3 ReduceScatter（分片归约）
| 算法 | 步数 | 复杂度 | 适用场景 |
|------|------|--------|---------|
| Ring | N-1 | O(N) | 中等数据 |

**Ring ReduceScatter**：N-1 步，每步发送一个 chunk 并接收归约后的 chunk。

### 2.4 AlltoAll（全局交换）
| 算法 | 步数 | 复杂度 | 适用场景 |
|------|------|--------|---------|
| Direct | N-1 | O(N) | 小数据，低延迟网络 |

**Direct AlltoAll**：每个 rank 直接向所有其他 rank 发送数据。

## 3. 拓扑适配策略

### 3.1 单节点（Full Mesh HCCS）
- 所有算法均支持
- HCCS 带宽 ~100 GB/s，延迟 ~0.001ms
- Ring 算法在单节点上带宽效率接近理论峰值

### 3.2 多节点（ROCE 互联）
- 节点内使用 HCCS，节点间使用 ROCE
- 算法选择器根据数据量自动选择最优算法

## 4. 算法选择策略
- 数据 ≤ 4MB：选择 Ring 算法
- 数据 > 4MB 且节点数为 2 的幂次：选择 RHD 算法
- 其他情况：选择 Ring 算法
