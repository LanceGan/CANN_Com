# 优化方案设计文档

## 1. 概述

在已完成的四阶段基础上，继续在三个方向优化：
1. Butterfly 算法 — 提升算法创新性
2. Agent 迭代优化循环 — 提升 Agent 自动化能力
3. 多节点拓扑验证 — 提升测试覆盖面

## 2. Butterfly 算法

### 2.1 设计目标
为 AllGather 和 ReduceScatter 实现 Butterfly 算法，展示算法创新能力。

### 2.2 算法原理
Butterfly 算法使用 log2(N) 步完成集合通信。每步通信量为总数据量的 1/2。

**AllGather Butterfly**：
- 每步每个 rank 与距离为 N/2^step 的伙伴交换数据
- 使用 XOR 计算伙伴：`partner = rank ^ (1 << step)`
- log2(N) 步后，所有 rank 拥有完整数据

**ReduceScatter Butterfly**：
- 每步每个 rank 与伙伴交换并归约数据
- log2(N) 步后，每个 rank 拥有归约后的 chunk

### 2.3 与 RHD 的区别
| 特性 | RHD | Butterfly |
|------|-----|-----------|
| 适用原语 | AllReduce | AllGather/ReduceScatter |
| 步数 | 2*log2(N) | log2(N) |
| 通信模式 | Halving + Doubling | 直接 Butterfly 交换 |
| 数据分块 | 按 chunk 分阶段 | 每步交换一半数据 |

### 2.4 实现计划
- `src/algorithm/allgather/allgather_butterfly.h/.cpp`
- `src/algorithm/reduce_scatter/reduce_scatter_butterfly.h/.cpp`
- `tests/unit/test_allgather_butterfly.cpp`
- `tests/unit/test_reduce_scatter_butterfly.cpp`
- 更新算法选择器，添加 Butterfly 选项
- 更新基准测试，添加 Butterfly 对比

## 3. Agent 迭代优化循环

### 3.1 设计目标
Agent 生成算法后自动运行测试，根据失败结果迭代改进。

### 3.2 流程设计
```
Design Agent → Code Agent → Test Agent → 编译运行测试
    ↑                                        ↓
    ←←←←← 分析失败原因，重新生成 ←←←←←←←←←←
```

### 3.3 关键组件

**TestRunner**：
- 调用 `cmake --build . --target test_xxx`
- 调用 `ctest -R test_xxx --output-on-failure`
- 解析输出，提取编译错误/测试失败信息

**FailureAnalyzer**：
- 从编译输出提取错误行号和错误类型
- 从测试输出提取失败的断言和期望值
- 生成结构化的修复建议

**IterationController**：
- 控制最大迭代次数（默认 3 次）
- 每次迭代记录到 `agent/logs/iteration_N.json`
- 如果连续失败，停止并报告

### 3.4 Agent 增强

**Code Agent**：
- 增加 `previous_errors` 参数
- 接收上次编译/测试失败信息
- 在 Prompt 中注入失败上下文

**Test Agent**：
- 增加 `run_and_verify` 模式
- 自动生成测试 → 编译 → 运行 → 解析结果
- 返回结构化的测试结果

**Orchestrator**：
- 增加 `max_iterations` 参数
- 实现迭代循环逻辑
- 记录每次迭代的结果

### 3.5 日志格式
```json
{
  "iteration": 1,
  "stage": "code",
  "success": false,
  "errors": ["undefined reference to 'xxx'"],
  "fix_suggestion": "Add missing include for xxx.h"
}
```

## 4. 多节点拓扑验证

### 4.1 设计目标
在模拟器中配置 2-4 节点集群，验证跨节点通信算法正确性。

### 4.2 拓扑配置
```
Node 0 (8 NPU) ──ROCE── Node 1 (8 NPU)
    │                        │
   ROCE                     ROCE
    │                        │
Node 2 (8 NPU) ──ROCE── Node 3 (8 NPU)
```

### 4.3 验证内容
- AllReduce Ring/RHD 在 16 ranks（2 节点）上正确性
- AllGather/ReduceScatter 在多节点上正确性
- 跨节点 vs 节点内带宽差异对性能的影响

### 4.4 测试用例
- `test_multinode.cpp`：多节点正确性测试
  - 2 节点 16 ranks AllReduce
  - 2 节点 16 ranks AllGather
  - 4 节点 32 ranks AllReduce（如果性能允许）

### 4.5 基准测试更新
- 添加 `--nodes N` 参数
- 对比单节点 vs 多节点性能

## 5. 实施优先级

| 优先级 | 内容 | 预计工作量 |
|--------|------|-----------|
| 1 | Butterfly 算法 | 中等 |
| 2 | 多节点拓扑验证 | 较小 |
| 3 | Agent 迭代优化循环 | 中等 |

## 6. 预期收益

- **算法创新性**：Butterfly 算法展示更深层的算法理解
- **Agent 能力**：迭代优化循环展示 Agent 的自主改进能力
- **测试覆盖**：多节点验证展示算法的可扩展性
