# CANN 分布式通信算法设计与优化

基于昇腾 CANN 的分布式集合通信算法实现，依托智能 Agent 技术完成全流程设计与开发。

## 项目概述

本项目实现了面向昇腾 NPU 集群的分布式集合通信算法，覆盖 5 种核心通信原语、9 种算法实现，配套完整的 Agent 自动化系统和模拟器验证环境。

### 核心特性

- **9 种算法实现**：AllReduce（Ring/RHD/Pipeline）、AllGather（Ring/Butterfly）、ReduceScatter（Ring/Butterfly）、AlltoAll（Direct）、Broadcast（Ring）
- **拓扑感知调度**：根据 HCCS/ROCE 链路类型和拓扑结构自动选择最优算法
- **混合精度支持**：FLOAT32、FP16、BF16、INT32
- **通信模拟器**：NPU 拓扑模型、链路带宽/延迟/拥塞模拟、故障注入、流量控制、超时重传
- **智能 Agent 系统**：Design/Code/Test/Optimize 四个 Agent，支持迭代优化循环
- **算法选择器**：根据数据量、拓扑和链路类型自动选择最优算法
- **多节点验证**：支持 2-4 节点集群拓扑验证

## 项目结构

```
CANN_Com/
├── src/
│   ├── algorithm/              # 算法实现
│   │   ├── allreduce/          # AllReduce (Ring, RHD, Pipeline)
│   │   ├── allgather/          # AllGather (Ring, Butterfly)
│   │   ├── reduce_scatter/     # ReduceScatter (Ring, Butterfly)
│   │   ├── alltoall/           # AlltoAll (Direct)
│   │   ├── broadcast/          # Broadcast (Ring)
│   │   ├── selector/           # 算法选择器（含拓扑感知）
│   │   └── hccl_api/           # HCCL Plugin Interface
│   ├── simulator/              # 通信模拟器
│   │   ├── topology/           # NPU 拓扑模型
│   │   ├── network/            # 链路模型
│   │   └── channel/            # 通信信道 (PureSim, FaultChannel)
│   └── common/                 # 公共类型和工具（含 FP16/BF16 转换）
├── agent/                      # Agent 系统 (Python)
│   ├── agents/                 # 4 个核心 Agent
│   ├── context/                # 共享上下文（代码库索引 + 知识库）
│   ├── prompts/                # 8 个 Prompt 模板
│   └── orchestrator.py         # 编排器（含迭代优化循环）
├── tests/
│   ├── unit/                   # 单元测试 (20 个套件, 110 个测试)
│   ├── fault/                  # 可靠性测试
│   └── benchmark/              # 性能基准测试（支持多节点）
├── docs/                       # 文档
│   ├── design/                 # 算法设计说明书
│   ├── performance/            # 性能测试报告
│   ├── reliability/            # 可靠性报告
│   └── agent/                  # Agent 专项说明
└── CMakeLists.txt
```

## 快速开始

### 环境要求

- C++17 编译器 (GCC 14+ / MinGW)
- CMake 3.16+
- Python 3.10+
- Google Test (系统安装或 FetchContent)

### 构建

```bash
# 克隆项目
git clone <repo-url>
cd CANN_Com

# 构建 C++ 部分
bash scripts/build.sh

# 安装 Python 依赖
pip install -r requirements.txt
```

### 运行测试

```bash
# C++ 测试
cd build && ctest --output-on-failure

# Python 测试
python -m pytest agent/tests/ -v
```

### 运行基准测试

```bash
cd build

# 单节点基准测试
./bench_comm --nranks 8

# 多节点基准测试
./bench_comm --multinode --nodes 2 --ranks-per-node 4
```

### 运行 Agent

```bash
# Mock 模式（无需 API Key）
python -m agent --primitive AllReduce --nranks 8 --stages design code test optimize

# 使用真实 LLM
python -m agent --primitive AllGather --llm-provider anthropic --stages design code
```

## 算法矩阵

| 原语 | Ring | RHD | Pipeline | Butterfly |
|------|------|-----|----------|-----------|
| AllReduce | ✓ | ✓ | ✓ | — |
| AllGather | ✓ | — | — | ✓ |
| ReduceScatter | ✓ | — | — | ✓ |
| AlltoAll | ✓ (Direct) | — | — | — |
| Broadcast | ✓ | — | — | — |

### 算法选择策略

**基础选择（Select）**：
- 数据 ≤ 4MB：Ring 算法（带宽效率高）
- 数据 > 4MB + 2 的幂次节点：RHD/Butterfly 算法（步数少）
- 其他：Ring 算法

**拓扑感知选择（SelectWithTopology）**：
- 小数据（≤64KB）：始终选择 Ring（最低延迟）
- 多节点：始终选择 Ring（适应 HCCS+ROCE 异构带宽）
- 单节点 + 大数据 + 2的幂次：RHD/Butterfly

## 混合精度支持

| 数据类型 | 大小 | 说明 |
|---------|------|------|
| FLOAT32 | 4 字节 | 标准单精度浮点 |
| FLOAT16 | 2 字节 | IEEE 754 半精度（1+5+10 位） |
| BFLOAT16 | 2 字节 | Brain Floating Point（1+8+7 位） |
| INT32 | 4 字节 | 32 位整数 |

归约操作时自动转换为 float 进行计算，再转回目标格式。

## 可靠性机制

- **故障注入**：FaultChannel 支持链路故障、超时、数据损坏模拟
- **流量控制**：并发发送数限制，防止网络拥塞
- **超时重传**：发送/接收超时后自动重试，最大重试次数可配置
- **错误恢复**：链路故障抛出异常，上层可捕获并处理

## Agent 系统

### 能力清单 (Skills)

| Agent | 职责 | 输入 | 输出 |
|-------|------|------|------|
| Design Agent | 算法设计 | 原语类型、拓扑、性能目标 | 算法伪代码、复杂度分析 |
| Code Agent | 代码生成 | 算法设计文档 | .h/.cpp 文件 |
| Test Agent | 测试验证 | 算法实现代码 | Google Test 用例 |
| Optimize Agent | 性能优化 | 性能剖析数据 | 瓶颈分析、优化建议 |

### 工作流

```
需求输入 → Design Agent → 算法设计
         → Code Agent → C++ 实现
         → Test Agent → 测试用例
         → Optimize Agent → 性能优化
```

### 迭代优化

Agent 支持自动迭代优化循环：生成 → 编译 → 测试 → 分析失败 → 重新生成。

```bash
python -m agent --primitive AllReduce --nranks 8 --stages design code test --verbose
```

### Prompt 模板（8 个）

| 类别 | 模板 | 内容 |
|------|------|------|
| Design | algorithm_design.md | 算法设计通用模板 |
| Design | ring_allreduce.md | Ring AllReduce Few-shot 示例 |
| Design | butterfly.md | Butterfly AllGather Few-shot 示例 |
| Design | pipeline.md | Pipeline AllReduce Few-shot 示例 |
| Code | hccl_plugin_template.md | HCCL 插件代码模板 |
| Code | coding_standards.md | 编码规范 |
| Test | test_generation.md | 测试生成模板 |
| Optimize | optimization_patterns.md | 6 种优化模式 |

## 测试覆盖

- **110 个 C++ 测试**（20 个套件）：正确性、边界条件、多节点、故障注入、混合精度、精度漂移、随机化
- **17 个 Python 测试**：Agent 功能、编排器、代码库索引
- **9 种算法 × 6 种数据量** 基准测试（支持单节点和多节点模式）

## 文档

- [算法设计说明书](docs/design/algorithm_design.md)
- [性能测试报告](docs/performance/performance_report.md)
- [可靠性测试报告](docs/reliability/reliability_report.md)
- [Agent 专项说明](docs/agent/agent_specification.md)

## 技术栈

| 组件 | 选型 |
|------|------|
| 算法层 | C++17 |
| Agent 层 | Python 3.10+ |
| 构建系统 | CMake 3.16+ |
| 测试框架 | Google Test + pytest |
| LLM 调用 | Anthropic SDK / OpenAI SDK |

## 许可证

本项目为比赛参赛作品。
