# CANN 分布式通信算法设计与优化

基于昇腾 CANN 的分布式集合通信算法实现，依托智能 Agent 技术完成全流程设计与开发。

## 项目概述

本项目实现了面向昇腾 NPU 集群的分布式集合通信算法，覆盖 4 种核心通信原语、7 种算法实现，配套完整的 Agent 自动化系统和模拟器验证环境。

### 核心特性

- **7 种算法实现**：AllReduce（Ring/RHD）、AllGather（Ring/Butterfly）、ReduceScatter（Ring/Butterfly）、AlltoAll（Direct）
- **通信模拟器**：NPU 拓扑模型、链路带宽/延迟/拥塞模拟、故障注入
- **智能 Agent 系统**：Design/Code/Test/Optimize 四个 Agent，支持迭代优化循环
- **算法选择器**：根据数据量和拓扑自动选择最优算法
- **多节点验证**：支持 2-4 节点集群拓扑验证

## 项目结构

```
CANN_Com/
├── src/
│   ├── algorithm/              # 算法实现
│   │   ├── allreduce/          # AllReduce (Ring, RHD)
│   │   ├── allgather/          # AllGather (Ring, Butterfly)
│   │   ├── reduce_scatter/     # ReduceScatter (Ring, Butterfly)
│   │   ├── alltoall/           # AlltoAll (Direct)
│   │   ├── selector/           # 算法选择器
│   │   └── hccl_api/           # HCCL Plugin Interface
│   ├── simulator/              # 通信模拟器
│   │   ├── topology/           # NPU 拓扑模型
│   │   ├── network/            # 链路模型
│   │   └── channel/            # 通信信道 (PureSim, FaultChannel)
│   └── common/                 # 公共类型和工具
├── agent/                      # Agent 系统 (Python)
│   ├── agents/                 # 4 个核心 Agent
│   ├── context/                # 共享上下文
│   ├── prompts/                # Prompt 模板
│   └── orchestrator.py         # 编排器
├── tests/
│   ├── unit/                   # 单元测试 (16 个套件, 69 个测试)
│   ├── fault/                  # 可靠性测试
│   └── benchmark/              # 性能基准测试
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
./bench_comm --nranks 8
```

### 运行 Agent

```bash
# Mock 模式（无需 API Key）
python -m agent --primitive AllReduce --nranks 8 --stages design code test optimize

# 使用真实 LLM
python -m agent --primitive AllGather --llm-provider anthropic --stages design code
```

## 算法矩阵

| 原语 | Ring | RHD | Butterfly |
|------|------|-----|-----------|
| AllReduce | ✓ | ✓ | — |
| AllGather | ✓ | — | ✓ |
| ReduceScatter | ✓ | — | ✓ |
| AlltoAll | ✓ (Direct) | — | — |

### 算法选择策略

- **数据 ≤ 4MB**：Ring 算法（带宽效率高）
- **数据 > 4MB + 2 的幂次节点**：RHD/Butterfly 算法（步数少）
- **其他**：Ring 算法

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

## 测试覆盖

- **69 个 C++ 测试**（16 个套件）：正确性、边界条件、多节点、故障注入
- **17 个 Python 测试**：Agent 功能、编排器、代码库索引
- **7 种算法 × 4 种数据量** 基准测试

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
