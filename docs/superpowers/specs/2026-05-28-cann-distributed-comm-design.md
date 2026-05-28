# CANN 分布式通信算法设计与优化 — 系统设计文档

## 1. 项目概述

### 1.1 比赛背景
赛题六：基于 CANN 的分布式通信算法设计与优化。面向昇腾 NPU 集群，依托智能 Agent 技术完成 HCCL 集合通信算法的全流程设计与代码开发。

### 1.2 项目约束
- **硬件环境**：仅模拟器，无实际昇腾 NPU 硬件
- **团队规模**：1 人
- **技术背景**：主要 ML 背景，分布式通信和系统开发经验较少
- **开发语言**：算法层 C/C++，Agent 层 Python
- **版本要求**：基于 CANN 8.0+，HCOMM 开源仓接口

### 1.3 核心交付物
- 4 种通信原语（AllReduce、AllGather、ReduceScatter、AlltoAll）的算法实现
- 双模式通信模拟器（PureSim / HCCL Plugin）
- 全流程 Agent 系统（Design / Code / Test / Optimize）
- 完整技术文档和演示材料

## 2. 整体架构

系统分为三层，每层有清晰的职责边界：

```
┌─────────────────────────────────────────────────┐
│                  Agent Layer                     │
│  ┌───────────┐ ┌───────────┐ ┌───────────────┐  │
│  │ Design    │ │  Code     │ │  Test         │  │
│  │ Agent     │ │  Agent    │ │  Agent        │  │
│  └───────────┘ └───────────┘ └───────────────┘  │
│  ┌───────────┐ ┌──────────────────────────────┐  │
│  │ Optimize  │ │    Prompt Engineering Hub    │  │
│  │ Agent     │ │    (模板库/Few-shot/CoT)     │  │
│  └───────────┘ └──────────────────────────────┘  │
├─────────────────────────────────────────────────┤
│               Algorithm Layer                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │AllReduce │ │AllGather │ │ReduceScatter     │ │
│  │(Ring/RHD │ │(Ring/Tree│ │AlltoAll          │ │
│  │ /NHR)    │ │ /Bucket) │ │(Direct/Pairwise) │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
│  ┌──────────────────────────────────────────┐    │
│  │      HCCL Plugin Interface (API层)       │    │
│  └──────────────────────────────────────────┘    │
├─────────────────────────────────────────────────┤
│               Simulator Layer                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │  NPU     │ │ Network  │ │  Communication   │ │
│  │ Topology │ │ Model    │ │  Channel         │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
│  ┌──────────────────────────────────────────┐    │
│  │      Dual Mode: PureSim / HCCL Plugin    │    │
│  └──────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

### 2.1 层间依赖
- **Simulator Layer**：最底层，不依赖上层。提供 NPU 拓扑、网络模型、通信信道的抽象。
- **Algorithm Layer**：依赖 Simulator Layer。通过 HCCL Plugin Interface 封装算法实现。
- **Agent Layer**：依赖 Algorithm Layer 和 Simulator Layer。读取代码和测试结果，生成/优化算法。

### 2.2 关键设计决策
1. **HCCL Plugin Interface** 作为算法层的统一 API，与昇腾官方 HCCL 接口对齐（hcclCommInit/hcclAllReduce 等），确保算法代码可无缝迁移到实机。
2. **Simulator 的双模式**通过策略模式实现——`PureSimBackend` 和 `HCCLPluginBackend` 实现同一接口。
3. **Agent 通过文件系统与代码交互**——读取代码、生成代码、运行测试、分析结果。

## 3. 模拟器层（Simulator Layer）

模拟器是整个系统的地基，需精准模拟昇腾 NPU 集群的通信特性。

### 3.1 NPU 拓扑模型

```
拓扑层级：
├── Node（单机）
│   ├── NPU Device × 8（昇腾 910B/910C）
│   ├── HCCS 链路（NPU 间高速互联，~100GB/s）
│   └── PCIe 链路（NPU-CPU，~32GB/s）
├── Switch（交换机）
│   └── ROCE 链路（机间互联，~100Gbps）
└── Cluster（集群）
    └── Node × N
```

- 每条链路有独立的**带宽、延迟、误码率**参数
- 支持配置不同拓扑：Full Mesh、Ring、Tree、Fat-Tree
- 支持异构场景：混合 910A2/910A3，非对称带宽

### 3.2 通信信道模拟

```
Channel 接口：
├── Send(dst, data, size)     → 模拟发送，按带宽/延迟计算完成时间
├── Recv(src, buffer, size)   → 模拟接收
├── Reduce(src, dst, op, type) → 模拟随路归约
└── Barrier()                  → 模拟同步屏障
```

- **时间模型**：`transfer_time = data_size / bandwidth + latency + congestion_delay`
- **拥塞模拟**：多对多通信时共享链路带宽按比例分配
- **故障注入**：支持链路断开、超时、数据损坏等故障场景

### 3.3 双模式切换

| 模式 | 用途 | 实现方式 |
|------|------|---------|
| PureSim | 纯算法验证，无需硬件 | 内存中模拟数据传输，计算时间戳 |
| HCCLPlugin | 端到端验证，对接真实接口 | 调用 HCCL API，通过 mock 层拦截硬件调用 |

### 3.4 模拟器 API 示例

```cpp
// 创建集群拓扑
Topology topo = TopologyBuilder()
    .addNode("node0", 8, NPUType::ASCEND_910B)
    .addNode("node1", 8, NPUType::ASCEND_910B)
    .connectNodes("node0", "node1", LinkType::ROCE, 100, "Gbps")
    .build();

// 创建模拟器实例
Simulator sim(topo, SimMode::PureSim);

// 运行通信算法
sim.runAlgorithm(allreduce_ring_algo, input_buf, output_buf, count, dtype);
auto stats = sim.getStats(); // 获取延迟、吞吐量、链路利用率
```

## 4. 算法层（Algorithm Layer）

算法层是核心竞争力建设的地方。采用策略模式，每个通信原语支持多种算法实现，可动态切换。

### 4.1 通信原语接口

与 HCCL 官方接口对齐：

```cpp
// HCCL Plugin Interface（对齐昇腾 HCCL API）
class HCCLComm {
    // 生命周期
    static Status Init(HCCLComm* comm, int ndev, int rank);
    static Status Destroy(HCCLComm comm);

    // 集合通信原语
    static Status AllReduce(void* sendbuf, void* recvbuf, size_t count,
                            HCCLDataType dtype, HCCLReduceOp op, HCCLComm comm);
    static Status AllGather(void* sendbuf, void* recvbuf, size_t count,
                            HCCLDataType dtype, HCCLComm comm);
    static Status ReduceScatter(void* sendbuf, void* recvbuf, size_t count,
                                HCCLDataType dtype, HCCLReduceOp op, HCCLComm comm);
    static Status AlltoAll(void* sendbuf, void* recvbuf, size_t count,
                           HCCLDataType dtype, HCCLComm comm);
};
```

### 4.2 算法实现矩阵

每个原语支持多种算法，按阶段递进实现：

| 原语 | Phase 1（经典） | Phase 2（优化） | Phase 3（创新） |
|------|----------------|----------------|----------------|
| **AllReduce** | Ring | Recursive Halving-Doubling | NHR（非均匀环）、自适应分块 |
| **AllGather** | Ring | Bucket / Tree | 动态 Butterfly |
| **ReduceScatter** | Ring | Recursive Halving-Doubling | 分块 Mesh |
| **AlltoAll** | Direct（全交换） | Pairwise Exchange | 分层 AlltoAll |

### 4.3 算法内部结构

每个算法实现统一的接口：

```cpp
// 算法基类
class Algorithm {
public:
    virtual ~Algorithm() = default;

    // 核心执行
    virtual Status Execute(void* sendbuf, void* recvbuf, size_t count,
                           HCCLDataType dtype, HCCLReduceOp op,
                           CommContext& ctx) = 0;

    // 性能分析（供 Agent 使用）
    virtual AlgorithmProfile Profile(size_t count, int nranks,
                                     const Topology& topo) const = 0;

    // 适用性判断
    virtual bool IsSuitable(size_t count, int nranks,
                            const Topology& topo) const = 0;

    virtual const char* Name() const = 0;
};
```

### 4.4 算法选择器（自适应调度）

```cpp
class AlgorithmSelector {
public:
    // 根据数据量、拓扑、硬件参数自动选择最优算法
    Algorithm* Select(PrimitiveType prim, size_t count,
                      int nranks, const Topology& topo);
};
```

选择策略：
- **小数据块**（≤64KB）：选择 Tree 或 Direct，减少启动开销
- **中等数据块**（64KB-256MB）：选择 Ring，带宽利用率高
- **大数据块**（≥256MB）：选择 Recursive Halving-Doubling 或 NHR，减少步数

## 5. Agent 层（Agent Layer）

Agent 是比赛的核心评审维度之一。设计为一个模块化的 Agent 系统，每个 Agent 负责一个明确的能力域。

### 5.1 Agent 架构

```
┌─────────────────────────────────────────────────┐
│              Agent Orchestrator                  │
│  （任务调度、Agent 间协调、状态管理）              │
├────────┬──────────┬──────────┬──────────────────┤
│ Design │  Code    │  Test    │  Optimize        │
│ Agent  │  Agent   │  Agent   │  Agent           │
├────────┴──────────┴──────────┴──────────────────┤
│              Shared Context Layer                │
│  （代码库索引、拓扑知识、HCCL API 文档、历史记录）  │
├─────────────────────────────────────────────────┤
│              Prompt Engineering Hub              │
│  （模板库、Few-shot 示例、Chain-of-Thought 模板） │
└─────────────────────────────────────────────────┘
```

### 5.2 四个核心 Agent

**1. Design Agent（算法设计 Agent）**
- **职责**：根据需求分析拓扑特征，设计算法方案
- **输入**：通信原语类型、拓扑参数、性能目标
- **输出**：算法伪代码、通信步数分析、理论复杂度
- **Prompt 策略**：
  - Chain-of-Thought：逐步推理通信模式
  - Few-shot：提供经典算法的设计思路作为示例
  - 约束注入：注入 HCCL API 能力边界和硬件限制

**2. Code Agent（代码生成 Agent）**
- **职责**：将算法设计转化为可编译的 C/C++ 代码
- **输入**：算法设计文档、HCCL API 接口定义、代码规范
- **输出**：.cpp/.h 文件、CMake 配置
- **Prompt 策略**：
  - 代码模板：提供 HCCL 插件的标准结构
  - 接口约束：强制使用 HCCL Plugin Interface
  - 自我检查：生成后自动检查编译错误

**3. Test Agent（测试验证 Agent）**
- **职责**：生成测试用例、运行测试、分析结果
- **输入**：算法实现代码、模拟器配置
- **输出**：测试用例、测试报告、正确性验证
- **Prompt 策略**：
  - 测试模板：标准的正确性/性能/边界测试框架
  - 故障场景：自动生成故障注入测试
  - 基准对比：与标准 Ring 算法对比结果

**4. Optimize Agent（性能优化 Agent）**
- **职责**：分析性能瓶颈、提出优化方案、生成优化后代码
- **输入**：性能剖析数据、当前算法代码
- **输出**：瓶颈分析报告、优化建议、优化后代码
- **Prompt 策略**：
  - 瓶颈模式库：常见通信瓶颈模式及解决方案
  - 优化知识：流水线、分块、重叠计算通信等技巧
  - A/B 对比：优化前后性能对比

### 5.3 Agent 工作流示例

以"AllReduce 创新算法"为例：

```
Design Agent:
  输入："为 8 机 64 卡集群设计 AllReduce 算法，拓扑为 Fat-Tree"
  → 分析拓扑 → 设计 NHR 算法方案 → 输出伪代码

Code Agent:
  输入：NHR 算法设计 + HCCL API 定义
  → 生成 AllReduceNHR.cpp → 检查编译

Test Agent:
  输入：AllReduceNHR.cpp + 模拟器配置
  → 生成测试用例 → 运行测试 → 输出报告

Optimize Agent:
  输入：性能报告 + 当前代码
  → 分析瓶颈（如：某条链路拥塞）→ 优化分块策略 → 输出优化后代码
```

### 5.4 Prompt 工程体系

```
agent/prompts/
├── design/
│   ├── algorithm_design.md      # 算法设计通用模板
│   ├── topology_analysis.md     # 拓扑分析模板
│   └── few_shot/
│       ├── ring_allreduce.md    # Ring 设计示例
│       └── butterfly.md         # Butterfly 设计示例
├── code/
│   ├── hccl_plugin_template.md  # HCCL 插件代码模板
│   ├── coding_standards.md      # 代码规范约束
│   └── self_check.md            # 代码自检清单
├── test/
│   ├── test_generation.md       # 测试用例生成模板
│   └── fault_injection.md       # 故障注入模板
└── optimize/
    ├── bottleneck_analysis.md   # 瓶颈分析模板
    └── optimization_patterns.md # 优化模式库
```

## 6. 项目结构

```
CANN_Com/
├── TASK.md                          # 比赛要求
├── CMakeLists.txt                   # 顶层构建配置
├── README.md                        # 项目说明
│
├── src/
│   ├── simulator/                   # 模拟器层
│   │   ├── topology/               # NPU 拓扑模型
│   │   │   ├── topology.h
│   │   │   ├── topology_builder.h
│   │   │   └── topology_builder.cpp
│   │   ├── network/                # 网络模型（带宽/延迟/拥塞）
│   │   │   ├── link_model.h
│   │   │   └── link_model.cpp
│   │   ├── channel/                # 通信信道
│   │   │   ├── channel.h
│   │   │   ├── pure_sim_channel.cpp
│   │   │   └── hccl_plugin_channel.cpp
│   │   └── simulator.h / .cpp      # 模拟器主类
│   │
│   ├── algorithm/                   # 算法层
│   │   ├── hccl_api/               # HCCL Plugin Interface
│   │   │   ├── hccl.h              # 官方接口定义
│   │   │   └── hccl_comm.cpp       # 通信器实现
│   │   ├── primitives/             # 通信原语
│   │   │   ├── allreduce/
│   │   │   │   ├── allreduce_ring.cpp
│   │   │   │   ├── allreduce_rhd.cpp
│   │   │   │   └── allreduce_nhr.cpp
│   │   │   ├── allgather/
│   │   │   ├── reduce_scatter/
│   │   │   └── alltoall/
│   │   └── selector/               # 自适应算法选择器
│   │       └── algorithm_selector.cpp
│   │
│   └── common/                      # 公共工具
│       ├── types.h                  # 数据类型定义
│       ├── error.h                  # 错误码
│       └── profiler.h              # 性能剖析工具
│
├── agent/                           # Agent 系统
│   ├── orchestrator/               # Agent 编排器
│   │   └── orchestrator.py
│   ├── agents/                     # 四个核心 Agent
│   │   ├── design_agent.py
│   │   ├── code_agent.py
│   │   ├── test_agent.py
│   │   └── optimize_agent.py
│   ├── prompts/                    # Prompt 工程
│   │   ├── design/
│   │   ├── code/
│   │   ├── test/
│   │   └── optimize/
│   └── context/                    # 共享上下文
│       ├── codebase_index.py       # 代码库索引
│       └── knowledge_base.py       # HCCL 知识库
│
├── tests/                           # 测试
│   ├── unit/                       # 单元测试
│   ├── integration/                # 集成测试
│   ├── benchmark/                  # 性能基准测试
│   │   ├── bench_allreduce.cpp
│   │   └── bench_allgather.cpp
│   └── fault/                      # 故障注入测试
│       └── fault_injection.cpp
│
├── scripts/                         # 工具脚本
│   ├── build.sh                    # 构建脚本
│   ├── run_tests.sh                # 测试运行
│   ├── run_benchmark.sh            # 压测脚本
│   └── generate_report.py          # 测试报告生成
│
├── docs/                            # 文档
│   ├── design/                     # 算法设计说明书
│   ├── performance/                # 性能测试报告
│   ├── reliability/                # 可靠性报告
│   └── agent/                      # Agent 专项说明
│
└── demo/                            # 演示材料
    └── demo_script.md              # 演示脚本
```

## 7. 技术栈

| 组件 | 选型 | 理由 |
|------|------|------|
| 算法层语言 | C/C++ | 比赛要求，HCCL 原生接口 |
| Agent 语言 | Python | LLM 生态丰富，开发效率高 |
| 构建系统 | CMake | C/C++ 标准构建，跨平台 |
| 测试框架 | Google Test | C++ 主流测试框架 |
| LLM 调用 | Anthropic SDK / OpenAI SDK | Agent 底层能力 |
| 代码索引 | tree-sitter | 代码结构化解析，供 Agent 使用 |

## 8. 实施阶段

### 8.1 四阶段路线图

```
Phase 1: 地基          Phase 2: 核心算法       Phase 3: Agent 系统       Phase 4: 创新与打磨
(2-3 周)               (3-4 周)               (2-3 周)                  (2 周)
┌──────────────┐      ┌──────────────┐       ┌──────────────┐         ┌──────────────┐
│ 模拟器核心    │      │ 4 种原语      │       │ 4 个 Agent   │         │ 创新算法      │
│ 拓扑模型     │ ──→  │ 经典算法实现   │  ──→  │ Prompt 体系   │  ──→   │ 性能优化      │
│ HCCL 接口层  │      │ 正确性验证    │       │ 端到端流程    │         │ 文档/演示     │
│ 测试框架     │      │ 基准测试      │       │ Agent 日志    │         │ 可靠性加固    │
└──────────────┘      └──────────────┘       └──────────────┘         └──────────────┘
```

### 8.2 各阶段目标

**Phase 1：地基（2-3 周）**
- 搭建项目骨架（CMake、目录结构、CI）
- 实现 NPU 拓扑模型（节点/链路/集群）
- 实现通信信道模拟（PureSim 模式）
- 定义 HCCL Plugin Interface 头文件
- 搭建测试框架（Google Test）

**Phase 2：核心算法（3-4 周）**
- 实现 AllReduce Ring（第一个端到端可运行的算法）
- 实现 AllGather Ring / ReduceScatter Ring / AlltoAll Direct
- 所有算法在模拟器上通过正确性测试
- 实现算法选择器（按数据量自动选算法）
- 性能基准测试框架搭建

**Phase 3：Agent 系统（2-3 周）**
- 构建 Agent 编排器（Python）
- 实现 Design Agent + Prompt 模板
- 实现 Code Agent + 代码模板
- 实现 Test Agent + 测试生成
- 实现 Optimize Agent + 瓶颈分析
- 端到端 Agent 工作流验证

**Phase 4：创新与打磨（2 周）**
- Agent 辅助生成创新算法（NHR、Butterfly 等）
- 性能优化（流水线、分块、计算通信重叠）
- 故障注入与可靠性测试
- 文档撰写（设计说明书、测试报告、Agent 说明）
- 演示视频录制

### 8.3 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 模拟器精度不足 | 算法性能数据不可信 | 对齐官方文档参数，Phase 1 重点验证 |
| C/C++ 开发效率低 | 进度滞后 | Agent Code Agent 辅助生成，模板化开发 |
| Agent 生成代码质量差 | 需大量人工修正 | 提供高质量 Prompt + few-shot，渐进式优化 |
| 创新算法难度超预期 | Phase 4 时间不够 | Phase 2 保证经典算法完备，创新作为加分项 |

### 8.4 成功标准

- **功能正确性**：4 种原语 × 至少 2 种算法 = 8 个算法实现，全部通过正确性测试
- **性能可观测**：每个算法有带宽/延迟/加速比数据
- **Agent 可运行**：完整的 Agent 工程可独立运行，有完整的日志记录
- **文档完备**：设计说明书、测试报告、Agent 说明齐全

## 9. 比赛交付物映射

| 比赛要求 | 对应位置 | 说明 |
|---------|---------|------|
| HCCL 算法插件 (.so) | `build/lib/` | CMake 构建产出 |
| 头文件 | `src/algorithm/hccl_api/hccl.h` | 对齐 HCCL 接口 |
| CMake | `CMakeLists.txt` + 各子目录 | 完整构建体系 |
| 测试用例 | `tests/` | 单元/集成/性能/故障 |
| 压测脚本 | `scripts/run_benchmark.sh` | 自动化压测 |
| 故障注入工具 | `tests/fault/` | 链路故障/超时/数据损坏 |
| Agent 运行日志 | `agent/logs/` | Agent 执行记录 |
| 算法设计说明书 | `docs/design/` | 每个算法的设计文档 |
| 性能测试报告 | `docs/performance/` | 带宽/延迟/加速比 |
| 可靠性报告 | `docs/reliability/` | 故障场景测试结果 |
| Agent 专项说明 | `docs/agent/` | 能力清单 + Prompt 方案 |
| Agent 工程 | `agent/` | 完整可运行的 Agent 系统 |
