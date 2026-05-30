# CANN Agent 专项说明

## 1. Agent 能力清单（Skills）

### 1.1 Design Agent（算法设计 Agent）
- **输入**：通信原语类型、拓扑参数、性能目标
- **输出**：算法伪代码、通信步数分析、理论复杂度
- **Prompt 策略**：Chain-of-Thought + Few-shot（Ring/Butterfly/Pipeline 示例）+ 约束注入
- **知识注入**：HCCL API、拓扑特征、硬件参数、现有算法参考

### 1.2 Code Agent（代码生成 Agent）
- **输入**：算法设计文档、HCCL API 接口定义
- **输出**：.cpp/.h 文件
- **Prompt 策略**：代码模板 + 接口约束 + 自我检查
- **支持**：接收上次编译错误信息，自动修复

### 1.3 Test Agent（测试验证 Agent）
- **输入**：算法实现代码
- **输出**：Google Test 测试用例
- **Prompt 策略**：测试模板 + 故障场景 + 基准对比
- **支持**：自动生成测试 → 编译 → 运行 → 解析结果

### 1.4 Optimize Agent（性能优化 Agent）
- **输入**：性能剖析数据、当前算法代码
- **输出**：瓶颈分析报告、优化建议
- **Prompt 策略**：瓶颈模式库 + 优化知识 + A/B 对比
- **支持**：读取基准测试结果，自动分析性能瓶颈

## 2. Prompt 工程方案

### 2.1 模板体系
```
agent/prompts/
├── design/
│   ├── algorithm_design.md      # 算法设计通用模板
│   ├── ring_allreduce.md        # Ring AllReduce Few-shot
│   ├── butterfly.md             # Butterfly AllGather Few-shot
│   └── pipeline.md              # Pipeline AllReduce Few-shot
├── code/
│   ├── hccl_plugin_template.md  # HCCL 插件代码模板
│   └── coding_standards.md      # 编码规范
├── test/
│   └── test_generation.md       # 测试生成模板
└── optimize/
    ├── bottleneck_analysis.md   # 瓶颈分析模板
    └── optimization_patterns.md # 6 种优化模式
```

### 2.2 知识注入
- HCCL API 接口定义自动注入
- 拓扑参数和硬件约束注入
- 现有算法实现作为参考注入
- 优化模式库（Pipeline、Hierarchical、Topology-Aware 等）

### 2.3 Few-shot 示例覆盖
| 算法类型 | 示例文件 | 内容 |
|---------|---------|------|
| Ring AllReduce | ring_allreduce.md | 完整设计 + 伪代码 + 复杂度分析 |
| Butterfly AllGather | butterfly.md | 递归加倍模式 + 与 Ring 对比 |
| Pipeline AllReduce | pipeline.md | 流水线分块策略 + 阶段可视化 |

## 3. Agent 工作流

### 3.1 基础流程
```
需求输入 → Design Agent → 算法设计
         → Code Agent → C++ 实现
         → Test Agent → 测试用例
         → Optimize Agent → 性能优化
```

### 3.2 迭代优化流程
```
需求输入 → Design Agent → 算法设计
         → Code Agent → C++ 实现
         → 编译 → 成功？→ Test Agent → 测试通过？
           ↓ 失败              ↓ 失败
         ←←←←←←←←←←←←←←←←←←←←←←←←
         → 最大迭代次数？→ 输出结果
```

### 3.3 迭代优化参数
- `max_iterations`：最大迭代次数（默认 3）
- 每次迭代记录到 `agent/logs/iteration_N.json`
- 编译失败时注入错误信息到下次 Code Agent 调用
- 测试失败时注入测试输出到下次 Code Agent 调用

## 4. 运行方式

```bash
# Mock 模式（无需 API Key）
python -m agent --primitive AllReduce --nranks 8 --stages design code test optimize

# 使用真实 LLM
python -m agent --primitive AllReduce --llm-provider anthropic --stages design code

# 迭代优化模式
python -m agent --primitive AllReduce --nranks 8 --stages design code test --verbose
```

## 5. Agent 工程结构
```
agent/
├── orchestrator.py      # 编排器（含迭代优化循环）
├── agents/              # 4 个核心 Agent
│   ├── base.py          # 基类 + LLM 接口（Mock/Anthropic/OpenAI）
│   ├── design_agent.py
│   ├── code_agent.py
│   ├── test_agent.py
│   └── optimize_agent.py
├── context/             # 共享上下文
│   ├── codebase_index.py    # 代码库索引
│   └── knowledge_base.py    # 领域知识（9 种算法 + 6 种优化模式）
├── prompts/             # 8 个 Prompt 模板
├── logs/                # 执行日志
├── output/              # 生成的代码和报告
└── __main__.py          # CLI 入口
```

## 6. 知识库内容

### 6.1 算法知识（9 种）
AllReduceRing、AllReduceRHD、AllReducePipeline、AllGatherRing、AllGatherButterfly、ReduceScatterRing、ReduceScatterButterfly、AlltoAllDirect、BroadcastRing

### 6.2 优化模式（6 种）
- Pipeline：通信与计算重叠
- Hierarchical：节点内 HCCS + 节点间 ROCE
- Topology-Aware：根据拓扑选择算法
- Chunk Size：优化分块大小
- Butterfly RS：对数步数归约分散
- Broadcast Ring：环形广播
