# CANN Agent 专项说明

## 1. Agent 能力清单（Skills）

### 1.1 Design Agent（算法设计 Agent）
- **输入**：通信原语类型、拓扑参数、性能目标
- **输出**：算法伪代码、通信步数分析、理论复杂度
- **Prompt 策略**：Chain-of-Thought + Few-shot（Ring AllReduce 示例）+ 约束注入

### 1.2 Code Agent（代码生成 Agent）
- **输入**：算法设计文档、HCCL API 接口定义
- **输出**：.cpp/.h 文件
- **Prompt 策略**：代码模板 + 接口约束 + 自我检查

### 1.3 Test Agent（测试验证 Agent）
- **输入**：算法实现代码
- **输出**：Google Test 测试用例
- **Prompt 策略**：测试模板 + 故障场景 + 基准对比

### 1.4 Optimize Agent（性能优化 Agent）
- **输入**：性能剖析数据、当前算法代码
- **输出**：瓶颈分析报告、优化建议
- **Prompt 策略**：瓶颈模式库 + 优化知识 + A/B 对比

## 2. Prompt 工程方案

### 2.1 模板体系
```
agent/prompts/
├── design/    # 算法设计模板 + Few-shot 示例
├── code/      # 代码模板 + 编码规范
├── test/      # 测试生成模板
└── optimize/  # 瓶颈分析模板
```

### 2.2 知识注入
- HCCL API 接口定义自动注入
- 拓扑参数和硬件约束注入
- 现有算法实现作为参考注入

## 3. Agent 工作流

```
需求输入 → Design Agent → 算法设计
         → Code Agent → C++ 实现
         → Test Agent → 测试用例
         → Optimize Agent → 性能优化
```

## 4. 运行方式

```bash
# Mock 模式（无需 API Key）
python -m agent --primitive AllReduce --nranks 8 --stages design code test optimize

# 使用真实 LLM
python -m agent --primitive AllReduce --llm-provider anthropic --stages design code
```

## 5. Agent 工程结构
```
agent/
├── orchestrator.py      # 编排器
├── agents/              # 4 个核心 Agent
├── context/             # 共享上下文（代码库索引 + 知识库）
├── prompts/             # Prompt 模板
├── logs/                # 执行日志
└── __main__.py          # CLI 入口
```
