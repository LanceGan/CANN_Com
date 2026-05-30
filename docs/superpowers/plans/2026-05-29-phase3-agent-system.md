# Phase 3: Agent System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a full-process Agent system (Design/Code/Test/Optimize) that can autonomously design algorithms, generate C++ code, run tests, and analyze performance for the CANN distributed communication project.

**Architecture:** Python-based modular Agent system. Each Agent is a class with a `run()` method that takes structured input and produces structured output. An Orchestrator coordinates agents in sequence. Agents use prompt templates from a Prompt Hub and can call real LLM APIs (Anthropic/OpenAI) or run in mock mode for testing. A SharedContext layer indexes the existing C++ codebase for agents to reference.

**Tech Stack:** Python 3.10+, pytest, anthropic/openai SDK (optional), tree-sitter (optional), Jinja2 for prompts

---

## File Structure

```
CANN_Com/
├── agent/
│   ├── __init__.py
│   ├── orchestrator.py              # NEW: Task scheduler, agent coordination
│   ├── config.py                    # NEW: Global config (LLM keys, paths)
│   ├── context/
│   │   ├── __init__.py
│   │   ├── codebase_index.py        # NEW: Parse C++ codebase, extract API surface
│   │   └── knowledge_base.py        # NEW: HCCL knowledge, topology facts
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                  # NEW: BaseAgent abstract class
│   │   ├── design_agent.py          # NEW: Algorithm design
│   │   ├── code_agent.py            # NEW: C++ code generation
│   │   ├── test_agent.py            # NEW: Test generation & execution
│   │   └── optimize_agent.py        # NEW: Performance analysis & optimization
│   ├── prompts/
│   │   ├── design/
│   │   │   ├── algorithm_design.md  # NEW
│   │   │   └── ring_allreduce.md    # NEW: Few-shot example
│   │   ├── code/
│   │   │   ├── hccl_plugin_template.md  # NEW
│   │   │   └── coding_standards.md      # NEW
│   │   ├── test/
│   │   │   └── test_generation.md   # NEW
│   │   └── optimize/
│   │       └── bottleneck_analysis.md  # NEW
│   ├── logs/                        # NEW: Agent execution logs
│   │   └── .gitkeep
│   └── tests/
│       ├── __init__.py
│       ├── test_codebase_index.py   # NEW
│       ├── test_agents.py           # NEW
│       └── test_orchestrator.py     # NEW
├── requirements.txt                 # NEW: Python dependencies
└── setup.py                         # NEW (optional)
```

---

### Task 1: Python Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `agent/__init__.py`
- Create: `agent/config.py`
- Create: `agent/agents/__init__.py`
- Create: `agent/context/__init__.py`
- Create: `agent/tests/__init__.py`
- Create: `agent/logs/.gitkeep`

- [ ] **Step 1: Create requirements.txt**

```
# f:/Projects/CANN_Com/requirements.txt
anthropic>=0.39.0
openai>=1.50.0
jinja2>=3.1.0
pytest>=8.0.0
pyyaml>=6.0
```

- [ ] **Step 2: Create agent/__init__.py**

```python
# f:/Projects/CANN_Com/agent/__init__.py
"""CANN Distributed Communication Agent System."""
__version__ = "0.1.0"
```

- [ ] **Step 3: Create agent/config.py**

```python
# f:/Projects/CANN_Com/agent/config.py
"""Global configuration for the Agent system."""
import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """Configuration for the Agent system."""

    # Project paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    src_dir: Path = field(default=None)
    tests_dir: Path = field(default=None)
    prompts_dir: Path = field(default=None)
    logs_dir: Path = field(default=None)

    # LLM settings
    llm_provider: str = "mock"  # "anthropic", "openai", "mock"
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model_name: str = "claude-sonnet-4-20250514"
    temperature: float = 0.3
    max_tokens: int = 4096

    # Build settings
    build_command: str = "bash scripts/build.sh"
    test_command: str = "cd build && ctest --output-on-failure"

    def __post_init__(self):
        if self.src_dir is None:
            self.src_dir = self.project_root / "src"
        if self.tests_dir is None:
            self.tests_dir = self.project_root / "tests"
        if self.prompts_dir is None:
            self.prompts_dir = Path(__file__).parent / "prompts"
        if self.logs_dir is None:
            self.logs_dir = Path(__file__).parent / "logs"


# Global default config
default_config = AgentConfig()
```

- [ ] **Step 4: Create __init__.py files for subpackages**

```python
# f:/Projects/CANN_Com/agent/agents/__init__.py
"""Agent implementations."""
```

```python
# f:/Projects/CANN_Com/agent/context/__init__.py
"""Shared context layer."""
```

```python
# f:/Projects/CANN_Com/agent/tests/__init__.py
"""Agent system tests."""
```

- [ ] **Step 5: Create logs directory**

```bash
mkdir -p f:/Projects/CANN_Com/agent/logs
touch f:/Projects/CANN_Com/agent/logs/.gitkeep
```

- [ ] **Step 6: Install dependencies and verify**

Run:
```bash
cd f:/Projects/CANN_Com && pip install -r requirements.txt && python -c "import agent; print(agent.__version__)"
```
Expected: `0.1.0`

- [ ] **Step 7: Commit**

```bash
git add requirements.txt agent/__init__.py agent/config.py agent/agents/__init__.py agent/context/__init__.py agent/tests/__init__.py agent/logs/.gitkeep
git commit -m "feat: add Python Agent system skeleton with config"
```

---

### Task 2: Codebase Index (Shared Context)

**Files:**
- Create: `agent/context/codebase_index.py`
- Create: `agent/tests/test_codebase_index.py`

- [ ] **Step 1: Write failing test**

```python
# f:/Projects/CANN_Com/agent/tests/test_codebase_index.py
"""Tests for codebase index."""
import pytest
from pathlib import Path
from agent.context.codebase_index import CodebaseIndex


@pytest.fixture
def index():
    """Create a codebase index from the real project."""
    project_root = Path(__file__).parent.parent.parent
    return CodebaseIndex(project_root)


def test_scan_source_files(index):
    """Index should find all .h and .cpp files in src/."""
    files = index.source_files()
    assert len(files) > 0
    # Should find at least the core algorithm files
    names = [f.name for f in files]
    assert "algorithm.h" in names
    assert "allreduce_ring.h" in names


def test_get_header_content(index):
    """Should be able to read header file content."""
    content = index.get_file_content("src/algorithm/algorithm.h")
    assert "class Algorithm" in content
    assert "CommContext" in content


def test_get_algorithm_headers(index):
    """Should list all algorithm header files."""
    headers = index.algorithm_headers()
    assert len(headers) >= 4  # allreduce, allgather, reduce_scatter, alltoall
    names = [h.name for h in headers]
    assert "allreduce_ring.h" in names


def test_get_test_files(index):
    """Should list all test files."""
    tests = index.test_files()
    assert len(tests) >= 5
    names = [t.name for t in tests]
    assert "test_allreduce.cpp" in names


def test_get_hccl_api(index):
    """Should return HCCL API header content."""
    api = index.hccl_api_content()
    assert "hcclAllReduce" in api
    assert "hcclAllGather" in api


def test_get_simulator_api(index):
    """Should return simulator API content."""
    sim = index.simulator_api_content()
    assert "class Simulator" in sim
    assert "class Topology" in sim


def test_file_not_found(index):
    """Should raise FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        index.get_file_content("src/nonexistent.h")
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd f:/Projects/CANN_Com && python -m pytest agent/tests/test_codebase_index.py -v 2>&1 | head -20
```
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.context.codebase_index'`

- [ ] **Step 3: Implement codebase_index.py**

```python
# f:/Projects/CANN_Com/agent/context/codebase_index.py
"""Indexes the C++ codebase for Agent reference."""
from pathlib import Path
from typing import List, Optional


class CodebaseIndex:
    """Provides structured access to the C++ codebase."""

    def __init__(self, project_root: Path):
        self._root = Path(project_root)
        self._src = self._root / "src"
        self._tests = self._root / "tests"

    def source_files(self) -> List[Path]:
        """List all .h and .cpp files in src/."""
        files = []
        for ext in ("*.h", "*.cpp"):
            files.extend(self._src.rglob(ext))
        return sorted(files)

    def get_file_content(self, relative_path: str) -> str:
        """Read a file relative to project root."""
        path = self._root / relative_path
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path.read_text(encoding="utf-8")

    def algorithm_headers(self) -> List[Path]:
        """List all algorithm header files."""
        algo_dir = self._src / "algorithm"
        headers = list(algo_dir.rglob("*.h"))
        # Exclude hccl_api/hccl.h (interface, not algorithm)
        return sorted([h for h in headers if "hccl_api" not in str(h)])

    def test_files(self) -> List[Path]:
        """List all test .cpp files."""
        return sorted(self._tests.rglob("test_*.cpp"))

    def hccl_api_content(self) -> str:
        """Return the HCCL Plugin Interface header."""
        return self.get_file_content("src/algorithm/hccl_api/hccl.h")

    def simulator_api_content(self) -> str:
        """Return combined simulator API headers."""
        parts = []
        for name in ["topology/topology.h", "channel/channel.h", "simulator.h"]:
            path = self._src / "simulator" / name
            if path.exists():
                parts.append(f"// === {name} ===")
                parts.append(path.read_text(encoding="utf-8"))
        return "\n\n".join(parts)

    def algorithm_api_content(self) -> str:
        """Return the Algorithm base class header."""
        return self.get_file_content("src/algorithm/algorithm.h")

    def get_algorithm_implementation(self, algo_name: str) -> Optional[str]:
        """Get the implementation of a specific algorithm."""
        for f in self._src.rglob("*.cpp"):
            if algo_name.lower() in f.stem.lower():
                return f.read_text(encoding="utf-8")
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd f:/Projects/CANN_Com && python -m pytest agent/tests/test_codebase_index.py -v
```
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/context/codebase_index.py agent/tests/test_codebase_index.py
git commit -m "feat: add codebase index for Agent shared context"
```

---

### Task 3: Knowledge Base

**Files:**
- Create: `agent/context/knowledge_base.py`

- [ ] **Step 1: Implement knowledge_base.py**

```python
# f:/Projects/CANN_Com/agent/context/knowledge_base.py
"""Domain knowledge base for Agent system."""
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class AlgorithmInfo:
    """Information about a communication algorithm."""
    name: str
    primitive: str  # AllReduce, AllGather, etc.
    complexity: str  # O(N), O(NlogN), etc.
    steps: str       # "2*(N-1)", "N-1", etc.
    description: str
    best_for: str    # "small data", "large data", etc.


@dataclass
class TopologyInfo:
    """Information about hardware topology."""
    name: str
    description: str
    link_types: List[str]
    bandwidth_range: str
    latency_range: str


class KnowledgeBase:
    """Domain knowledge for communication algorithm design."""

    ALGORITHMS: Dict[str, AlgorithmInfo] = {
        "AllReduceRing": AlgorithmInfo(
            name="AllReduceRing",
            primitive="AllReduce",
            complexity="O(2*(N-1))",
            steps="2*(N-1)",
            description="Ring AllReduce: Reduce-Scatter phase followed by AllGather phase.",
            best_for="Medium to large data, bandwidth-optimal for single-node.",
        ),
        "AllGatherRing": AlgorithmInfo(
            name="AllGatherRing",
            primitive="AllGather",
            complexity="O(N-1)",
            steps="N-1",
            description="Ring AllGather: propagate chunks around a ring.",
            best_for="Medium data, simple implementation.",
        ),
        "ReduceScatterRing": AlgorithmInfo(
            name="ReduceScatterRing",
            primitive="ReduceScatter",
            complexity="O(N-1)",
            steps="N-1",
            description="Ring ReduceScatter: reduce and scatter data in a ring pattern.",
            best_for="Medium data, paired with AllGather for AllReduce.",
        ),
        "AlltoAllDirect": AlgorithmInfo(
            name="AlltoAllDirect",
            primitive="AlltoAll",
            complexity="O(N-1)",
            steps="N-1",
            description="Direct AlltoAll: each rank sends directly to every other rank.",
            best_for="Small data, low-latency networks.",
        ),
    }

    TOPOLOGIES: Dict[str, TopologyInfo] = {
        "SingleNode": TopologyInfo(
            name="SingleNode",
            description="Single machine with 8 NPU devices, Full Mesh HCCS interconnect.",
            link_types=["HCCS"],
            bandwidth_range="100-200 GB/s",
            latency_range="0.001-0.01 ms",
        ),
        "MultiNode": TopologyInfo(
            name="MultiNode",
            description="Multiple nodes connected via ROCE switch.",
            link_types=["HCCS", "ROCE"],
            bandwidth_range="12.5-25 GB/s (ROCE)",
            latency_range="0.01-0.1 ms",
        ),
    }

    OPTIMIZATION_PATTERNS: Dict[str, str] = {
        "pipeline": "Overlap communication with computation by pipelining chunks.",
        "chunking": "Split large messages into smaller chunks for better latency hiding.",
        "hierarchical": "Use intra-node fast path (HCCS) then inter-node (ROCE).",
        "butterfly": "Logarithmic number of steps, good for latency-bound scenarios.",
        "recursive_hd": "Recursive Halving-Doubling, bandwidth-optimal for large data.",
    }

    @classmethod
    def get_algorithm(cls, name: str) -> AlgorithmInfo:
        return cls.ALGORITHMS.get(name)

    @classmethod
    def list_algorithms(cls, primitive: str = None) -> List[AlgorithmInfo]:
        algos = list(cls.ALGORITHMS.values())
        if primitive:
            algos = [a for a in algos if a.primitive == primitive]
        return algos

    @classmethod
    def get_topology(cls, name: str) -> TopologyInfo:
        return cls.TOPOLOGIES.get(name)

    @classmethod
    def get_optimization_patterns(cls) -> Dict[str, str]:
        return cls.OPTIMIZATION_PATTERNS
```

- [ ] **Step 2: Verify it works**

Run:
```bash
cd f:/Projects/CANN_Com && python -c "from agent.context.knowledge_base import KnowledgeBase; print(KnowledgeBase.get_algorithm('AllReduceRing').description)"
```
Expected: Prints the AllReduceRing description.

- [ ] **Step 3: Commit**

```bash
git add agent/context/knowledge_base.py
git commit -m "feat: add domain knowledge base for Agent system"
```

---

### Task 4: Prompt Engineering Hub

**Files:**
- Create: `agent/prompts/design/algorithm_design.md`
- Create: `agent/prompts/design/ring_allreduce.md`
- Create: `agent/prompts/code/hccl_plugin_template.md`
- Create: `agent/prompts/code/coding_standards.md`
- Create: `agent/prompts/test/test_generation.md`
- Create: `agent/prompts/optimize/bottleneck_analysis.md`

- [ ] **Step 1: Create design prompt — algorithm_design.md**

```markdown
# f:/Projects/CANN_Com/agent/prompts/design/algorithm_design.md
# Algorithm Design Prompt Template

You are an expert in distributed communication algorithms for Ascend NPU clusters.

## Task
Design a {{primitive}} algorithm for the following topology:
- Topology: {{topology_name}}
- Number of ranks: {{nranks}}
- Data size: {{data_size}} bytes
- Performance goal: {{performance_goal}}

## Available HCCL API
```
{{hccl_api}}
```

## Existing Algorithms (for reference)
{{existing_algorithms}}

## Design Requirements
1. Describe the algorithm step by step
2. Analyze communication complexity (number of steps)
3. Analyze data volume per step
4. Identify the optimal chunk size strategy
5. Consider hardware-specific optimizations (HCCS vs ROCE)

## Output Format
Provide:
1. **Algorithm Name**: [name]
2. **Algorithm Type**: [Ring/Tree/Butterfly/NHR/etc.]
3. **Step-by-step Description**: [numbered steps]
4. **Complexity Analysis**: [steps, data volume]
5. **Pseudocode**: [structured pseudocode]
6. **Expected Performance**: [theoretical bandwidth, latency]
```

- [ ] **Step 2: Create design few-shot — ring_allreduce.md**

```markdown
# f:/Projects/CANN_Com/agent/prompts/design/ring_allreduce.md
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
```

- [ ] **Step 3: Create code prompt — hccl_plugin_template.md**

```markdown
# f:/Projects/CANN_Com/agent/prompts/code/hccl_plugin_template.md
# HCCL Plugin Code Template

## File Structure

### Header File (.h)
```cpp
#pragma once
#include "algorithm/algorithm.h"

namespace cann {

class {{ClassName}} : public Algorithm {
public:
    Status Execute(void* sendbuf, void* recvbuf, size_t count,
                   HCCLDataType dtype, HCCLReduceOp op,
                   CommContext& ctx) override;

    const char* Name() const override { return "{{ClassName}}"; }

    int NumSteps(uint32_t nranks) const override {
        return {{num_steps_expression}};
    }
};

} // namespace cann
```

### Implementation File (.cpp)
```cpp
#include "{{header_include}}"
#include <cstring>
#include <vector>

namespace cann {

Status {{ClassName}}::Execute(void* sendbuf, void* recvbuf, size_t count,
                               HCCLDataType dtype, HCCLReduceOp op,
                               CommContext& ctx) {
    uint32_t rank = ctx.rank();
    uint32_t nranks = ctx.nranks();
    size_t elem_size = GetDataTypeSize(dtype);

    // Edge cases
    if (nranks <= 1) {
        if (sendbuf != recvbuf) std::memcpy(recvbuf, sendbuf, count * elem_size);
        return Status::SUCCESS;
    }
    if (count == 0) return Status::SUCCESS;

    // Algorithm implementation
    {{algorithm_body}}

    return Status::SUCCESS;
}

} // namespace cann
```

## Usage
1. Replace `{{ClassName}}` with your algorithm class name
2. Replace `{{header_include}}` with the header path
3. Replace `{{num_steps_expression}}` with the step count formula
4. Replace `{{algorithm_body}}` with the algorithm logic
```

- [ ] **Step 4: Create code prompt — coding_standards.md**

```markdown
# f:/Projects/CANN_Com/agent/prompts/code/coding_standards.md
# C++ Coding Standards for CANN Algorithms

## Namespace
All code must be in the `cann` namespace.

## Headers
- Use `#pragma once` for include guards
- Include `"algorithm/algorithm.h"` for base class
- Include `<cstring>` for memcpy, `<vector>` for std::vector

## Algorithm Class
- Inherit from `Algorithm` base class
- Implement `Execute()`, `Name()`, `NumSteps()`
- Name format: `{Primitive}{Variant}` (e.g., `AllReduceRing`, `AllGatherTree`)

## Execute Method Signature
```cpp
Status Execute(void* sendbuf, void* recvbuf, size_t count,
               HCCLDataType dtype, HCCLReduceOp op, CommContext& ctx) override;
```

## Communication Pattern
- Use `ctx.send(data, bytes, dst_rank)` and `ctx.recv(buf, bytes, src_rank)`
- Use `ReduceBuffer(dst, src, count, dtype, op)` for reduction
- Use `GetDataTypeSize(dtype)` for element size

## Edge Cases
Always handle:
- `nranks <= 1`: direct memcpy
- `count == 0`: no-op

## Memory
- Use `std::vector<uint8_t>` for temporary buffers
- Never use raw `new`/`delete`
```

- [ ] **Step 5: Create test prompt — test_generation.md**

```markdown
# f:/Projects/CANN_Com/agent/prompts/test/test_generation.md
# Test Generation Prompt Template

## Test Structure

Generate a Google Test file for the algorithm `{{ClassName}}`.

### Required Test Cases

1. **Basic correctness**: Small nranks (2-4), single element, verify output
2. **Scaling**: 8 ranks, larger data (1024+ elements)
3. **Edge cases**: count < nranks, nranks = 1
4. **Metadata**: Name() and NumSteps() verification

### Test Template
```cpp
#include <gtest/gtest.h>
#include "{{header_path}}"
#include "algorithm/algorithm.h"
#include "simulator/simulator.h"
#include "simulator/topology/topology_builder.h"
#include <vector>
#include <thread>

using namespace cann;

class {{TestClassName}} : public ::testing::Test {
protected:
    void SetUp() override { PureSimChannel::clearMailbox(); }
    void TearDown() override { PureSimChannel::clearMailbox(); }
};

TEST_F({{TestClassName}}, BasicCorrectness) {
    uint32_t nranks = 4;
    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();
    Simulator sim(topo, SimMode::PureSim);
    {{ClassName}} algo;

    // Each rank has rank_id as input
    std::vector<std::vector<float>> outputs(nranks, std::vector<float>(output_count));
    std::vector<std::thread> threads;

    for (uint32_t r = 0; r < nranks; r++) {
        threads.emplace_back([&, r]() {
            // Setup input
            CommContext ctx(r, nranks, sim.getChannel(r));
            algo.Execute(input, output, count, dtype, op, ctx);
        });
    }
    for (auto& t : threads) t.join();

    // Verify outputs
}
```

### Multithreading
Each rank runs in its own `std::thread`. Use `PureSimChannel::clearMailbox()` in SetUp/TearDown.
```

- [ ] **Step 6: Create optimize prompt — bottleneck_analysis.md**

```markdown
# f:/Projects/CANN_Com/agent/prompts/optimize/bottleneck_analysis.md
# Bottleneck Analysis Prompt Template

## Task
Analyze the performance of `{{algorithm_name}}` and identify bottlenecks.

## Input Data
- Algorithm code: {{algorithm_code}}
- Benchmark results: {{benchmark_results}}
- Topology: {{topology_info}}

## Analysis Framework

### 1. Communication Bottleneck
- Is the algorithm bandwidth-bound or latency-bound?
- What is the achieved bandwidth vs theoretical peak?
- Are there load imbalances between ranks?

### 2. Step Analysis
- How many communication steps?
- Can steps be overlapped (pipelining)?
- Is there unnecessary serialization?

### 3. Data Movement
- Total data moved per rank?
- Redundant data movement?
- Can data be aggregated before sending?

### 4. Hardware Utilization
- HCCS vs ROCE utilization?
- NUMA-aware placement?
- Buffer reuse efficiency?

## Output Format
```
## Bottleneck Report: {{algorithm_name}}

### Primary Bottleneck
[description]

### Optimization Opportunities
1. [optimization 1]: [expected improvement]
2. [optimization 2]: [expected improvement]

### Recommended Next Steps
[actionable recommendations]
```
```

- [ ] **Step 7: Commit**

```bash
git add agent/prompts/
git commit -m "feat: add Prompt Engineering Hub with templates for all agents"
```

---

### Task 5: Base Agent and LLM Interface

**Files:**
- Create: `agent/agents/base.py`

- [ ] **Step 1: Implement base.py**

```python
# f:/Projects/CANN_Com/agent/agents/base.py
"""Base Agent class and LLM interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from pathlib import Path
import json
import time
import logging

from agent.config import AgentConfig, default_config

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from an agent execution."""
    success: bool
    output: Any
    artifacts: Dict[str, str] = field(default_factory=dict)  # filename -> content
    metrics: Dict[str, float] = field(default_factory=dict)
    errors: list = field(default_factory=list)
    duration_ms: float = 0.0


class LLMInterface:
    """Interface to LLM APIs with mock fallback."""

    def __init__(self, config: AgentConfig = None):
        self._config = config or default_config

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text from LLM."""
        if self._config.llm_provider == "mock":
            return self._mock_generate(system_prompt, user_prompt)
        elif self._config.llm_provider == "anthropic":
            return self._call_anthropic(system_prompt, user_prompt)
        elif self._config.llm_provider == "openai":
            return self._call_openai(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unknown provider: {self._config.llm_provider}")

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """Call Anthropic Claude API."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self._config.anthropic_api_key)
            message = client.messages.create(
                model=self._config.model_name,
                max_tokens=self._config.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return f"ERROR: {e}"

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI API."""
        try:
            import openai
            client = openai.OpenAI(api_key=self._config.openai_api_key)
            response = client.chat.completions.create(
                model=self._config.model_name,
                max_tokens=self._config.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"ERROR: {e}"

    def _mock_generate(self, system_prompt: str, user_prompt: str) -> str:
        """Mock LLM for testing — returns a template-based response."""
        # Extract key info from user prompt to generate a reasonable mock
        if "AllReduce" in user_prompt and "design" in user_prompt.lower():
            return self._mock_allreduce_design()
        elif "AllGather" in user_prompt and "design" in user_prompt.lower():
            return self._mock_allgather_design()
        elif "test" in user_prompt.lower() and "generate" in user_prompt.lower():
            return self._mock_test_generation()
        elif "optim" in user_prompt.lower():
            return self._mock_optimization()
        elif "code" in user_prompt.lower() or "implement" in user_prompt.lower():
            return self._mock_code_generation()
        else:
            return "Mock LLM: I understand your request. Here is my analysis based on the provided context."

    def _mock_allreduce_design(self) -> str:
        return """## Algorithm Name: AllReduceRing
## Algorithm Type: Ring

### Step-by-step Description
1. Phase 1 - Reduce-Scatter (N-1 steps):
   - Each rank sends one chunk per step to its right neighbor
   - Each rank receives and reduces one chunk from its left neighbor
   - After N-1 steps, each rank owns one fully-reduced chunk

2. Phase 2 - AllGather (N-1 steps):
   - Each rank sends its reduced chunk to its right neighbor
   - Each rank receives chunks from its left neighbor
   - After N-1 steps, all ranks have the complete result

### Complexity Analysis
- Steps: 2*(N-1)
- Data per step: count/N elements
- Bandwidth efficiency: (N-1)/N

### Pseudocode
AllReduceRing(sendbuf, recvbuf, count, nranks, rank):
    copy sendbuf to recvbuf
    chunk_size = count / nranks
    for step in 0 to nranks-2:
        send chunk[(rank-step)%N] to (rank+1)%N
        recv chunk[(rank-step-1)%N] from (rank-1)%N
        reduce received into recvbuf
    for step in 0 to nranks-2:
        send chunk[(rank+1-step)%N] to (rank+1)%N
        recv chunk[(rank-step)%N] from (rank-1)%N"""

    def _mock_allgather_design(self) -> str:
        return """## Algorithm Name: AllGatherRing
## Algorithm Type: Ring

### Step-by-step Description
1. N-1 steps: each rank sends one chunk and receives one chunk
2. Data propagates around the ring

### Complexity Analysis
- Steps: N-1
- Data per step: count elements

### Pseudocode
AllGatherRing(sendbuf, recvbuf, count, nranks, rank):
    copy sendbuf to recvbuf[rank]
    for step in 0 to nranks-2:
        send chunk[(rank-step)%N] to (rank+1)%N
        recv chunk[(rank-step-1)%N] from (rank-1)%N"""

    def _mock_test_generation(self) -> str:
        return """```cpp
#include <gtest/gtest.h>
#include "algorithm/allreduce/allreduce_ring.h"
#include "algorithm/algorithm.h"
#include "simulator/simulator.h"
#include "simulator/topology/topology_builder.h"
#include <vector>
#include <thread>

using namespace cann;

class AllReduceTest : public ::testing::Test {
protected:
    void SetUp() override { PureSimChannel::clearMailbox(); }
    void TearDown() override { PureSimChannel::clearMailbox(); }
};

TEST_F(AllReduceTest, FourRanks) {
    uint32_t nranks = 4;
    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();
    Simulator sim(topo, SimMode::PureSim);
    AllReduceRing algo;
    // ... test implementation
}
```"""

    def _mock_code_generation(self) -> str:
        return """```cpp
// Generated algorithm implementation
#include "algorithm/allreduce/allreduce_ring.h"
#include <cstring>
#include <vector>

namespace cann {

Status AllReduceRing::Execute(void* sendbuf, void* recvbuf, size_t count,
                               HCCLDataType dtype, HCCLReduceOp op,
                               CommContext& ctx) {
    uint32_t rank = ctx.rank();
    uint32_t nranks = ctx.nranks();
    size_t elem_size = GetDataTypeSize(dtype);

    if (nranks <= 1) {
        if (sendbuf != recvbuf) std::memcpy(recvbuf, sendbuf, count * elem_size);
        return Status::SUCCESS;
    }

    // Implementation follows ring pattern
    // ... (generated by LLM)
    return Status::SUCCESS;
}

} // namespace cann
```"""

    def _mock_optimization(self) -> str:
        return """## Bottleneck Report

### Primary Bottleneck
Communication latency in the ring pattern.

### Optimization Opportunities
1. Pipeline communication with computation
2. Use hierarchical algorithms for multi-node
3. Optimize chunk size for cache locality

### Recommended Next Steps
1. Implement pipelined ring algorithm
2. Add topology-aware algorithm selection"""


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, config: AgentConfig = None, llm: LLMInterface = None):
        self._config = config or default_config
        self._llm = llm or LLMInterface(self._config)
        self._prompts_dir = self._config.prompts_dir

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name."""
        ...

    @abstractmethod
    def run(self, **kwargs) -> AgentResult:
        """Execute the agent's task."""
        ...

    def _load_prompt(self, relative_path: str) -> str:
        """Load a prompt template from the prompts directory."""
        path = self._prompts_dir / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        return path.read_text(encoding="utf-8")

    def _render_prompt(self, template: str, **kwargs) -> str:
        """Render a prompt template with variables."""
        result = template
        for key, value in kwargs.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result

    def _log_execution(self, result: AgentResult):
        """Log agent execution to file."""
        log_dir = self._config.logs_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        log_file = log_dir / f"{self.name}_{timestamp}.json"
        log_data = {
            "agent": self.name,
            "success": result.success,
            "duration_ms": result.duration_ms,
            "artifacts": list(result.artifacts.keys()),
            "metrics": result.metrics,
            "errors": result.errors,
        }
        log_file.write_text(json.dumps(log_data, indent=2), encoding="utf-8")
```

- [ ] **Step 2: Verify it works**

Run:
```bash
cd f:/Projects/CANN_Com && python -c "from agent.agents.base import LLMInterface, AgentResult; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add agent/agents/base.py
git commit -m "feat: add base Agent class with LLM interface and mock mode"
```

---

### Task 6: Design Agent

**Files:**
- Create: `agent/agents/design_agent.py`
- Create: `agent/tests/test_agents.py`

- [ ] **Step 1: Write failing test**

```python
# f:/Projects/CANN_Com/agent/tests/test_agents.py
"""Tests for all agents."""
import pytest
from pathlib import Path
from agent.agents.design_agent import DesignAgent
from agent.context.codebase_index import CodebaseIndex


@pytest.fixture
def project_root():
    return Path(__file__).parent.parent.parent


@pytest.fixture
def design_agent(project_root):
    from agent.config import AgentConfig
    config = AgentConfig(project_root=project_root, llm_provider="mock")
    return DesignAgent(config=config)


def test_design_agent_run(design_agent):
    """Design agent should produce an algorithm design."""
    result = design_agent.run(
        primitive="AllReduce",
        nranks=8,
        data_size=1024*1024,
        topology="SingleNode",
    )
    assert result.success
    assert "algorithm" in result.output.lower() or "ring" in result.output.lower()


def test_design_agent_has_artifacts(design_agent):
    """Design agent should produce design artifacts."""
    result = design_agent.run(
        primitive="AllReduce",
        nranks=4,
        data_size=1024,
    )
    assert result.success
    assert len(result.artifacts) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd f:/Projects/CANN_Com && python -m pytest agent/tests/test_agents.py::test_design_agent_run -v 2>&1 | head -10
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement design_agent.py**

```python
# f:/Projects/CANN_Com/agent/agents/design_agent.py
"""Design Agent — designs communication algorithms."""
import time
from typing import Any, Dict

from agent.agents.base import BaseAgent, AgentResult
from agent.config import AgentConfig
from agent.context.codebase_index import CodebaseIndex
from agent.context.knowledge_base import KnowledgeBase


class DesignAgent(BaseAgent):
    """Agent that designs communication algorithms based on requirements."""

    @property
    def name(self) -> str:
        return "design_agent"

    def run(self, **kwargs) -> AgentResult:
        """Design an algorithm.

        Args:
            primitive: Communication primitive (AllReduce, AllGather, etc.)
            nranks: Number of ranks
            data_size: Data size in bytes
            topology: Topology name (SingleNode, MultiNode)
            performance_goal: Performance target (optional)

        Returns:
            AgentResult with algorithm design in output and artifacts.
        """
        start = time.time()

        primitive = kwargs.get("primitive", "AllReduce")
        nranks = kwargs.get("nranks", 8)
        data_size = kwargs.get("data_size", 1024 * 1024)
        topology = kwargs.get("topology", "SingleNode")
        performance_goal = kwargs.get("performance_goal", "balanced")

        try:
            # Load context
            index = CodebaseIndex(self._config.project_root)
            hccl_api = index.hccl_api_content()
            existing_algos = "\n".join(
                f"- {a.name}: {a.description}"
                for a in KnowledgeBase.list_algorithms(primitive)
            )
            topo_info = KnowledgeBase.get_topology(topology)
            topo_desc = f"{topo_info.name}: {topo_info.description}" if topo_info else "Unknown"

            # Load and render prompt
            template = self._load_prompt("design/algorithm_design.md")
            user_prompt = self._render_prompt(
                template,
                primitive=primitive,
                nranks=nranks,
                data_size=data_size,
                topology_name=topo_desc,
                performance_goal=performance_goal,
                hccl_api=hccl_api[:2000],  # Truncate for context window
                existing_algorithms=existing_algos,
            )

            # Load few-shot example
            few_shot = self._load_prompt("design/ring_allreduce.md")
            system_prompt = f"""You are an expert in distributed communication algorithms for Ascend NPU clusters.
Here is an example of a well-designed algorithm:

{few_shot}
"""

            # Call LLM
            output = self._llm.generate(system_prompt, user_prompt)

            duration = (time.time() - start) * 1000

            result = AgentResult(
                success=True,
                output=output,
                artifacts={"design.md": output},
                metrics={"duration_ms": duration, "nranks": nranks, "data_size": data_size},
            )
            self._log_execution(result)
            return result

        except Exception as e:
            duration = (time.time() - start) * 1000
            result = AgentResult(
                success=False,
                output="",
                errors=[str(e)],
                duration_ms=duration,
            )
            self._log_execution(result)
            return result
```

- [ ] **Step 4: Run tests**

Run:
```bash
cd f:/Projects/CANN_Com && python -m pytest agent/tests/test_agents.py -v
```
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/agents/design_agent.py agent/tests/test_agents.py
git commit -m "feat: add Design Agent with prompt templates"
```

---

### Task 7: Code Agent, Test Agent, Optimize Agent

**Files:**
- Create: `agent/agents/code_agent.py`
- Create: `agent/agents/test_agent.py`
- Create: `agent/agents/optimize_agent.py`
- Modify: `agent/tests/test_agents.py`

- [ ] **Step 1: Implement code_agent.py**

```python
# f:/Projects/CANN_Com/agent/agents/code_agent.py
"""Code Agent — generates C++ algorithm implementations."""
import time
import re

from agent.agents.base import BaseAgent, AgentResult
from agent.config import AgentConfig
from agent.context.codebase_index import CodebaseIndex


class CodeAgent(BaseAgent):
    """Agent that generates C++ code from algorithm designs."""

    @property
    def name(self) -> str:
        return "code_agent"

    def run(self, **kwargs) -> AgentResult:
        """Generate C++ implementation from a design.

        Args:
            design: Algorithm design text (from DesignAgent)
            class_name: C++ class name (e.g., "AllReduceNHR")
            primitive: Communication primitive

        Returns:
            AgentResult with generated .h and .cpp in artifacts.
        """
        start = time.time()

        design = kwargs.get("design", "")
        class_name = kwargs.get("class_name", "Algorithm")
        primitive = kwargs.get("primitive", "AllReduce")

        try:
            index = CodebaseIndex(self._config.project_root)
            algo_api = index.algorithm_api_content()
            coding_standards = self._load_prompt("code/coding_standards.md")
            template = self._load_prompt("code/hccl_plugin_template.md")

            system_prompt = f"""You are an expert C++ developer for Ascend NPU communication algorithms.

## Coding Standards
{coding_standards}

## Algorithm Base Class API
{algo_api}

## Code Template
{template}
"""

            user_prompt = f"""Generate a complete C++ implementation for the following algorithm design:

## Algorithm Design
{design}

## Class Name: {class_name}
## Primitive: {primitive}

Generate:
1. A header file (.h) with the class declaration
2. An implementation file (.cpp) with the algorithm logic

Output the code in markdown code blocks with file paths as labels.
"""

            output = self._llm.generate(system_prompt, user_prompt)

            # Extract code blocks
            artifacts = self._extract_code_blocks(output, class_name)

            duration = (time.time() - start) * 1000

            result = AgentResult(
                success=True,
                output=output,
                artifacts=artifacts,
                metrics={"duration_ms": duration},
            )
            self._log_execution(result)
            return result

        except Exception as e:
            duration = (time.time() - start) * 1000
            result = AgentResult(success=False, output="", errors=[str(e)], duration_ms=duration)
            self._log_execution(result)
            return result

    def _extract_code_blocks(self, text: str, class_name: str) -> dict:
        """Extract C++ code blocks from markdown."""
        artifacts = {}
        # Find all ```cpp blocks
        blocks = re.findall(r'```(?:cpp|c\+\+)\n(.*?)```', text, re.DOTALL)
        for i, block in enumerate(blocks):
            if "class" in block and ".h" not in artifacts:
                artifacts[f"{class_name.lower()}.h"] = block.strip()
            elif "Status" in block and "Execute" in block:
                artifacts[f"{class_name.lower()}.cpp"] = block.strip()
            elif i == 0:
                artifacts[f"{class_name.lower()}.cpp"] = block.strip()
        return artifacts
```

- [ ] **Step 2: Implement test_agent.py**

```python
# f:/Projects/CANN_Com/agent/agents/test_agent.py
"""Test Agent — generates and runs tests for algorithms."""
import time
import subprocess
import re

from agent.agents.base import BaseAgent, AgentResult
from agent.config import AgentConfig
from agent.context.codebase_index import CodebaseIndex


class TestAgent(BaseAgent):
    """Agent that generates test cases and runs them."""

    @property
    def name(self) -> str:
        return "test_agent"

    def run(self, **kwargs) -> AgentResult:
        """Generate and optionally run tests.

        Args:
            algorithm_code: Algorithm .cpp code
            algorithm_header: Algorithm .h code
            class_name: Algorithm class name
            run_tests: Whether to actually compile and run tests

        Returns:
            AgentResult with test code in artifacts.
        """
        start = time.time()

        algo_code = kwargs.get("algorithm_code", "")
        algo_header = kwargs.get("algorithm_header", "")
        class_name = kwargs.get("class_name", "Algorithm")
        run_tests = kwargs.get("run_tests", False)

        try:
            test_template = self._load_prompt("test/test_generation.md")

            system_prompt = f"""You are an expert test engineer for C++ distributed communication algorithms.

## Test Template
{test_template}
"""

            user_prompt = f"""Generate Google Test cases for the following algorithm:

## Header
{algo_header}

## Implementation
{algo_code}

## Class Name: {class_name}

Generate comprehensive tests including:
1. Basic correctness (2-4 ranks)
2. Scaling (8 ranks, large data)
3. Algorithm metadata (Name, NumSteps)
"""

            output = self._llm.generate(system_prompt, user_prompt)

            # Extract test code
            artifacts = {}
            blocks = re.findall(r'```(?:cpp|c\+)\n(.*?)```', output, re.DOTALL)
            if blocks:
                artifacts[f"test_{class_name.lower()}.cpp"] = blocks[0].strip()

            # Optionally run tests
            test_result = None
            if run_tests and artifacts:
                test_result = self._run_build()

            duration = (time.time() - start) * 1000

            result = AgentResult(
                success=True,
                output=output,
                artifacts=artifacts,
                metrics={"duration_ms": duration},
            )
            if test_result:
                result.metrics["build_exit_code"] = test_result.get("exit_code", -1)
            self._log_execution(result)
            return result

        except Exception as e:
            duration = (time.time() - start) * 1000
            result = AgentResult(success=False, output="", errors=[str(e)], duration_ms=duration)
            self._log_execution(result)
            return result

    def _run_build(self) -> dict:
        """Run the build and tests."""
        try:
            result = subprocess.run(
                self._config.build_command,
                shell=True,
                cwd=self._config.project_root,
                capture_output=True,
                text=True,
                timeout=120,
            )
            return {"exit_code": result.returncode, "stdout": result.stdout[-1000:], "stderr": result.stderr[-1000:]}
        except subprocess.TimeoutExpired:
            return {"exit_code": -1, "error": "timeout"}
        except Exception as e:
            return {"exit_code": -1, "error": str(e)}
```

- [ ] **Step 3: Implement optimize_agent.py**

```python
# f:/Projects/CANN_Com/agent/agents/optimize_agent.py
"""Optimize Agent — analyzes performance and suggests optimizations."""
import time

from agent.agents.base import BaseAgent, AgentResult
from agent.config import AgentConfig
from agent.context.codebase_index import CodebaseIndex
from agent.context.knowledge_base import KnowledgeBase


class OptimizeAgent(BaseAgent):
    """Agent that analyzes performance and suggests optimizations."""

    @property
    def name(self) -> str:
        return "optimize_agent"

    def run(self, **kwargs) -> AgentResult:
        """Analyze performance and suggest optimizations.

        Args:
            algorithm_code: Current algorithm implementation
            algorithm_name: Name of the algorithm
            benchmark_results: Benchmark output (optional)

        Returns:
            AgentResult with optimization suggestions.
        """
        start = time.time()

        algo_code = kwargs.get("algorithm_code", "")
        algo_name = kwargs.get("algorithm_name", "")
        benchmark_results = kwargs.get("benchmark_results", "No benchmark data available.")

        try:
            bottlenek_template = self._load_prompt("optimize/bottleneck_analysis.md")
            patterns = KnowledgeBase.get_optimization_patterns()
            patterns_text = "\n".join(f"- **{k}**: {v}" for k, v in patterns.items())

            system_prompt = f"""You are an expert in distributed communication performance optimization.

## Known Optimization Patterns
{patterns_text}

## Analysis Template
{bottlenek_template}
"""

            user_prompt = f"""Analyze the following algorithm and suggest optimizations:

## Algorithm: {algo_name}
{algo_code}

## Benchmark Results
{benchmark_results}

Provide a bottleneck analysis and specific optimization recommendations.
"""

            output = self._llm.generate(system_prompt, user_prompt)

            duration = (time.time() - start) * 1000

            result = AgentResult(
                success=True,
                output=output,
                artifacts={"optimization_report.md": output},
                metrics={"duration_ms": duration},
            )
            self._log_execution(result)
            return result

        except Exception as e:
            duration = (time.time() - start) * 1000
            result = AgentResult(success=False, output="", errors=[str(e)], duration_ms=duration)
            self._log_execution(result)
            return result
```

- [ ] **Step 4: Update test file with tests for all agents**

Append to `agent/tests/test_agents.py`:

```python
from agent.agents.code_agent import CodeAgent
from agent.agents.test_agent import TestAgent
from agent.agents.optimize_agent import OptimizeAgent


@pytest.fixture
def code_agent(project_root):
    from agent.config import AgentConfig
    config = AgentConfig(project_root=project_root, llm_provider="mock")
    return CodeAgent(config=config)


@pytest.fixture
def test_agent(project_root):
    from agent.config import AgentConfig
    config = AgentConfig(project_root=project_root, llm_provider="mock")
    return TestAgent(config=config)


@pytest.fixture
def optimize_agent(project_root):
    from agent.config import AgentConfig
    config = AgentConfig(project_root=project_root, llm_provider="mock")
    return OptimizeAgent(config=config)


def test_code_agent_run(code_agent):
    """Code agent should produce C++ code artifacts."""
    result = code_agent.run(
        design="Ring AllReduce algorithm",
        class_name="AllReduceRing",
        primitive="AllReduce",
    )
    assert result.success
    assert len(result.artifacts) > 0


def test_test_agent_run(test_agent):
    """Test agent should produce test code."""
    result = test_agent.run(
        algorithm_code="Status Execute(...) { return Status::SUCCESS; }",
        algorithm_header="class AllReduceRing : public Algorithm { ... };",
        class_name="AllReduceRing",
    )
    assert result.success
    assert len(result.artifacts) > 0


def test_optimize_agent_run(optimize_agent):
    """Optimize agent should produce optimization report."""
    index = CodebaseIndex(optimize_agent._config.project_root)
    algo_code = index.get_algorithm_implementation("allreduce_ring") or "Ring AllReduce code"
    result = optimize_agent.run(
        algorithm_code=algo_code,
        algorithm_name="AllReduceRing",
    )
    assert result.success
    assert "optim" in result.output.lower() or "bottleneck" in result.output.lower()
```

- [ ] **Step 5: Run all agent tests**

Run:
```bash
cd f:/Projects/CANN_Com && python -m pytest agent/tests/test_agents.py -v
```
Expected: All 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add agent/agents/code_agent.py agent/agents/test_agent.py agent/agents/optimize_agent.py agent/tests/test_agents.py
git commit -m "feat: add Code, Test, and Optimize agents"
```

---

### Task 8: Orchestrator

**Files:**
- Create: `agent/orchestrator.py`
- Create: `agent/tests/test_orchestrator.py`

- [ ] **Step 1: Write failing test**

```python
# f:/Projects/CANN_Com/agent/tests/test_orchestrator.py
"""Tests for the Agent Orchestrator."""
import pytest
from pathlib import Path
from agent.orchestrator import Orchestrator


@pytest.fixture
def project_root():
    return Path(__file__).parent.parent.parent


@pytest.fixture
def orchestrator(project_root):
    from agent.config import AgentConfig
    config = AgentConfig(project_root=project_root, llm_provider="mock")
    return Orchestrator(config=config)


def test_orchestrator_design_only(orchestrator):
    """Orchestrator should run design agent."""
    result = orchestrator.run_pipeline(
        primitive="AllReduce",
        nranks=8,
        data_size=1024*1024,
        stages=["design"],
    )
    assert result["design"]["success"]


def test_orchestrator_full_pipeline(orchestrator):
    """Orchestrator should run the full design→code→test pipeline."""
    result = orchestrator.run_pipeline(
        primitive="AllReduce",
        nranks=4,
        data_size=1024,
        stages=["design", "code", "test"],
    )
    assert result["design"]["success"]
    assert result["code"]["success"]
    assert result["test"]["success"]


def test_orchestrator_with_optimize(orchestrator):
    """Orchestrator should include optimization stage."""
    result = orchestrator.run_pipeline(
        primitive="AllReduce",
        nranks=8,
        data_size=1024*1024,
        stages=["design", "code", "test", "optimize"],
    )
    assert all(r["success"] for r in result.values())


def test_orchestrator_logs(orchestrator):
    """Orchestrator should save execution logs."""
    result = orchestrator.run_pipeline(
        primitive="AllReduce",
        nranks=4,
        stages=["design"],
    )
    log_dir = orchestrator._config.logs_dir
    assert log_dir.exists()
    log_files = list(log_dir.glob("*.json"))
    assert len(log_files) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd f:/Projects/CANN_Com && python -m pytest agent/tests/test_orchestrator.py -v 2>&1 | head -10
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement orchestrator.py**

```python
# f:/Projects/CANN_Com/agent/orchestrator.py
"""Agent Orchestrator — coordinates the full agent pipeline."""
import time
import json
import logging
from typing import Dict, List, Any
from pathlib import Path

from agent.config import AgentConfig, default_config
from agent.agents.base import LLMInterface, AgentResult
from agent.agents.design_agent import DesignAgent
from agent.agents.code_agent import CodeAgent
from agent.agents.test_agent import TestAgent
from agent.agents.optimize_agent import OptimizeAgent
from agent.context.codebase_index import CodebaseIndex

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates multiple agents in a pipeline."""

    def __init__(self, config: AgentConfig = None):
        self._config = config or default_config
        self._llm = LLMInterface(self._config)

        self._design = DesignAgent(config=self._config, llm=self._llm)
        self._code = CodeAgent(config=self._config, llm=self._llm)
        self._test = TestAgent(config=self._config, llm=self._llm)
        self._optimize = OptimizeAgent(config=self._config, llm=self._llm)

    def run_pipeline(self, **kwargs) -> Dict[str, Dict]:
        """Run the agent pipeline.

        Args:
            primitive: Communication primitive
            nranks: Number of ranks
            data_size: Data size in bytes
            topology: Topology name
            stages: List of stages to run (default: all)
            class_name: Algorithm class name (auto-generated if not provided)

        Returns:
            Dict mapping stage name to result dict.
        """
        stages = kwargs.get("stages", ["design", "code", "test", "optimize"])
        primitive = kwargs.get("primitive", "AllReduce")
        nranks = kwargs.get("nranks", 8)
        data_size = kwargs.get("data_size", 1024 * 1024)
        topology = kwargs.get("topology", "SingleNode")
        class_name = kwargs.get("class_name", f"{primitive}Agent")

        results = {}
        pipeline_start = time.time()

        logger.info(f"Starting pipeline: {primitive} on {topology} with {nranks} ranks")

        # Stage 1: Design
        if "design" in stages:
            logger.info("Running Design Agent...")
            design_result = self._design.run(
                primitive=primitive,
                nranks=nranks,
                data_size=data_size,
                topology=topology,
            )
            results["design"] = {
                "success": design_result.success,
                "output": design_result.output,
                "artifacts": design_result.artifacts,
            }

        # Stage 2: Code Generation
        if "code" in stages:
            logger.info("Running Code Agent...")
            design_text = results.get("design", {}).get("output", "Ring algorithm")
            code_result = self._code.run(
                design=design_text,
                class_name=class_name,
                primitive=primitive,
            )
            results["code"] = {
                "success": code_result.success,
                "output": code_result.output,
                "artifacts": code_result.artifacts,
            }

        # Stage 3: Test Generation
        if "test" in stages:
            logger.info("Running Test Agent...")
            code_artifacts = results.get("code", {}).get("artifacts", {})
            algo_code = code_artifacts.get(f"{class_name.lower()}.cpp", "")
            algo_header = code_artifacts.get(f"{class_name.lower()}.h", "")
            test_result = self._test.run(
                algorithm_code=algo_code,
                algorithm_header=algo_header,
                class_name=class_name,
            )
            results["test"] = {
                "success": test_result.success,
                "output": test_result.output,
                "artifacts": test_result.artifacts,
            }

        # Stage 4: Optimization
        if "optimize" in stages:
            logger.info("Running Optimize Agent...")
            index = CodebaseIndex(self._config.project_root)
            algo_code = index.get_algorithm_implementation(class_name.lower()) or ""
            opt_result = self._optimize.run(
                algorithm_code=algo_code,
                algorithm_name=class_name,
            )
            results["optimize"] = {
                "success": opt_result.success,
                "output": opt_result.output,
                "artifacts": opt_result.artifacts,
            }

        pipeline_duration = (time.time() - pipeline_start) * 1000
        logger.info(f"Pipeline complete in {pipeline_duration:.1f}ms")

        # Save pipeline log
        self._save_pipeline_log(results, pipeline_duration)

        return results

    def _save_pipeline_log(self, results: Dict, duration_ms: float):
        """Save the full pipeline execution log."""
        log_dir = self._config.logs_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        log_file = log_dir / f"pipeline_{timestamp}.json"

        log_data = {
            "timestamp": timestamp,
            "duration_ms": duration_ms,
            "stages": {
                name: {
                    "success": r["success"],
                    "artifacts": list(r.get("artifacts", {}).keys()),
                }
                for name, r in results.items()
            },
        }
        log_file.write_text(json.dumps(log_data, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Run orchestrator tests**

Run:
```bash
cd f:/Projects/CANN_Com && python -m pytest agent/tests/test_orchestrator.py -v
```
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/orchestrator.py agent/tests/test_orchestrator.py
git commit -m "feat: add Agent Orchestrator with full pipeline coordination"
```

---

### Task 9: Final Integration and CLI

**Files:**
- Create: `agent/__main__.py` (CLI entry point)
- Run all tests

- [ ] **Step 1: Create CLI entry point**

```python
# f:/Projects/CANN_Com/agent/__main__.py
"""CLI entry point for the Agent system."""
import argparse
import json
import logging
import sys
from pathlib import Path

from agent.config import AgentConfig
from agent.orchestrator import Orchestrator


def main():
    parser = argparse.ArgumentParser(description="CANN Communication Algorithm Agent")
    parser.add_argument("--primitive", default="AllReduce",
                        choices=["AllReduce", "AllGather", "ReduceScatter", "AlltoAll"],
                        help="Communication primitive")
    parser.add_argument("--nranks", type=int, default=8, help="Number of ranks")
    parser.add_argument("--data-size", type=int, default=1024*1024, help="Data size in bytes")
    parser.add_argument("--topology", default="SingleNode",
                        choices=["SingleNode", "MultiNode"],
                        help="Topology type")
    parser.add_argument("--stages", nargs="+",
                        default=["design", "code", "test", "optimize"],
                        help="Pipeline stages to run")
    parser.add_argument("--llm-provider", default="mock",
                        choices=["mock", "anthropic", "openai"],
                        help="LLM provider")
    parser.add_argument("--class-name", default=None, help="Algorithm class name")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    project_root = Path(__file__).parent.parent
    config = AgentConfig(
        project_root=project_root,
        llm_provider=args.llm_provider,
    )

    orchestrator = Orchestrator(config=config)

    results = orchestrator.run_pipeline(
        primitive=args.primitive,
        nranks=args.nranks,
        data_size=args.data_size,
        topology=args.topology,
        stages=args.stages,
        class_name=args.class_name or f"{args.primitive}Agent",
    )

    # Print summary
    print("\n=== Agent Pipeline Results ===")
    for stage, result in results.items():
        status = "PASS" if result["success"] else "FAIL"
        artifacts = list(result.get("artifacts", {}).keys())
        print(f"  {stage}: {status} (artifacts: {', '.join(artifacts) or 'none'})")

    # Save artifacts
    output_dir = project_root / "agent" / "output"
    output_dir.mkdir(exist_ok=True)
    for stage, result in results.items():
        for filename, content in result.get("artifacts", {}).items():
            out_path = output_dir / filename
            out_path.write_text(content, encoding="utf-8")
            print(f"  Saved: {out_path}")

    print("\n=== Done ===")
    return 0 if all(r["success"] for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the full CLI**

Run:
```bash
cd f:/Projects/CANN_Com && python -m agent --primitive AllReduce --nranks 8 --stages design code test --verbose
```
Expected: Pipeline runs, prints PASS for each stage, saves artifacts to `agent/output/`.

- [ ] **Step 3: Run all agent tests**

Run:
```bash
cd f:/Projects/CANN_Com && python -m pytest agent/tests/ -v
```
Expected: All tests pass (test_codebase_index + test_agents + test_orchestrator).

- [ ] **Step 4: Commit**

```bash
git add agent/__main__.py
git commit -m "feat: add Agent CLI entry point with full pipeline support"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** Phase 3 goals from design spec:
  - Agent Orchestrator (Task 8) ✓
  - Design Agent + Prompt templates (Tasks 4, 6) ✓
  - Code Agent + code templates (Tasks 4, 7) ✓
  - Test Agent + test generation (Tasks 4, 7) ✓
  - Optimize Agent + bottleneck analysis (Tasks 4, 7) ✓
  - Shared Context layer (Tasks 2, 3) ✓
  - Prompt Engineering Hub (Task 4) ✓
  - End-to-end pipeline (Tasks 8, 9) ✓
  - Agent execution logs (Task 8) ✓
  - CLI entry point (Task 9) ✓
- [x] **No placeholders:** All steps contain complete code
- [x] **Type consistency:** `AgentResult`, `AgentConfig`, `LLMInterface`, `BaseAgent`, `Orchestrator` — used consistently
