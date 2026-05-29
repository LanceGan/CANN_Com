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
    artifacts: Dict[str, str] = field(default_factory=dict)
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
        """Mock LLM for testing."""
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

    // Ring algorithm implementation
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
        ...

    @abstractmethod
    def run(self, **kwargs) -> AgentResult:
        ...

    def _load_prompt(self, relative_path: str) -> str:
        path = self._prompts_dir / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        return path.read_text(encoding="utf-8")

    def _render_prompt(self, template: str, **kwargs) -> str:
        result = template
        for key, value in kwargs.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result

    def _log_execution(self, result: AgentResult):
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
