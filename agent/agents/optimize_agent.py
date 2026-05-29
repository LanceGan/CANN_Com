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
