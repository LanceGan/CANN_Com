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

            artifacts = {}
            blocks = re.findall(r'```(?:cpp|c\+)\n(.*?)```', output, re.DOTALL)
            if blocks:
                artifacts[f"test_{class_name.lower()}.cpp"] = blocks[0].strip()

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
