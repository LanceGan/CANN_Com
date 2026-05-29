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
            class_name: C++ class name
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
        blocks = re.findall(r'```(?:cpp|c\+\+)\n(.*?)```', text, re.DOTALL)
        for i, block in enumerate(blocks):
            if "class" in block and ".h" not in artifacts:
                artifacts[f"{class_name.lower()}.h"] = block.strip()
            elif "Status" in block and "Execute" in block:
                artifacts[f"{class_name.lower()}.cpp"] = block.strip()
            elif i == 0:
                artifacts[f"{class_name.lower()}.cpp"] = block.strip()
        return artifacts
