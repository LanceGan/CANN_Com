"""Design Agent — designs communication algorithms."""
import time

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
            index = CodebaseIndex(self._config.project_root)
            hccl_api = index.hccl_api_content()
            existing_algos = "\n".join(
                f"- {a.name}: {a.description}"
                for a in KnowledgeBase.list_algorithms(primitive)
            )
            topo_info = KnowledgeBase.get_topology(topology)
            topo_desc = f"{topo_info.name}: {topo_info.description}" if topo_info else "Unknown"

            template = self._load_prompt("design/algorithm_design.md")
            user_prompt = self._render_prompt(
                template,
                primitive=primitive,
                nranks=nranks,
                data_size=data_size,
                topology_name=topo_desc,
                performance_goal=performance_goal,
                hccl_api=hccl_api[:2000],
                existing_algorithms=existing_algos,
            )

            few_shot = self._load_prompt("design/ring_allreduce.md")
            system_prompt = f"""You are an expert in distributed communication algorithms for Ascend NPU clusters.
Here is an example of a well-designed algorithm:

{few_shot}
"""

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
