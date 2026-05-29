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
            class_name: Algorithm class name

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

        self._save_pipeline_log(results, pipeline_duration)

        return results

    def _save_pipeline_log(self, results: Dict, duration_ms: float):
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
