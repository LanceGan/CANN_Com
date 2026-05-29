"""Tests for all agents."""
import pytest
from pathlib import Path
from agent.agents.design_agent import DesignAgent
from agent.agents.code_agent import CodeAgent
from agent.agents.test_agent import TestAgent
from agent.agents.optimize_agent import OptimizeAgent
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
    result = design_agent.run(
        primitive="AllReduce",
        nranks=8,
        data_size=1024*1024,
        topology="SingleNode",
    )
    assert result.success
    assert "algorithm" in result.output.lower() or "ring" in result.output.lower()


def test_design_agent_has_artifacts(design_agent):
    result = design_agent.run(
        primitive="AllReduce",
        nranks=4,
        data_size=1024,
    )
    assert result.success
    assert len(result.artifacts) > 0


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
    result = code_agent.run(
        design="Ring AllReduce algorithm",
        class_name="AllReduceRing",
        primitive="AllReduce",
    )
    assert result.success
    assert len(result.artifacts) > 0


def test_test_agent_run(test_agent):
    result = test_agent.run(
        algorithm_code="Status Execute(...) { return Status::SUCCESS; }",
        algorithm_header="class AllReduceRing : public Algorithm { ... };",
        class_name="AllReduceRing",
    )
    assert result.success
    assert len(result.artifacts) > 0


def test_optimize_agent_run(optimize_agent):
    index = CodebaseIndex(optimize_agent._config.project_root)
    algo_code = index.get_algorithm_implementation("allreduce_ring") or "Ring AllReduce code"
    result = optimize_agent.run(
        algorithm_code=algo_code,
        algorithm_name="AllReduceRing",
    )
    assert result.success
    assert "optim" in result.output.lower() or "bottleneck" in result.output.lower()
