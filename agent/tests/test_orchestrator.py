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
    result = orchestrator.run_pipeline(
        primitive="AllReduce",
        nranks=8,
        data_size=1024*1024,
        stages=["design"],
    )
    assert result["design"]["success"]


def test_orchestrator_full_pipeline(orchestrator):
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
    result = orchestrator.run_pipeline(
        primitive="AllReduce",
        nranks=8,
        data_size=1024*1024,
        stages=["design", "code", "test", "optimize"],
    )
    assert all(r["success"] for r in result.values())


def test_orchestrator_logs(orchestrator):
    result = orchestrator.run_pipeline(
        primitive="AllReduce",
        nranks=4,
        stages=["design"],
    )
    log_dir = orchestrator._config.logs_dir
    assert log_dir.exists()
    log_files = list(log_dir.glob("*.json"))
    assert len(log_files) > 0


def test_orchestrator_iterative(orchestrator):
    """Orchestrator should support iterative optimization."""
    result = orchestrator.run_iterative_pipeline(
        primitive="AllReduce",
        nranks=4,
        max_iterations=1,
    )
    assert "design" in result
    assert "code" in result
