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
