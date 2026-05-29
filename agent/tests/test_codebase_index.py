"""Tests for codebase index."""
import pytest
from pathlib import Path
from agent.context.codebase_index import CodebaseIndex


@pytest.fixture
def index():
    project_root = Path(__file__).parent.parent.parent
    return CodebaseIndex(project_root)


def test_scan_source_files(index):
    files = index.source_files()
    assert len(files) > 0
    names = [f.name for f in files]
    assert "algorithm.h" in names
    assert "allreduce_ring.h" in names


def test_get_header_content(index):
    content = index.get_file_content("src/algorithm/algorithm.h")
    assert "class Algorithm" in content
    assert "CommContext" in content


def test_get_algorithm_headers(index):
    headers = index.algorithm_headers()
    assert len(headers) >= 4
    names = [h.name for h in headers]
    assert "allreduce_ring.h" in names


def test_get_test_files(index):
    tests = index.test_files()
    assert len(tests) >= 5
    names = [t.name for t in tests]
    assert "test_allreduce.cpp" in names


def test_get_hccl_api(index):
    api = index.hccl_api_content()
    assert "hcclAllReduce" in api
    assert "hcclAllGather" in api


def test_get_simulator_api(index):
    sim = index.simulator_api_content()
    assert "class Simulator" in sim or "Topology" in sim


def test_file_not_found(index):
    with pytest.raises(FileNotFoundError):
        index.get_file_content("src/nonexistent.h")
