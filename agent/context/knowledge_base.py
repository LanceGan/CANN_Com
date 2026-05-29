"""Domain knowledge base for Agent system."""
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class AlgorithmInfo:
    """Information about a communication algorithm."""
    name: str
    primitive: str
    complexity: str
    steps: str
    description: str
    best_for: str


@dataclass
class TopologyInfo:
    """Information about hardware topology."""
    name: str
    description: str
    link_types: List[str]
    bandwidth_range: str
    latency_range: str


class KnowledgeBase:
    """Domain knowledge for communication algorithm design."""

    ALGORITHMS: Dict[str, AlgorithmInfo] = {
        "AllReduceRing": AlgorithmInfo(
            name="AllReduceRing",
            primitive="AllReduce",
            complexity="O(2*(N-1))",
            steps="2*(N-1)",
            description="Ring AllReduce: Reduce-Scatter phase followed by AllGather phase.",
            best_for="Medium to large data, bandwidth-optimal for single-node.",
        ),
        "AllGatherRing": AlgorithmInfo(
            name="AllGatherRing",
            primitive="AllGather",
            complexity="O(N-1)",
            steps="N-1",
            description="Ring AllGather: propagate chunks around a ring.",
            best_for="Medium data, simple implementation.",
        ),
        "ReduceScatterRing": AlgorithmInfo(
            name="ReduceScatterRing",
            primitive="ReduceScatter",
            complexity="O(N-1)",
            steps="N-1",
            description="Ring ReduceScatter: reduce and scatter data in a ring pattern.",
            best_for="Medium data, paired with AllGather for AllReduce.",
        ),
        "AlltoAllDirect": AlgorithmInfo(
            name="AlltoAllDirect",
            primitive="AlltoAll",
            complexity="O(N-1)",
            steps="N-1",
            description="Direct AlltoAll: each rank sends directly to every other rank.",
            best_for="Small data, low-latency networks.",
        ),
    }

    TOPOLOGIES: Dict[str, TopologyInfo] = {
        "SingleNode": TopologyInfo(
            name="SingleNode",
            description="Single machine with 8 NPU devices, Full Mesh HCCS interconnect.",
            link_types=["HCCS"],
            bandwidth_range="100-200 GB/s",
            latency_range="0.001-0.01 ms",
        ),
        "MultiNode": TopologyInfo(
            name="MultiNode",
            description="Multiple nodes connected via ROCE switch.",
            link_types=["HCCS", "ROCE"],
            bandwidth_range="12.5-25 GB/s (ROCE)",
            latency_range="0.01-0.1 ms",
        ),
    }

    OPTIMIZATION_PATTERNS: Dict[str, str] = {
        "pipeline": "Overlap communication with computation by pipelining chunks.",
        "chunking": "Split large messages into smaller chunks for better latency hiding.",
        "hierarchical": "Use intra-node fast path (HCCS) then inter-node (ROCE).",
        "butterfly": "Logarithmic number of steps, good for latency-bound scenarios.",
        "recursive_hd": "Recursive Halving-Doubling, bandwidth-optimal for large data.",
    }

    @classmethod
    def get_algorithm(cls, name: str) -> AlgorithmInfo:
        return cls.ALGORITHMS.get(name)

    @classmethod
    def list_algorithms(cls, primitive: str = None) -> List[AlgorithmInfo]:
        algos = list(cls.ALGORITHMS.values())
        if primitive:
            algos = [a for a in algos if a.primitive == primitive]
        return algos

    @classmethod
    def get_topology(cls, name: str) -> TopologyInfo:
        return cls.TOPOLOGIES.get(name)

    @classmethod
    def get_optimization_patterns(cls) -> Dict[str, str]:
        return cls.OPTIMIZATION_PATTERNS
