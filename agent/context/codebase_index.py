"""Indexes the C++ codebase for Agent reference."""
from pathlib import Path
from typing import List, Optional


class CodebaseIndex:
    """Provides structured access to the C++ codebase."""

    def __init__(self, project_root: Path):
        self._root = Path(project_root)
        self._src = self._root / "src"
        self._tests = self._root / "tests"

    def source_files(self) -> List[Path]:
        files = []
        for ext in ("*.h", "*.cpp"):
            files.extend(self._src.rglob(ext))
        return sorted(files)

    def get_file_content(self, relative_path: str) -> str:
        path = (self._root / relative_path).resolve()
        if not path.is_relative_to(self._root.resolve()):
            raise ValueError(f"Path traversal detected: {relative_path}")
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path.read_text(encoding="utf-8")

    def algorithm_headers(self) -> List[Path]:
        algo_dir = self._src / "algorithm"
        headers = list(algo_dir.rglob("*.h"))
        return sorted([h for h in headers if "hccl_api" not in str(h)])

    def test_files(self) -> List[Path]:
        return sorted(self._tests.rglob("test_*.cpp"))

    def hccl_api_content(self) -> str:
        return self.get_file_content("src/algorithm/hccl_api/hccl.h")

    def simulator_api_content(self) -> str:
        parts = []
        for name in ["topology/topology.h", "channel/channel.h", "simulator.h"]:
            path = self._src / "simulator" / name
            if path.exists():
                parts.append(f"// === {name} ===")
                parts.append(path.read_text(encoding="utf-8"))
        return "\n\n".join(parts)

    def algorithm_api_content(self) -> str:
        return self.get_file_content("src/algorithm/algorithm.h")

    def get_algorithm_implementation(self, algo_name: str) -> Optional[str]:
        for f in self._src.rglob("*.cpp"):
            if algo_name.lower() in f.stem.lower():
                return f.read_text(encoding="utf-8")
        return None
