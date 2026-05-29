"""Global configuration for the Agent system."""
import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """Configuration for the Agent system."""

    # Project paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    src_dir: Path = field(default=None)
    tests_dir: Path = field(default=None)
    prompts_dir: Path = field(default=None)
    logs_dir: Path = field(default=None)

    # LLM settings
    llm_provider: str = "mock"  # "anthropic", "openai", "mock"
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model_name: str = "claude-sonnet-4-20250514"
    temperature: float = 0.3
    max_tokens: int = 4096

    # Build settings
    build_command: str = "bash scripts/build.sh"
    test_command: str = "cd build && ctest --output-on-failure"

    def __post_init__(self):
        if self.src_dir is None:
            self.src_dir = self.project_root / "src"
        if self.tests_dir is None:
            self.tests_dir = self.project_root / "tests"
        if self.prompts_dir is None:
            self.prompts_dir = Path(__file__).parent / "prompts"
        if self.logs_dir is None:
            self.logs_dir = Path(__file__).parent / "logs"


# Global default config
default_config = AgentConfig()
