"""CLI entry point for the Agent system."""
import argparse
import json
import logging
import sys
from pathlib import Path

from agent.config import AgentConfig
from agent.orchestrator import Orchestrator


def main():
    parser = argparse.ArgumentParser(description="CANN Communication Algorithm Agent")
    parser.add_argument("--primitive", default="AllReduce",
                        choices=["AllReduce", "AllGather", "ReduceScatter", "AlltoAll"],
                        help="Communication primitive")
    parser.add_argument("--nranks", type=int, default=8, help="Number of ranks")
    parser.add_argument("--data-size", type=int, default=1024*1024, help="Data size in bytes")
    parser.add_argument("--topology", default="SingleNode",
                        choices=["SingleNode", "MultiNode"],
                        help="Topology type")
    parser.add_argument("--stages", nargs="+",
                        default=["design", "code", "test", "optimize"],
                        help="Pipeline stages to run")
    parser.add_argument("--llm-provider", default="mock",
                        choices=["mock", "anthropic", "openai"],
                        help="LLM provider")
    parser.add_argument("--class-name", default=None, help="Algorithm class name")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    project_root = Path(__file__).parent.parent
    config = AgentConfig(
        project_root=project_root,
        llm_provider=args.llm_provider,
    )

    orchestrator = Orchestrator(config=config)

    results = orchestrator.run_pipeline(
        primitive=args.primitive,
        nranks=args.nranks,
        data_size=args.data_size,
        topology=args.topology,
        stages=args.stages,
        class_name=args.class_name or f"{args.primitive}Agent",
    )

    print("\n=== Agent Pipeline Results ===")
    for stage, result in results.items():
        status = "PASS" if result["success"] else "FAIL"
        artifacts = list(result.get("artifacts", {}).keys())
        print(f"  {stage}: {status} (artifacts: {', '.join(artifacts) or 'none'})")

    output_dir = project_root / "agent" / "output"
    output_dir.mkdir(exist_ok=True)
    for stage, result in results.items():
        for filename, content in result.get("artifacts", {}).items():
            out_path = output_dir / filename
            out_path.write_text(content, encoding="utf-8")
            print(f"  Saved: {out_path}")

    print("\n=== Done ===")
    return 0 if all(r["success"] for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
