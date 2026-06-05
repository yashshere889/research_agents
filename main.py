from __future__ import annotations

import argparse

from research_agents.orchestrator import ResearchOrchestrator
from research_agents.storage.state_manager import StateManager
from research_agents.utils.logger import setup_logger


def main() -> None:
    setup_logger()
    parser = argparse.ArgumentParser(description="Multi-Agent Research System")
    parser.add_argument("question", type=str, help="Research question to investigate")
    parser.add_argument("--resume", type=str, default=None, help="Session ID to resume")
    args = parser.parse_args()

    orchestrator = ResearchOrchestrator(research_question=args.question)
    if args.resume:
        orchestrator.state = StateManager().load(args.resume)

    state = orchestrator.run()
    paper_path = state.paper.pdf_path if state.paper else "not generated"
    print("\n" + "=" * 60)
    print("Research complete!")
    print(f"Paper: {paper_path}")
    print(f"Total tokens used: {state.total_tokens_used:,}")
    print(f"Review cycles: {len(state.review_cycles)}")


if __name__ == "__main__":
    main()

