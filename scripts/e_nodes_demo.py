"""
Ad-hoc runner for E1 → E2 → E3 with verbose logs and configurable depth.
Use this instead of typing shell snippets each time.

Usage (ensure API keys are exported in your shell):
    uv run python scripts/e_nodes_demo.py
    uv run python scripts/e_nodes_demo.py --depth 2 --lightweight
"""
import argparse
import pprint
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nodes.E_nodes.e1_web_evidence import run as e1
from nodes.E_nodes.e2_source_reliability import run as e2
from nodes.E_nodes.e3_claim_evidence_scorer import run as e3


def main():
    parser = argparse.ArgumentParser(description="E-node demo runner (E1→E2→E3)")
    parser.add_argument("--depth", type=int, default=3, help="Search depth multiplier (default 3)")
    parser.add_argument("--lightweight", action="store_true", help="Skip expensive reliability checks (whois/about)")
    args = parser.parse_args()

    state = {
        "claims": [
            {"claim_text": "An apple is red."},
            {"claim_text": "A lengthened position bias under tension is better for muscle hypertrophy."},
        ],
        "debug": True,
        "use_cache": False,
        "search_depth": args.depth,
        "lightweight_reliability": args.lightweight,
    }

    state = e1(state)
    state = e2(state)
    state = e3(state)

    print("\n=== Evidence Items (trimmed) ===")
    for ev in state.get("evidence", []):
        print(
            f"{ev.get('claim_text','')[:50]:<50} | source: {ev.get('source')} | "
            f"stance: {ev.get('stance')} | ranking: {ev.get('ranking_mode')} | "
            f"rel_score: {ev.get('reliability_score')}"
        )

    print("\n=== Claim Scores ===")
    for c in state.get("claims", []):
        print(
            f"{c.get('claim_text','')[:50]:<50} | score: {c.get('evidence_score')} | verdict: {c.get('verdict')}"
        )

    print("\n=== Full State (pp) ===")
    pprint.pprint(state)


if __name__ == "__main__":
    main()
