"""Build an abstraction for ROVER Trials from its rules text alone.

Enumerates all 5^3 = 125 rovers, runs the four-phase Abstraction Agent
pipeline, and writes the resulting bucket map to a JSON file. No game
evaluator, simulator, or game-tree traversal is used.

Usage:
    export OPENAI_BASE_URL=...   # OpenAI-compatible endpoint
    export OPENAI_API_KEY=...
    python examples/run_rover.py --model gpt-5.5 --k 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from abstraction_agent import AbstractionAgent
from game_configs import build_rover_config

TERRAIN_NAMES = ("sand", "grass", "rock", "mud", "ice")


def hand_to_text(rover: Any, board: List[int]) -> str:
    p, g, a = rover
    return f"POWER={p} GRIP={g} AFFINITY={TERRAIN_NAMES[a]}"


def board_to_text(board: List[int]) -> str:
    return (
        "Terrain not yet revealed; it will be one of sand, grass, rock, mud, ice "
        "(each equally likely). sand & grass are smooth; rock, mud, ice are rough."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--k", type=int, default=10, help="number of buckets")
    parser.add_argument("--bonus", type=int, default=10, help="affinity bonus b")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="rover_abstraction.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    # All 125 rovers: (power, grip, affinity), each dial in {0..4}.
    rovers = [(p, g, a) for p in range(5) for g in range(5) for a in range(5)]

    agent = AbstractionAgent(
        config=build_rover_config(args.bonus), model=args.model, k=args.k,
        batch_size=20, min_std=0.10, max_corr=0.90, seed=args.seed,
        discovery_reasoning_effort="medium", scoring_reasoning_effort="low",
    )
    result = agent.run(hands=rovers, board=[], hand_to_text=hand_to_text,
                       board_to_text=board_to_text)

    out = {
        "game": "rover_trials", "k": args.k, "model": args.model,
        "features": result.selected_features,
        "dropped_features": result.dropped_features,
        "parse_rate": result.parse_rate,
        "bucket_map": {hand_to_text(r, []): b for r, b in result.bucket_map.items()},
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote abstraction for {len(rovers)} rovers into {args.k} buckets -> {args.out}")
    print(f"Selected features: {result.selected_features}")


if __name__ == "__main__":
    main()
