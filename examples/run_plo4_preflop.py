"""Build a PLO4 preflop abstraction from the rules text alone.

Enumerates the 16,432 canonical 4-card starting hands (suit isomorphism),
runs the four-phase Abstraction Agent pipeline, and writes the bucket map
to a JSON file. No hand evaluator or equity calculator is used.

Usage:
    export OPENAI_BASE_URL=...   # OpenAI-compatible endpoint
    export OPENAI_API_KEY=...
    python examples/run_plo4_preflop.py --model gpt-5.5 --k 30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import combinations, permutations
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from abstraction_agent import AbstractionAgent
from card_utils import Hand, card_to_str
from game_configs import PLO4_PREFLOP_CONFIG


def canonical_plo4_key(hand: Hand) -> Tuple[Tuple[int, int], ...]:
    """Canonical form of a 4-card hand under suit permutation."""
    cards = [(c // 4, c % 4) for c in hand]
    suits = sorted({s for _, s in cards})
    best = None
    for perm in permutations(range(4), len(suits)):
        mapping = dict(zip(suits, perm))
        candidate = tuple(
            sorted(((r, mapping[s]) for r, s in cards), key=lambda x: (x[0], x[1]))
        )
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best


def enumerate_plo4_preflop() -> List[Hand]:
    """All 16,432 canonical PLO4 starting hands."""
    reps: Dict[Tuple[Tuple[int, int], ...], Hand] = {}
    for hand in combinations(range(52), 4):
        reps.setdefault(canonical_plo4_key(hand), tuple(sorted(hand)))
    hands = [reps[key] for key in sorted(reps)]
    assert len(hands) == 16432
    return hands


def hand_to_text(hand: Any, board: List[int]) -> str:
    return " ".join(card_to_str(c) for c in hand)


def board_to_text(board: List[int]) -> str:
    return "(preflop - no community cards dealt yet)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--k", type=int, default=30, help="number of buckets")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="plo4_preflop_abstraction.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    hands = enumerate_plo4_preflop()

    agent = AbstractionAgent(
        config=PLO4_PREFLOP_CONFIG, model=args.model, k=args.k,
        batch_size=30, min_std=0.10, max_corr=0.90, seed=args.seed,
    )
    result = agent.run(hands=hands, board=[], hand_to_text=hand_to_text,
                       board_to_text=board_to_text)

    out = {
        "game": "plo4_preflop", "k": args.k, "model": args.model,
        "features": result.selected_features,
        "dropped_features": result.dropped_features,
        "parse_rate": result.parse_rate,
        "bucket_map": {hand_to_text(h, []): b for h, b in result.bucket_map.items()},
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote abstraction for {len(hands)} hands into {args.k} buckets -> {args.out}")
    print(f"Selected features: {result.selected_features}")


if __name__ == "__main__":
    main()
