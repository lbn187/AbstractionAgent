## Citation

```bibtex
@inproceedings{li2026abstraction,
  title={Abstraction Agent: LLM-Guided Abstraction for Imperfect-Information Games},
  author={Li, Boning and Huang, Longbo},
  booktitle={Findings of the Empirical Methods in Natural Language Processing},
  year={2026}
}
```

# Abstraction Agent

Code for the EMNLP 2026 Findings paper **"Abstraction Agent: LLM-Guided
Abstraction for Imperfect-Information Games"** (Boning Li, Longbo Huang).

The Abstraction Agent is a zero-shot pipeline that turns a natural-language
game description into an information abstraction (a mapping from private
holdings to buckets) with no game-specific evaluator, no training data, and
no game-tree traversal. It runs in four phases:

1. **Feature discovery** — the LLM proposes continuous strategic features in
   [0, 1] with calibration anchors, from the rules text alone.
2. **Private-state scoring** — every holding is scored on the discovered
   features in batched LLM calls.
3. **Feature selection** — low-variance and highly correlated features are
   dropped.
4. **Clustering** — the surviving feature vectors are partitioned into K
   buckets with multi-restart k-means.

The output is an abstraction scheme (`bucket_map`: holding -> bucket id).

## Layout

```
abstraction_agent/   the four-phase pipeline (game-agnostic)
  agent.py             AbstractionAgent orchestrator
  pipeline.py          the four phases
  prompts.py           prompt templates (identical across games)
  config.py            GameConfig / FeatureDef / AgentResult dataclasses
  llm_api.py           LLM call layer + feature post-processing
  cache.py             file cache for discovery/scoring responses
game_configs/        natural-language game descriptions used in the paper
                     (HUNL turn/flop/preflop, PLO4 preflop, Riichi Mahjong,
                      ROVER Trials)
card_utils.py        card encoding helpers for the poker examples
examples/            runnable entry points that write a bucket map to JSON
```

## Setup

```bash
pip install -r requirements.txt

# OpenAI-compatible endpoint (used for gpt-*, llama, qwen, ...)
export OPENAI_BASE_URL=...
export OPENAI_API_KEY=...
# or, for native Anthropic models:
export ANTHROPIC_API_KEY=...
```

## Usage

```bash
# ROVER Trials: 125 rovers -> 10 buckets, from the rules text alone
python examples/run_rover.py --model gpt-5.5 --k 10

# PLO4 preflop: 16,432 canonical hands -> 30 buckets
python examples/run_plo4_preflop.py --model gpt-5.5 --k 30
```

Each script writes a JSON file containing the discovered features, the
parse rate, and the bucket map.

To apply the agent to a new game, write a `GameConfig` (rules text, context,
a hint for the number of holdings, and whether future public information
exists) plus two callbacks that render a holding and the public state as
text, then call:

```python
from abstraction_agent import AbstractionAgent

agent = AbstractionAgent(config=my_config, model="gpt-5.5", k=20)
result = agent.run(hands=my_hands, board=my_board,
                   hand_to_text=my_hand_to_text, board_to_text=my_board_to_text)
result.bucket_map   # {holding: bucket_id}
```

LLM responses are cached under `./cache/` (override with the
`ABSTRACTION_AGENT_CACHE` environment variable), so re-runs with the same
model and holdings are free.
