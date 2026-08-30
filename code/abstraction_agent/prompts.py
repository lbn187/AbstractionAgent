"""Game-agnostic prompt templates for the abstraction agent."""
from __future__ import annotations

from typing import List

from .config import FeatureDef, GameConfig


def feature_discovery_system_prompt() -> str:
    """System prompt for Phase 1: feature discovery."""
    return (
        "You are a game-theory analyst studying strategic games. "
        "Your task is to identify numerical features that capture "
        "meaningful strategic differences between private holdings."
    )


def feature_discovery_user_prompt(config: GameConfig) -> str:
    """User prompt for Phase 1: feature discovery."""
    future_req = ""
    if config.has_future_cards:
        future_req = f"""
5. FUTURE AWARENESS: {config.future_card_note}
   Features should capture BOTH the current state AND the potential for
   change when new information is revealed. Include at least one feature
   measuring how a holding's value might shift."""

    return f"""## Game Description
{config.game_description}

## Task
Propose 3-8 numerical features that capture strategically meaningful
differences between private holdings in this game.

## Requirements

1. NUMERICAL: Each feature is scored on a continuous scale from 0.000
   to 1.000, given to 3 decimal places. Use the full range, including
   values near 0.000 and 1.000 for clearly weak/strong holdings.
   Provide clear anchor definitions for 0.0, 0.5, and 1.0.

2. SPREAD: Each feature MUST produce varied scores across
   ~{config.n_hands_hint} holdings. The standard deviation should be at
   least 0.15. Avoid features where most holdings score nearly the same.

3. INDEPENDENCE: Features must capture DIFFERENT strategic dimensions.
   Pearson |r| between any two features should be < 0.9.
   Redundant features waste capacity.

4. COMPLETENESS: Features should collectively cover the major strategic
   axes of this game.
{future_req}
## Anti-Patterns (DO NOT propose features that)
- Score nearly the same for most holdings (std < 0.10)
- Are strongly correlated with another proposed feature (|r| > 0.90)
- Have >80% of scores clustered near 0.0 or near 1.0
- Measure rare events affecting <5% of holdings

## Output Format
Return a JSON array. Each element:
{{"name": "FEATURE_NAME", "description": "...", "anchors": {{"0.0": "...", "0.5": "...", "1.0": "..."}}}}
"""


def feature_scoring_system_prompt(
    config: GameConfig,
    feature_defs: List[FeatureDef],
) -> str:
    """System prompt for Phase 2: hand scoring."""
    features_text = _format_feature_defs(feature_defs)
    feature_names = [f.name for f in feature_defs]
    example_scores = " ".join(f"{n}=0.750" for n in feature_names[:3])
    if len(feature_names) > 3:
        example_scores += " ..."
    return (
        "You are evaluating private holdings for strategic abstraction "
        "in an imperfect-information game.\n\n"
        f"## Game Context\n\n{config.game_context}\n\n"
        "## Your Task\n\n"
        "Rate each holding on the following strategic dimensions. Each "
        "score is a decimal between 0.000 and 1.000, given to 3 decimal "
        "places. Use the full range, including values near 0.000 and "
        "1.000 for clearly weak/strong holdings.\n\n"
        "IMPORTANT - scores must DISCRIMINATE between holdings: different "
        "holdings almost never deserve identical scores. Think of each "
        "score as the holding's percentile within the batch and use fine "
        "3-decimal granularity (e.g. 0.412 vs 0.437, NOT coarse values "
        "like 0.0/0.5/1.0 repeated across many holdings). A feature scored "
        "identically for most holdings is useless for abstraction.\n\n"
        f"{features_text}\n\n"
        "## Output Format\n\n"
        "One line per holding, fields separated by spaces:\n"
        f"<index>: {example_scores}\n\n"
        "Where each score is a decimal to 3 places (e.g., 0.875).\n\n"
        "Output ONLY the evaluation lines. No explanations, no "
        "markdown, no backticks, no headers."
    )


def hand_batch_user_prompt(
    board_text: str,
    hands_text: List[str],
    start_idx: int = 0,
) -> str:
    """User prompt for a batch of hands to score."""
    lines = [board_text, ""]
    for i, ht in enumerate(hands_text):
        lines.append(f"{start_idx + i}: {ht}")
    return "\n".join(lines)


def _format_feature_defs(feature_defs: List[FeatureDef]) -> str:
    """Format feature definitions for inclusion in prompts."""
    parts = []
    for feat in feature_defs:
        anchors = feat.anchors
        parts.append(
            f"- {feat.name}: {feat.description}\n"
            f"  0.0 = {anchors.get('0.0', anchors.get('0', '?'))}\n"
            f"  0.5 = {anchors.get('0.5', anchors.get('50', '?'))}\n"
            f"  1.0 = {anchors.get('1.0', anchors.get('100', '?'))}"
        )
    return "\n".join(parts)
