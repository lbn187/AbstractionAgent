"""Data structures for the abstraction agent."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class FeatureDef:
    """A strategic feature discovered by the LLM."""

    name: str
    description: str
    anchors: Dict[str, str]


@dataclass(frozen=True)
class GameConfig:
    """Game-agnostic configuration injected into the agent.

    All game-specific knowledge lives here, not in the agent code.
    """

    name: str
    game_description: str
    game_context: str
    n_hands_hint: int
    has_future_cards: bool
    future_card_note: str = ""


@dataclass
class AgentResult:
    """Output of a full agent run."""

    bucket_map: Dict[Any, int]
    feature_defs: List[FeatureDef]
    selected_features: List[str]
    dropped_features: Dict[str, str]
    feature_matrix: np.ndarray
    parse_rate: float
    descriptions: Dict[Any, str]
    timings: Dict[str, float]
    scoring_cache_key: str = ""
    # Standardized matrix actually fed to k-means (rows parallel to bucket_map
    # key order), kept so K can be re-swept offline without re-scoring.
    std_matrix: Optional[np.ndarray] = None
    cluster_hands_order: Optional[List[Any]] = None
