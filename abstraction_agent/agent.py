"""AbstractionAgent: game-agnostic LLM abstraction orchestrator."""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from .config import AgentResult, GameConfig
from .pipeline import (
    cluster_hands,
    discover_features,
    parse_and_select,
    score_hands,
)

logger = logging.getLogger(__name__)


class AbstractionAgent:
    """End-to-end LLM abstraction agent.

    Discovers strategic features via LLM, scores all holdings,
    selects informative features, and clusters into K groups.

    Game-specific knowledge is injected via GameConfig.
    This class contains zero domain-specific terminology.
    """

    def __init__(
        self,
        config: GameConfig,
        model: str = "gpt-5.5",
        k: int = 20,
        batch_size: int = 20,
        batch_delay: float = 10,
        min_std: float = 0.10,
        max_corr: float = 0.90,
        seed: int = 0,
        cache_version: str = "v1",
        max_workers: int = 1,
        discovery_temperature: float = 0.7,
        discovery_reasoning_effort: Optional[str] = "high",
        scoring_temperature: float = 0.0,
        scoring_reasoning_effort: Optional[str] = "low",
    ):
        self.config = config
        self.model = model
        self.k = k
        self.batch_size = batch_size
        self.batch_delay = batch_delay
        self.min_std = min_std
        self.max_corr = max_corr
        self.seed = seed
        self.cache_version = cache_version
        self.max_workers = max_workers
        self.discovery_temperature = discovery_temperature
        self.discovery_reasoning_effort = discovery_reasoning_effort
        self.scoring_temperature = scoring_temperature
        self.scoring_reasoning_effort = scoring_reasoning_effort

    def run(
        self,
        hands: list,
        board: List[int],
        hand_to_text: Callable[[Any, List[int]], str],
        board_to_text: Callable[[List[int]], str],
        discriminative_target: Optional[Dict[Any, float]] = None,
        corr_weight_gamma: float = 0.0,
    ) -> AgentResult:
        """Run the full abstraction pipeline.

        Args:
            hands: Ordered list of private holdings.
            board: Public information cards.
            hand_to_text: Converts (hand, board) to text for LLM.
            board_to_text: Converts board to text for LLM.
            discriminative_target: Optional {hand: scalar} (e.g. per-hand
                equity). Enables equity-aware feature selection (fix A) and,
                with corr_weight_gamma>0, corr-weighted clustering (fix B).
                None -> legacy behaviour, unchanged.
            corr_weight_gamma: Exponent for |target-corr| column weighting
                (fix B). 0 disables weighting. Requires discriminative_target.

        Returns:
            AgentResult with bucket assignments and diagnostics.
        """
        timings: Dict[str, float] = {}

        # Phase 1: Feature Discovery
        logger.info("=" * 60)
        logger.info("Phase 1: Feature Discovery (model=%s)", self.model)
        t0 = time.time()
        feature_defs = discover_features(
            self.config,
            self.model,
            self.cache_version,
            temperature=self.discovery_temperature,
            reasoning_effort=self.discovery_reasoning_effort,
        )
        timings["discovery"] = time.time() - t0
        logger.info(
            "Discovered %d features in %.1fs:",
            len(feature_defs), timings["discovery"],
        )
        for i, feat in enumerate(feature_defs):
            logger.info(
                "  [%d] %s: %s", i, feat.name,
                feat.description[:80] + "..." if len(feat.description) > 80
                else feat.description,
            )

        # Phase 2: Hand Scoring
        logger.info("=" * 60)
        logger.info(
            "Phase 2: Scoring %d holdings (batch=%d, workers=%d)",
            len(hands), self.batch_size, self.max_workers,
        )
        t0 = time.time()
        descriptions, scoring_cache = score_hands(
            self.config, hands, board, feature_defs, self.model,
            hand_to_text, board_to_text,
            self.batch_size, self.batch_delay, self.max_workers,
            temperature=self.scoring_temperature,
            reasoning_effort=self.scoring_reasoning_effort,
        )
        timings["scoring"] = time.time() - t0

        # In cache-only mode (max_workers=0), filter to hands with valid scores
        if self.max_workers == 0:
            scored_hands = [h for h in hands if h in descriptions
                           and "=" in descriptions[h]]
            logger.info(
                "Cache-only: %d/%d hands have cached scores",
                len(scored_hands), len(hands),
            )
            if len(scored_hands) < self.k:
                raise RuntimeError(
                    f"Cache-only mode: only {len(scored_hands)} hands cached, "
                    f"need at least K={self.k}"
                )
            hands = scored_hands

        logger.info("Scoring done in %.1fs", timings["scoring"])

        # Phase 3: Parse + Select + Standardize
        logger.info("=" * 60)
        logger.info("Phase 3: Feature Selection")
        t0 = time.time()
        feature_names = [f.name for f in feature_defs]
        # Build the discriminative target aligned to the (possibly filtered)
        # hand order. Fixes A/B activate only when a target is supplied.
        target_vec: Optional[Any] = None
        if discriminative_target is not None:
            import numpy as _np
            target_vec = _np.array(
                [discriminative_target[h] for h in hands], dtype=float,
            )
        raw_matrix, std_matrix, selected_names, drop_log, parse_rate = (
            parse_and_select(
                descriptions, hands, feature_names,
                self.min_std, self.max_corr,
                discriminative_target=target_vec,
                corr_weight_gamma=corr_weight_gamma,
            )
        )
        timings["selection"] = time.time() - t0
        logger.info(
            "Selected %d/%d features, parse_rate=%.1f%% "
            "(equity_aware=%s, corr_gamma=%.1f)",
            len(selected_names), len(feature_names), 100 * parse_rate,
            discriminative_target is not None, corr_weight_gamma,
        )

        # Phase 4: Clustering
        logger.info("=" * 60)
        logger.info("Phase 4: Clustering (K=%d)", self.k)
        t0 = time.time()
        labels = cluster_hands(std_matrix, self.k, self.seed)
        bucket_map = {h: int(labels[i]) for i, h in enumerate(hands)}
        timings["clustering"] = time.time() - t0
        logger.info("Clustering done in %.1fs", timings["clustering"])

        total_time = sum(timings.values())
        logger.info("=" * 60)
        logger.info("Agent complete. Total time: %.1fs", total_time)

        return AgentResult(
            bucket_map=bucket_map,
            feature_defs=feature_defs,
            selected_features=selected_names,
            dropped_features=drop_log,
            feature_matrix=raw_matrix,
            parse_rate=parse_rate,
            descriptions=descriptions,
            timings=timings,
            scoring_cache_key=scoring_cache,
            std_matrix=std_matrix,
            cluster_hands_order=list(hands),
        )
