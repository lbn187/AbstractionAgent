"""Persistent file-based cache for embeddings and reasoning results."""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.environ.get("ABSTRACTION_AGENT_CACHE", "cache"))


def _make_key(prefix: str, *parts: str) -> str:
    raw = "|".join([prefix] + list(parts))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_embedding_cache(
    model_name: str,
    board_str: str,
    hands_hash: str,
) -> Optional[np.ndarray]:
    """Load cached embedding matrix if available."""
    key = _make_key("emb", model_name, board_str, hands_hash)
    path = CACHE_DIR / f"{key}.npz"
    if path.exists():
        logger.info("Cache hit: %s", key)
        return np.load(path)["arr_0"]
    return None


def save_embedding_cache(
    model_name: str,
    board_str: str,
    hands_hash: str,
    embeddings: np.ndarray,
) -> None:
    """Save embedding matrix to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _make_key("emb", model_name, board_str, hands_hash)
    path = CACHE_DIR / f"{key}.npz"
    np.savez_compressed(path, embeddings)
    logger.info("Cache saved: %s (%s)", key, embeddings.shape)


def get_reasoning_cache(
    model_name: str,
    prompt_style: str,
    board_str: str,
    batch_hash: str,
) -> Optional[dict]:
    """Load cached reasoning result if available."""
    key = _make_key("reason", model_name, prompt_style, board_str, batch_hash)
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        logger.info("Cache hit: %s", key)
        with open(path) as f:
            return json.load(f)
    return None


def save_reasoning_cache(
    model_name: str,
    prompt_style: str,
    board_str: str,
    batch_hash: str,
    result: dict,
) -> None:
    """Save reasoning result to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _make_key("reason", model_name, prompt_style, board_str, batch_hash)
    path = CACHE_DIR / f"{key}.json"
    with open(path, "w") as f:
        json.dump(result, f)
    logger.info("Cache saved: %s", key)


def get_description_cache(
    model_name: str,
    prompt_style: str,
    board_str: str,
    batch_hash: str,
    run_id: str = "",
) -> Optional[dict]:
    """Load cached hand descriptions if available."""
    parts = [model_name, prompt_style, board_str, batch_hash]
    if run_id:
        parts.append(run_id)
    key = _make_key("desc", *parts)
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        logger.info("Desc cache hit: %s", key)
        with open(path) as f:
            return json.load(f)
    return None


def save_description_cache(
    model_name: str,
    prompt_style: str,
    board_str: str,
    batch_hash: str,
    descriptions: dict,
    run_id: str = "",
) -> None:
    """Save hand descriptions to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parts = [model_name, prompt_style, board_str, batch_hash]
    if run_id:
        parts.append(run_id)
    key = _make_key("desc", *parts)
    path = CACHE_DIR / f"{key}.json"
    with open(path, "w") as f:
        json.dump(descriptions, f)
    logger.info("Desc cache saved: %s (%d entries)", key, len(descriptions))


def hands_hash(hands_texts: list) -> str:
    """Compute a short hash of hand text list for cache keys."""
    raw = "\n".join(str(t) for t in hands_texts)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Feature discovery cache (Plan D: Auto-Feature)
# ---------------------------------------------------------------------------

def get_feature_discovery_cache(
    model: str,
    game_type: str,
    prompt_version: str = "v1",
) -> Optional[List[Dict[str, Any]]]:
    """Load cached feature discovery result if available."""
    key = _make_key("feat_disc", model, game_type, prompt_version)
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        logger.info("Feature discovery cache hit: %s", key)
        with open(path) as f:
            return json.load(f)
    return None


def save_feature_discovery_cache(
    model: str,
    game_type: str,
    features: List[Dict[str, Any]],
    prompt_version: str = "v1",
) -> None:
    """Save feature discovery result to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _make_key("feat_disc", model, game_type, prompt_version)
    path = CACHE_DIR / f"{key}.json"
    with open(path, "w") as f:
        json.dump(features, f, indent=2)
    logger.info(
        "Feature discovery cache saved: %s (%d features)",
        key, len(features),
    )
