"""Core pipeline phases for the abstraction agent."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.cluster.vq import kmeans2

from .cache import (
    get_description_cache,
    get_feature_discovery_cache,
    save_description_cache,
    save_feature_discovery_cache,
)
from .llm_api import (
    _auto_features_to_matrix,
    _call_anthropic,
    select_features,
)
from .config import AgentResult, FeatureDef, GameConfig
from .prompts import (
    feature_discovery_system_prompt,
    feature_discovery_user_prompt,
    feature_scoring_system_prompt,
    hand_batch_user_prompt,
)

logger = logging.getLogger(__name__)

# Optional cache-namespace tag. When set (e.g. after a model version bump),
# feature-discovery and hand-scoring cache keys change so stale outputs from
# the previous model version are not reused. Empty by default: cache keys
# keep their historical shape for old runs.
_CACHE_TAG = os.environ.get("ABSTRACTION_EVAL_AGENT_CACHE_TAG", "")

_MAX_LLM_RETRIES = 12
# Discovery runs once per round right after the Ollama daemon reloads the
# model, which is when transient empty/garbled-response glitch windows
# (~15-20 min) occur; give it a larger budget than per-batch scoring.
_MAX_DISCOVERY_RETRIES = 24
_RETRY_BACKOFF = 30
# Empty/unparseable proxy responses are usually transient; retry fast.
_RETRY_BACKOFF_TRANSIENT = 5
# Substrings identifying a transient empty/garbled response (fast retry) vs.
# an origin-overload error such as a Cloudflare 524 timeout (slow retry).
_TRANSIENT_ERROR_MARKERS = (
    "Could not parse streaming response",
    "Unexpected API response",
    "No features parsed",
)


# -----------------------------------------------------------------------
# Phase 1: Feature Discovery
# -----------------------------------------------------------------------


def discover_features(
    config: GameConfig,
    model: str,
    cache_version: str = "v1",
    temperature: float = 0.7,
    reasoning_effort: Optional[str] = "high",
) -> List[FeatureDef]:
    """Ask the LLM to propose strategic features for this game.

    Args:
        config: Game configuration.
        model: LLM model name.
        cache_version: Bump to bypass stale cache.
        temperature: Sampling temperature for creative feature discovery.
        reasoning_effort: Provider-specific reasoning effort hint.

    Returns:
        List of discovered feature definitions.
    """
    cache_key = f"{cache_version}_td{temperature:.2f}_r{reasoning_effort or 'none'}"
    if _CACHE_TAG:
        cache_key += f"_tag{_CACHE_TAG}"
    cached = get_feature_discovery_cache(model, config.name, cache_key)
    if cached is not None:
        return [_dict_to_feature_def(d) for d in cached]

    sys_prompt = feature_discovery_system_prompt()
    usr_prompt = feature_discovery_user_prompt(config)

    for attempt in range(1, _MAX_DISCOVERY_RETRIES + 1):
        raw = ""
        try:
            raw = _call_anthropic(
                sys_prompt, usr_prompt, model,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
            features = _parse_feature_json(raw)
            if not features:
                raise ValueError("No features parsed from LLM response")

            save_feature_discovery_cache(
                model, config.name,
                [_feature_def_to_dict(f) for f in features],
                cache_key,
            )
            logger.info(
                "Discovered %d features for %s (attempt %d)",
                len(features), config.name, attempt,
            )
            return features
        except Exception as e:
            if e.__class__.__name__ == "FatalAPIError":
                raise
            logger.warning(
                "Feature discovery attempt %d/%d failed: %s",
                attempt, _MAX_DISCOVERY_RETRIES, e,
            )
            if raw:
                logger.warning(
                    "Unparseable discovery response head: %r", raw[:300],
                )
            if attempt < _MAX_DISCOVERY_RETRIES:
                msg = str(e)
                transient = any(m in msg for m in _TRANSIENT_ERROR_MARKERS)
                time.sleep(_RETRY_BACKOFF_TRANSIENT if transient else _RETRY_BACKOFF)

    raise RuntimeError(
        f"Feature discovery failed after {_MAX_DISCOVERY_RETRIES} attempts"
    )


def _normalize_feature_name(name: str, taken: set) -> str:
    """Collapse a feature name to a single ``\\w+`` token, unique within a set.

    The score-line parser (``_auto_features_to_matrix`` / ``_parse_score_output``)
    matches ``KEY=VALUE`` with ``(\\w+)=``, so a multi-word name like
    "Current Hand Strength" is unparseable (only "Strength" is captured and it is
    not in the name set -> the whole batch scores 0). Some models (e.g. llama3.3)
    name features with spaces; single-token names (gpt-5.5, qwen2.5) are returned
    unchanged, so this is an identity for those models. We use the SAME normalized
    name in the scoring prompt and in parsing, so the model's own scores are used
    verbatim -- this only fixes tokenization, it never alters the values.
    """
    base = re.sub(r"\W+", "_", name.strip()).strip("_") or "Feature"
    candidate = base
    i = 2
    while candidate.upper() in taken:
        candidate = f"{base}_{i}"
        i += 1
    taken.add(candidate.upper())
    return candidate


def _extract_json_array(text: str) -> Any:
    """Extract the first complete JSON array embedded in ``text``.

    Some models (e.g. llama3.3) wrap the requested JSON array in prose
    ("Here are the features: [...] Each feature is ..."), which a strict
    ``json.loads`` rejects ("Expecting value" / "Extra data"). We scan for
    the first position where a complete JSON value can be decoded, and use
    that value verbatim -- the array contents are never modified, so this
    only fixes extraction, not values. Pure-JSON responses decode at the
    first candidate position and behave exactly as before.
    """
    decoder = json.JSONDecoder()
    first_list = None
    idx = text.find("[")
    while idx != -1:
        try:
            data, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx = text.find("[", idx + 1)
            continue
        if isinstance(data, list) and data and all(
            isinstance(x, dict) for x in data
        ):
            return data
        if first_list is None and isinstance(data, list):
            first_list = data
        # Skip past this decoded value (e.g. a "[0, 1]" range in prose)
        # and keep scanning for the actual array of feature objects.
        idx = text.find("[", end)
    if first_list is not None:
        return first_list
    # No embedded array found: fall back to strict parse so the caller
    # sees the original, informative JSONDecodeError.
    return json.loads(text)


def _parse_feature_json(raw: str) -> List[FeatureDef]:
    """Parse JSON array of features from LLM output."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    data = _extract_json_array(text)
    if not isinstance(data, list) or not (3 <= len(data) <= 8):
        raise ValueError(
            f"Expected 3-8 features, got {type(data).__name__} "
            f"of length {len(data) if isinstance(data, list) else '?'}"
        )

    features = []
    taken: set = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        if "name" not in item or "description" not in item:
            continue
        features.append(FeatureDef(
            name=_normalize_feature_name(str(item["name"]), taken),
            description=item["description"],
            anchors=item.get("anchors", {}),
        ))
    if not (3 <= len(features) <= 8):
        raise ValueError(
            f"Expected 3-8 valid features, got {len(features)}"
        )
    return features


def _feature_def_to_dict(f: FeatureDef) -> Dict[str, Any]:
    return {"name": f.name, "description": f.description, "anchors": f.anchors}


def scoring_cache_key(
    config: GameConfig,
    feature_defs: List[FeatureDef],
    system_prompt: str,
    temperature: float,
    reasoning_effort: Optional[str],
) -> str:
    payload = {
        "config_name": config.name,
        "game_description": config.game_description,
        "game_context": config.game_context,
        "n_hands_hint": config.n_hands_hint,
        "has_future_cards": config.has_future_cards,
        "future_card_note": config.future_card_note,
        "system_prompt": system_prompt,
        "features": [_feature_def_to_dict(f) for f in feature_defs],
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]
    return f"agent_{config.name}_{digest}"


def _dict_to_feature_def(d: Dict[str, Any]) -> FeatureDef:
    return FeatureDef(
        name=d["name"], description=d["description"],
        anchors=d.get("anchors", {}),
    )


# -----------------------------------------------------------------------
# Phase 2: Hand Scoring
# -----------------------------------------------------------------------


def score_hands(
    config: GameConfig,
    hands: list,
    board: List[int],
    feature_defs: List[FeatureDef],
    model: str,
    hand_to_text: Callable[[Any, List[int]], str],
    board_to_text: Callable[[List[int]], str],
    batch_size: int = 20,
    batch_delay: float = 10,
    max_workers: int = 1,
    temperature: float = 0.0,
    reasoning_effort: Optional[str] = "low",
) -> Tuple[Dict[Any, str], str]:
    """Score all hands on discovered features via batched LLM calls.

    Args:
        max_workers: Number of parallel API calls. >1 enables concurrent scoring.
        temperature: Sampling temperature for deterministic feature scoring.
        reasoning_effort: Provider-specific reasoning effort hint.

    Returns:
        {hand: "FEAT1=0.xx FEAT2=0.yy ..."} for each hand.
    """
    sys_prompt = feature_scoring_system_prompt(config, feature_defs)
    board_text = board_to_text(board)

    feature_names = [f.name for f in feature_defs]
    style_key = scoring_cache_key(
        config, feature_defs, sys_prompt, temperature, reasoning_effort,
    )

    descriptions: Dict[Any, str] = {}
    total_batches = (len(hands) + batch_size - 1) // batch_size

    # First pass: collect cached results and identify uncached batches
    uncached_batches: List[int] = []
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        batch = hands[start : start + batch_size]
        batch_texts = [hand_to_text(h, board) for h in batch]

        user_prompt = hand_batch_user_prompt(board_text, batch_texts, start)
        batch_hash = hashlib.sha256(
            f"{user_prompt}|{model}|{style_key}".encode()
        ).hexdigest()[:12]

        cached = get_description_cache(
            model, style_key, board_text, batch_hash, run_id=_CACHE_TAG)
        if cached is not None:
            for str_idx, desc in cached.items():
                idx = int(str_idx)
                if start <= idx < start + len(batch):
                    descriptions[batch[idx - start]] = desc
        else:
            uncached_batches.append(batch_idx)

    logger.info(
        "Cache check: %d/%d batches cached, %d to score",
        total_batches - len(uncached_batches), total_batches, len(uncached_batches),
    )

    if not uncached_batches:
        for hand in hands:
            if hand not in descriptions:
                descriptions[hand] = hand_to_text(hand, board)
        return descriptions, style_key

    # If max_workers == 0, cache-only mode: skip uncached batches
    if max_workers == 0:
        logger.info("Cache-only mode: skipping %d uncached batches", len(uncached_batches))
        return descriptions, style_key

    # Score uncached batches (parallel if max_workers > 1)
    if max_workers <= 1:
        for i, batch_idx in enumerate(uncached_batches):
            _score_single_batch(
                batch_idx, hands, board, batch_size, hand_to_text,
                board_text, sys_prompt, model, style_key, descriptions,
                i + 1, len(uncached_batches), temperature, reasoning_effort,
                feature_names,
            )
            if batch_delay > 0 and i < len(uncached_batches) - 1:
                time.sleep(batch_delay)
    else:
        _score_batches_parallel(
            uncached_batches, hands, board, batch_size, hand_to_text,
            board_text, sys_prompt, model, style_key, descriptions,
            max_workers, temperature, reasoning_effort, feature_names,
        )

    for hand in hands:
        if hand not in descriptions:
            descriptions[hand] = hand_to_text(hand, board)

    return descriptions, style_key


def _score_single_batch(
    batch_idx: int,
    hands: list,
    board: List[int],
    batch_size: int,
    hand_to_text: Callable,
    board_text: str,
    sys_prompt: str,
    model: str,
    style_key: str,
    descriptions: Dict[Any, str],
    progress_num: int,
    progress_total: int,
    temperature: float,
    reasoning_effort: Optional[str],
    feature_names: List[str],
) -> None:
    """Score a single batch and update descriptions dict."""
    start = batch_idx * batch_size
    batch = hands[start : start + batch_size]
    batch_texts = [hand_to_text(h, board) for h in batch]
    user_prompt = hand_batch_user_prompt(board_text, batch_texts, start)
    batch_hash = hashlib.sha256(
        f"{user_prompt}|{model}|{style_key}".encode()
    ).hexdigest()[:12]
    total_batches = (len(hands) + batch_size - 1) // batch_size

    parsed = _score_batch_with_parse_retry(
        sys_prompt, user_prompt, model, len(batch), start,
        batch_idx + 1, total_batches, temperature, reasoning_effort,
        feature_names,
    )

    if len(parsed) >= len(batch) * 0.5:
        save_description_cache(
            model, style_key, board_text, batch_hash,
            {str(k): v for k, v in parsed.items()},
            run_id=_CACHE_TAG,
        )

    for idx, desc in parsed.items():
        local_idx = idx - start
        if 0 <= local_idx < len(batch):
            descriptions[batch[local_idx]] = desc

    logger.info(
        "  Batch %d/%d: parsed %d/%d",
        progress_num, progress_total, len(parsed), len(batch),
    )


def _score_batches_parallel(
    uncached_batches: List[int],
    hands: list,
    board: List[int],
    batch_size: int,
    hand_to_text: Callable,
    board_text: str,
    sys_prompt: str,
    model: str,
    style_key: str,
    descriptions: Dict[Any, str],
    max_workers: int,
    temperature: float,
    reasoning_effort: Optional[str],
    feature_names: List[str],
) -> None:
    """Score multiple batches in parallel using ThreadPoolExecutor."""
    import threading
    lock = threading.Lock()
    completed = [0]
    total = len(uncached_batches)
    total_batches = (len(hands) + batch_size - 1) // batch_size

    def process_batch(batch_idx: int) -> None:
        start = batch_idx * batch_size
        batch = hands[start : start + batch_size]
        batch_texts = [hand_to_text(h, board) for h in batch]
        user_prompt = hand_batch_user_prompt(board_text, batch_texts, start)
        batch_hash = hashlib.sha256(
            f"{user_prompt}|{model}|{style_key}".encode()
        ).hexdigest()[:12]

        parsed = _score_batch_with_parse_retry(
            sys_prompt, user_prompt, model, len(batch), start,
            batch_idx + 1, total_batches, temperature, reasoning_effort,
            feature_names,
        )

        if len(parsed) >= len(batch) * 0.5:
            save_description_cache(
                model, style_key, board_text, batch_hash,
                {str(k): v for k, v in parsed.items()},
                run_id=_CACHE_TAG,
            )

        with lock:
            for idx, desc in parsed.items():
                local_idx = idx - start
                if 0 <= local_idx < len(batch):
                    descriptions[batch[local_idx]] = desc
            completed[0] += 1
            logger.info(
                "  [parallel] %d/%d done (batch %d/%d): parsed %d/%d",
                completed[0], total, batch_idx + 1, total_batches,
                len(parsed), len(batch),
            )

    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch, bi) for bi in uncached_batches]
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                failures.append(exc)
                logger.error("Batch failed with exception: %s", exc)
    if failures:
        raise RuntimeError(f"{len(failures)} scoring batches failed")


_PARSE_RETRIES = 3
_MIN_PARSE_RATE = 0.5


def _score_batch_with_parse_retry(
    sys_prompt: str,
    user_prompt: str,
    model: str,
    n_hands: int,
    start_idx: int,
    batch_num: int,
    total_batches: int,
    temperature: float,
    reasoning_effort: Optional[str],
    feature_names: List[str],
) -> Dict[int, str]:
    """Call LLM and parse; retry on parse failure or out-of-range scores."""
    parsed: Dict[int, str] = {}
    for attempt in range(1, _PARSE_RETRIES + 1):
        raw = _call_llm_with_retry(
            sys_prompt, user_prompt, model, temperature, reasoning_effort,
        )
        if raw is None:
            logger.warning(
                "  Batch %d/%d attempt %d: LLM failure",
                batch_num, total_batches, attempt,
            )
            continue
        parsed, all_in_range = _parse_score_output(
            raw, n_hands, start_idx, feature_names,
        )
        if not all_in_range:
            logger.warning(
                "  Batch %d/%d attempt %d: out-of-range score, "
                "retrying whole batch",
                batch_num, total_batches, attempt,
            )
            if attempt < _PARSE_RETRIES:
                time.sleep(_RETRY_BACKOFF)
            continue
        if len(parsed) >= n_hands * _MIN_PARSE_RATE:
            return parsed
        logger.warning(
            "  Batch %d/%d attempt %d: parsed %d/%d (retrying)",
            batch_num, total_batches, attempt, len(parsed), n_hands,
        )
        if attempt < _PARSE_RETRIES:
            time.sleep(_RETRY_BACKOFF)
    logger.error(
        "  Batch %d/%d: all %d parse attempts failed",
        batch_num, total_batches, _PARSE_RETRIES,
    )
    return parsed if parsed else {}


def _call_llm_with_retry(
    sys_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    reasoning_effort: Optional[str],
) -> Optional[str]:
    """Call LLM with exponential backoff retries."""
    for attempt in range(1, _MAX_LLM_RETRIES + 1):
        try:
            return _call_anthropic(
                sys_prompt, user_prompt, model,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
        except Exception as e:
            if e.__class__.__name__ == "FatalAPIError":
                raise
            logger.warning(
                "LLM call attempt %d/%d failed: %s",
                attempt, _MAX_LLM_RETRIES, e,
            )
            if attempt < _MAX_LLM_RETRIES:
                backoff = _RETRY_BACKOFF * (2 ** (attempt - 1))
                time.sleep(min(backoff, 300))
    logger.error("All %d LLM attempts failed for this batch", _MAX_LLM_RETRIES)
    return None


def _parse_score_output(
    text: str, n_hands: int, start_idx: int, feature_names: List[str],
) -> Tuple[Dict[int, str], bool]:
    """Parse scored output lines like '0: FEAT1=0.850 FEAT2=0.320'.

    Validates that all scores are decimals in [0,1].

    Returns:
        (parsed, all_in_range) where all_in_range is False if ANY matched
        line contained an out-of-range score. The caller retries the whole
        batch when all_in_range is False.
    """
    result: Dict[int, str] = {}
    all_in_range = True
    line_pattern = re.compile(r"^\s*`?\s*(\d+)\s*:\s*(.+?)`?\s*$", re.MULTILINE)
    kv_pattern = re.compile(r"(\w+)=([\d.]+)")
    name_set = {n.upper() for n in feature_names}

    for match in line_pattern.finditer(text):
        idx = int(match.group(1))
        desc = match.group(2).strip()
        if not (start_idx <= idx < start_idx + n_hands and desc):
            continue

        # Validate all scores in this line are decimals in [0,1]
        valid = True
        for key, val_str in kv_pattern.findall(desc):
            if key.upper() in name_set:
                try:
                    val = float(val_str)
                    if not (0.0 <= val <= 1.0):
                        valid = False
                        break
                except ValueError:
                    valid = False
                    break

        if valid:
            result[idx] = desc
        else:
            all_in_range = False

    return result, all_in_range


# -----------------------------------------------------------------------
# Phase 3: Feature Selection + Standardization
# -----------------------------------------------------------------------


def corr_weight_columns(
    standardized: np.ndarray,
    selected_names: List[str],
    target: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Scale each standardized column by |corr(col, target)|**gamma (fix B).

    High-signal features (aligned with the discriminative target, e.g.
    equity) keep their scale and dominate KMeans; low-signal noise columns
    shrink toward zero. gamma=0 -> all weights 1.0 (no-op, legacy behaviour).

    Args:
        standardized: [n_hands, n_selected] z-scored matrix.
        selected_names: column names (for logging parity; unused in math).
        target: [n_hands] discriminative target vector.
        gamma: exponent on |corr|; 0 disables weighting.

    Returns:
        Weighted matrix, same shape.
    """
    if gamma <= 0.0 or standardized.shape[1] == 0:
        return standardized
    tgt = np.asarray(target, dtype=float).ravel()
    tgt_std = float(np.std(tgt))
    weights = np.ones(standardized.shape[1])
    if tgt_std > 0.0:
        for j in range(standardized.shape[1]):
            col = standardized[:, j]
            if float(np.std(col)) > 0.0:
                r = np.corrcoef(col, tgt)[0, 1]
                w = 0.0 if np.isnan(r) else abs(float(r))
                weights[j] = w ** gamma
    return standardized * weights


def parse_and_select(
    descriptions: Dict[Any, str],
    hands: list,
    feature_names: List[str],
    min_std: float = 0.10,
    max_corr: float = 0.90,
    discriminative_target: Optional[np.ndarray] = None,
    corr_weight_gamma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, str], float]:
    """Parse scores, filter features, and standardize.

    Args:
        descriptions: per-hand LLM score strings.
        hands: ordered holdings parallel to rows.
        feature_names: discovered feature names parallel to score columns.
        min_std: low-variance drop threshold.
        max_corr: de-correlation threshold.
        discriminative_target: optional [n_hands] vector (e.g. equity).
            Passed to select_features (fix A: equity-aware de-corr tie-break)
            and used for corr-weighting (fix B). None -> legacy behaviour.
        corr_weight_gamma: exponent for |target-corr| column weighting
            (fix B). 0 -> no weighting (legacy). Requires discriminative_target.

    Returns:
        (raw_matrix, standardized_matrix, selected_names, drop_log, parse_rate)
    """
    raw_matrix, parsed_count = _auto_features_to_matrix(
        descriptions, hands, feature_names,
    )
    parse_rate = parsed_count / max(len(hands), 1)

    selected, selected_names, drop_log = select_features(
        raw_matrix, feature_names, min_std=min_std, max_corr=max_corr,
        discriminative_target=discriminative_target,
    )

    std = selected.std(axis=0)
    std[std < 1e-10] = 1.0
    standardized = selected / std

    if discriminative_target is not None and corr_weight_gamma > 0.0:
        standardized = corr_weight_columns(
            standardized, selected_names, discriminative_target,
            corr_weight_gamma,
        )

    return raw_matrix, standardized, selected_names, drop_log, parse_rate


# -----------------------------------------------------------------------
# Phase 4: Clustering
# -----------------------------------------------------------------------


def cluster_hands(
    features: np.ndarray,
    k: int,
    seed: int = 0,
    n_restarts: int = 10,
) -> np.ndarray:
    """Multi-restart k-means clustering.

    Returns:
        Labels array of shape [n_hands].
    """
    k = min(k, len(features))
    best_labels: Optional[np.ndarray] = None
    best_inertia = float("inf")

    for run in range(n_restarts):
        rng = np.random.RandomState(seed + run)
        centroids, labels = kmeans2(
            features, k, minit="++", seed=rng, iter=50,
        )
        inertia = sum(
            np.sum((features[labels == j] - centroids[j]) ** 2)
            for j in range(k)
            if np.any(labels == j)
        )
        if inertia < best_inertia:
            best_labels, best_inertia = labels, inertia

    assert best_labels is not None, "n_restarts must be >= 1"
    return best_labels
