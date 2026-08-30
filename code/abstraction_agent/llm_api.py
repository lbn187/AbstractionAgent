"""LLM API layer and feature post-processing for the Abstraction Agent.

Contains exactly what the four-phase pipeline needs:
  - _call_anthropic / _consume_openai_stream: one LLM call (Anthropic SDK for
    claude-* models, OpenAI-compatible streaming endpoint for everything else).
  - _auto_features_to_matrix: parse "FEAT=0.xx" scoring lines into a matrix.
  - select_features: drop low-variance and highly correlated features.

Environment variables:
  OPENAI_BASE_URL / OPENAI_API_KEY  for OpenAI-compatible endpoints.
  ANTHROPIC_API_KEY                 for native Anthropic models.
  LLM_FORCE_OPENAI_PROXY=1          route claude-* through the OpenAI endpoint.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

Hand = Any  # any hashable private-state identifier

logger = logging.getLogger(__name__)

def _call_anthropic(
    system_prompt: str,
    user_prompt: str,
    model: str = "claude-sonnet-4-6",
    temperature: float = 0.0,
    reasoning_effort: Optional[str] = None,
) -> str:
    """Call LLM API and return text response.

    Dispatches to Anthropic SDK for Claude models, OpenAI SDK for others.
    """
    # By default Claude models use the native Anthropic SDK. Setting
    # LLM_FORCE_OPENAI_PROXY=1 routes them through the OpenAI-compatible
    # endpoint instead (e.g. the cctq proxy that can host claude-* models),
    # which lets a single OPENAI_BASE_URL/OPENAI_API_KEY serve both vendors.
    # Default behaviour is unchanged so existing experiments still reproduce.
    _force_openai = os.environ.get("LLM_FORCE_OPENAI_PROXY") == "1"
    if model.startswith("claude") and not _force_openai:
        import anthropic

        client = anthropic.Anthropic(max_retries=0)
        create_kwargs = {
            "model": model,
            "max_tokens": 8192,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "timeout": 120.0,
        }
        response = client.messages.create(**create_kwargs)
        if not response.content or not hasattr(response.content[0], "text"):
            raise ValueError(f"Unexpected API response: {response.content}")
        return response.content[0].text

    from openai import OpenAI

    base_url = os.environ.get("OPENAI_BASE_URL")
    client = OpenAI(base_url=base_url, max_retries=0) if base_url else OpenAI(max_retries=0)
    create_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 8192,
        "temperature": temperature,
        "timeout": 300.0,
        # Native streaming. The cctq.ai proxy ignores stream=False for
        # reasoning models and buffers server-side, which intermittently
        # yields an empty completion (choices:[], completion_tokens=0).
        # Consuming the stream incrementally avoids that failure mode.
        "stream": True,
    }
    if reasoning_effort:
        create_kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
    return _consume_openai_stream(client, create_kwargs)

def _consume_openai_stream(client, create_kwargs: dict) -> str:
    """Consume an OpenAI streaming completion into a single text string.

    Handles two transport shapes the proxy may return:
      1. A native Stream object (SDK-parsed chunks) — the normal case.
      2. A raw SSE string (proxy bypassed the SDK parser) — fallback.

    Empty-choice chunks (reasoning deltas, trailing usage-only chunks) are
    skipped rather than treated as errors.
    """
    import json as _json

    response = client.chat.completions.create(**create_kwargs)

    # Fallback: proxy returned the whole SSE body as a plain string.
    if isinstance(response, str):
        content_parts: List[str] = []
        for line in response.split("\n"):
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                chunk = _json.loads(payload)
            except _json.JSONDecodeError:
                continue
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content"):
                    content_parts.append(delta["content"])
        text = "".join(content_parts)
        if not text:
            raise ValueError(
                f"Could not parse streaming response: {response[:200]}"
            )
        return text

    # Native streaming iterator.
    content_parts = []
    for chunk in response:
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        delta = choices[0].delta
        piece = getattr(delta, "content", None)
        if piece:
            content_parts.append(piece)
    text = "".join(content_parts)
    if not text:
        raise ValueError("Empty streaming response (no content chunks)")
    return text

def _auto_features_to_matrix(
    descriptions: Dict[Hand, str],
    hands: List[Hand],
    feature_names: List[str],
) -> Tuple[np.ndarray, int]:
    """Convert auto-feature scored descriptions to feature matrix.

    Each description is a single line of KEY=VALUE pairs (e.g.,
    "FEAT1=0.850 FEAT2=0.320"). The index prefix is already stripped
    by generate_hand_descriptions/_parse_descriptions.

    Args:
        descriptions: {hand: raw_text_line} from generate_hand_descriptions.
        hands: Ordered list of hands.
        feature_names: Feature names from Phase 1.

    Returns:
        (features matrix [n_hands, n_features], parsed_count)
    """
    upper_names = [n.upper() for n in feature_names]
    name_set = set(upper_names)
    n_features = len(upper_names)
    features = np.full((len(hands), n_features), 0.5)
    kv_pattern = re.compile(r"(\w+)=([\d.]+)")

    parsed_count = 0
    for i, hand in enumerate(hands):
        desc = descriptions.get(hand, "")
        matched: Dict[str, float] = {}
        for key, val_str in kv_pattern.findall(desc):
            if key.upper() in name_set:
                try:
                    val = float(val_str)
                    if 0.0 <= val <= 1.0:
                        matched[key.upper()] = round(val, 3)
                except ValueError:
                    pass
        if matched:
            parsed_count += 1
            for j, fname in enumerate(upper_names):
                if fname in matched:
                    features[i, j] = matched[fname]

    logger.info(
        "Auto-feature parsing: %d/%d hands parsed successfully",
        parsed_count, len(hands),
    )
    return features, parsed_count

def select_features(
    features: np.ndarray,
    feature_names: List[str],
    min_std: float = 0.10,
    max_corr: float = 0.90,
    min_features: int = 2,
    discriminative_target: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str], Dict[str, str]]:
    """Post-hoc feature selection: drop low-variance and correlated features.

    Args:
        features: [n_hands, n_features] matrix.
        feature_names: Names parallel to columns.
        min_std: Minimum per-column standard deviation.
        max_corr: Maximum allowed absolute Pearson correlation.
        min_features: Minimum features to retain (fallback).
        discriminative_target: Optional [n_hands] vector (e.g. equity). When
            given, the de-correlation tie-break keeps the member of a
            correlated pair that is MORE correlated with the target (instead
            of the larger-std one), and the min_features fallback ranks by
            |target-corr| (instead of std). This prevents equity-aligned
            signal features from being dropped as "redundant" on
            equity-dominated boards. When None, behaviour is unchanged.

    Returns:
        (selected_features, selected_names, drop_log)
    """
    n_features = len(feature_names)
    stds = features.std(axis=0)
    keep_mask = [True] * n_features
    drop_log: Dict[str, str] = {}

    # Per-feature |correlation with discriminative target| (0.0 if unavailable).
    tgt_corr = np.zeros(n_features)
    if discriminative_target is not None:
        tgt = np.asarray(discriminative_target, dtype=float).ravel()
        tgt_std = float(np.std(tgt))
        for j in range(n_features):
            if stds[j] > 0.0 and tgt_std > 0.0:
                r = np.corrcoef(features[:, j], tgt)[0, 1]
                tgt_corr[j] = 0.0 if np.isnan(r) else abs(float(r))

    for j in range(n_features):
        if stds[j] < min_std:
            keep_mask[j] = False
            drop_log[feature_names[j]] = (
                f"low_variance (std={stds[j]:.4f} < {min_std})"
            )

    surviving = [j for j in range(n_features) if keep_mask[j]]
    if len(surviving) > 1:
        sub = features[:, surviving]
        corr = np.corrcoef(sub.T)
        for a in range(len(surviving)):
            for b in range(a + 1, len(surviving)):
                j_a, j_b = surviving[a], surviving[b]
                if not keep_mask[j_a] or not keep_mask[j_b]:
                    continue
                r = abs(corr[a, b])
                if r > max_corr:
                    if discriminative_target is not None:
                        # Keep the member more aligned with the target.
                        if tgt_corr[j_a] < tgt_corr[j_b]:
                            drop_j, keep_j = j_a, j_b
                        else:
                            drop_j, keep_j = j_b, j_a
                    elif stds[j_a] < stds[j_b]:
                        drop_j, keep_j = j_a, j_b
                    else:
                        drop_j, keep_j = j_b, j_a
                    keep_mask[drop_j] = False
                    drop_log[feature_names[drop_j]] = (
                        f"correlated with {feature_names[keep_j]} "
                        f"(|r|={r:.4f} > {max_corr})"
                    )

    selected = [j for j in range(n_features) if keep_mask[j]]
    if len(selected) < min_features:
        if discriminative_target is not None:
            ranked = sorted(range(n_features), key=lambda j: -tgt_corr[j])
            fallback_by = "|target-corr|"
        else:
            ranked = sorted(range(n_features), key=lambda j: -stds[j])
            fallback_by = "std"
        selected = ranked[:min_features]
        drop_log["__fallback__"] = (
            f"only {sum(keep_mask)} survived, "
            f"fell back to top-{min_features} by {fallback_by}"
        )

    selected_names = [feature_names[j] for j in selected]
    logger.info(
        "Feature selection: %d/%d kept %s",
        len(selected), n_features, selected_names,
    )
    for name, reason in drop_log.items():
        if name != "__fallback__":
            logger.info("  Dropped %s: %s", name, reason)

    return features[:, selected], selected_names, drop_log
