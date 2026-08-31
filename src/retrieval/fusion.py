"""B3: Reciprocal Rank Fusion.

Combines several rankings of the same candidate pool into one. RRF is used
here rather than a weighted sum of raw scores because the three rankers
produce numbers on incomparable scales: BM25 is unbounded and depends on
corpus statistics, cosine similarity sits in [-1, 1], and the category
signal is a small integer depth. Normalising those against each other
means picking arbitrary constants that need retuning whenever any ranker
changes. RRF sidesteps that by throwing the magnitudes away and keeping
only the ordering, which is the part that is actually comparable.

    score(d) = sum over rankers r of  weight_r / (k + rank_r(d))

``k`` damps the influence of the very top positions, so one ranker cannot
dominate the fusion on the strength of a single confident guess. k=60 is
the value from the original Cormack et al. paper and is a sane default;
it is exposed so B5 tuning can move it.
"""
from __future__ import annotations

import numpy as np

RRF_K = 60.0


def rank_positions(scores: np.ndarray) -> np.ndarray:
    """Convert a score array into 1-based ranks (highest score = rank 1).

    Ties take the same treatment as ``argsort`` gives them, which is
    arbitrary but stable, and RRF is insensitive to tie order at depth.
    """
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float32)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float32)
    return ranks


def reciprocal_rank_fusion(
    score_lists: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
    k: float = RRF_K,
) -> np.ndarray:
    """Fuse several score arrays over the same pool into one fused score.

    Every array in ``score_lists`` must be parallel to the same candidate
    pool. A ranker that contributed nothing this turn (all zeros, e.g. the
    dense index on a query with no usable text) is skipped rather than
    allowed to inject its arbitrary tie ordering into the fusion.
    """
    if not score_lists:
        raise ValueError("reciprocal_rank_fusion needs at least one ranker")
    weights = weights or {}
    fused: np.ndarray | None = None
    for name, scores in score_lists.items():
        if scores is None or not len(scores) or not np.any(scores):
            continue
        weight = float(weights.get(name, 1.0))
        if weight == 0.0:
            continue
        contribution = weight / (k + rank_positions(scores))
        fused = contribution if fused is None else fused + contribution
    if fused is None:
        # Every ranker abstained. Return zeros so the caller falls back to
        # its own ordering rather than crashing mid-session.
        length = len(next(iter(score_lists.values())))
        return np.zeros(length, dtype=np.float32)
    return fused
