# MIT License
# Copyright (c) 2024 MANTIS
"""
Deterministic sybil-cluster detection and salience collapse.

Motivation
----------
On each weight-calculation cycle, every challenge's scorer fits an L2 (or
ElasticNet) logistic regression over miner embeddings.  When a cluster of
N near-identical clones submits the same predictions, the loss surface is
*flat* in the cluster's coefficient subspace: many splits of the cluster's
total weight across its members yield nearly identical loss.

The exact convergence point of L-BFGS in that flat subspace is sensitive
to ULP-level numerical noise, which differs between validators because
runtime CPU dispatch in BLAS picks different SIMD micro-kernels on
different hardware.  Two validators with byte-identical input data can
therefore end up assigning the cluster's weight to *different* clones
(say, clone17 on validator A vs clone23 on validator B).  The within-
cluster total agrees, but the per-key allocation flickers — and that
flicker dominates Vtrust loss in competitive challenges.

This module provides the fix:

  1. ``find_clusters`` discovers tight clusters of clones from raw
     embedding histories using a *conservative* pairwise criterion.
  2. ``select_representatives`` picks the lex-min hotkey of each cluster
     as the canonical representative (purely deterministic, independent
     of UID re-registration, metagraph sync timing, or data noise).
  3. ``collapse_salience`` re-attributes every cluster member's salience
     to the cluster's representative, summing the masses.

Conservatism
------------
The pairwise match criterion is intentionally strict:

  - Two miners are linked iff their per-timestep embeddings agree to
    within ``MATCH_EPS`` on at least ``MIN_MATCH_FRACTION`` of joint-active
    timesteps (default 1e-3 / 99 %).
  - Both miners must have at least ``MIN_SUBMISSIONS`` non-zero rows and
    they must share at least ``MIN_JOINT`` joint-active timesteps.
  - Linkage is single-linkage (transitive closure over pairwise matches).

Two miners running materially different strategies will not survive the
99 % match-fraction at ε ≈ one float16 ULP.  False-negatives (missing a
true clone pair) are far cheaper than false-positives (merging non-clones)
for the Vtrust use case, so the defaults err on the strict side.

Determinism notes
-----------------
Drand decryption misses on validator X drop entire timesteps for *all*
miners simultaneously, so the joint-active mask between any two miners
differs by at most a few percent across validators (and the drop is the
same whole rows for both miners in the pair).  Decryption itself is
byte-deterministic, so embedding values on jointly-decrypted timesteps
are bit-identical between validators.  Therefore the pairwise
match-fraction and the resulting union-find structure are stable across
validators with high probability.  The lex-min representative pick is
purely deterministic given the cluster set.
"""

from __future__ import annotations

import logging
from typing import Dict, FrozenSet, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# -- Defaults (conservative) ------------------------------------------------

# Per-element absolute difference allowed between two miners' embeddings on
# a single timestep for that timestep to count as a "match".  One float16
# ULP at unit scale is ~1e-3; embeddings are stored as float16, so this is
# the tightest meaningful tolerance.
MATCH_EPS: float = 1e-3

# Fraction of joint-active timesteps that must satisfy the per-timestep
# match criterion for two miners to be linked.
MIN_MATCH_FRACTION: float = 0.99

# Minimum number of non-zero submission rows a miner must have to be
# considered for clustering at all.
MIN_SUBMISSIONS: int = 200

# Minimum number of joint-active timesteps required to evaluate a pair.
MIN_JOINT: int = 200

# Pre-filter: reject pair candidates whose per-dimension means over their
# own non-zero rows differ by more than this anywhere.  Much looser than
# MATCH_EPS to avoid filtering out true clone pairs whose submission
# windows do not fully overlap.  Set to None to disable.
MEAN_PREFILTER_EPS: float | None = 0.05


def _miner_means(X: np.ndarray, nz_mask: np.ndarray, miners: np.ndarray) -> np.ndarray:
    """Per-dimension mean over each miner's non-zero rows.

    X: (T, H, D) float
    nz_mask: (T, H) bool
    miners: (M,) int indices into the H axis
    Returns: (M, D) float
    """
    M = miners.size
    D = X.shape[2]
    out = np.zeros((M, D), dtype=np.float32)
    for k in range(M):
        i = int(miners[k])
        mask = nz_mask[:, i]
        if mask.any():
            out[k] = X[mask, i, :].astype(np.float32, copy=False).mean(axis=0)
    return out


def find_clusters(
    hist: Tuple[np.ndarray, Dict[str, int]],
    *,
    dim: int | None = None,
    eps: float = MATCH_EPS,
    match_fraction: float = MIN_MATCH_FRACTION,
    min_submissions: int = MIN_SUBMISSIONS,
    min_joint: int = MIN_JOINT,
    mean_prefilter_eps: float | None = MEAN_PREFILTER_EPS,
) -> List[FrozenSet[str]]:
    """Find conservative single-linkage clusters of near-identical miners.

    ``hist`` is the ``(X_flat, hk2idx)`` tuple already used by every scorer:
      X_flat is (T, H * D) and ``hk2idx`` maps hotkey strings to columns
      [0, H).  ``dim`` is the per-miner D; if None, inferred as
      ``X_flat.shape[1] // H``.

    Returns a list of non-singleton clusters (each a frozenset of hotkey
    strings).  Singletons are intentionally omitted — they are implicitly
    their own representatives.
    """
    X_flat, hk2idx = hist
    if not hk2idx or X_flat is None:
        return []
    X_flat = np.asarray(X_flat)
    if X_flat.ndim != 2:
        return []
    H = len(hk2idx)
    if H <= 1 or X_flat.shape[1] % H != 0:
        return []
    D = int(dim) if dim is not None else X_flat.shape[1] // H
    if D <= 0 or D * H != X_flat.shape[1]:
        return []
    T = int(X_flat.shape[0])
    if T < min_joint:
        return []

    # View as (T, H, D); copy is forced to float32 only when materialised in pair work.
    X = X_flat.reshape(T, H, D)

    # Per-(timestep, miner) submission mask.
    nz_mask = np.any(X != 0, axis=2)  # (T, H) bool
    sub_counts = nz_mask.sum(axis=0)  # (H,) int
    active = np.where(sub_counts >= min_submissions)[0]
    if active.size < 2:
        return []

    idx2hk: List[str | None] = [None] * H
    for hk, i in hk2idx.items():
        if 0 <= i < H:
            idx2hk[i] = hk


    means = _miner_means(X, nz_mask, active) if mean_prefilter_eps is not None else None

    # Union-find restricted to active miners.
    parent: Dict[int, int] = {int(i): int(i) for i in active}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Stable: always make the smaller index the root so that the union-find
        # state itself is invariant to discovery order.
        parent[max(ra, rb)] = min(ra, rb)

    n_active = int(active.size)
    eps_f32 = np.float32(eps)
    n_pairs_checked = 0
    n_pairs_matched = 0
    n_pairs_skipped_mean = 0
    n_pairs_skipped_joint = 0

    for ii in range(n_active):
        i = int(active[ii])
        nz_i = nz_mask[:, i]
        Xi = X[:, i, :]
        for jj in range(ii + 1, n_active):
            j = int(active[jj])
            if means is not None:
                if np.max(np.abs(means[ii] - means[jj])) > mean_prefilter_eps:
                    n_pairs_skipped_mean += 1
                    continue
            joint = nz_i & nz_mask[:, j]
            n_joint = int(joint.sum())
            if n_joint < min_joint:
                n_pairs_skipped_joint += 1
                continue
            n_pairs_checked += 1
            # Per-row max-abs-diff across the D dims, then count rows
            # where the row is within ε.
            Xi_j = Xi[joint].astype(np.float32, copy=False)
            Xj_j = X[joint, j, :].astype(np.float32, copy=False)
            row_max = np.max(np.abs(Xi_j - Xj_j), axis=1)
            n_match = int(np.count_nonzero(row_max < eps_f32))
            if n_match >= int(np.ceil(match_fraction * n_joint)):
                union(i, j)
                n_pairs_matched += 1

    by_root: Dict[int, List[int]] = {}
    for i in active:
        i_int = int(i)
        r = find(i_int)
        by_root.setdefault(r, []).append(i_int)

    out: List[FrozenSet[str]] = []
    for members in by_root.values():
        if len(members) < 2:
            continue
        hks = frozenset(idx2hk[i] for i in members if idx2hk[i] is not None)
        if len(hks) >= 2:
            out.append(hks)

    if out or n_pairs_checked > 0:
        logger.info(
            "cluster scan: H=%d active=%d pairs_checked=%d matched=%d "
            "skipped_mean=%d skipped_joint=%d clusters=%d total_members=%d",
            H, n_active, n_pairs_checked, n_pairs_matched,
            n_pairs_skipped_mean, n_pairs_skipped_joint,
            len(out), sum(len(c) for c in out),
        )

    return out


def select_representatives(clusters: List[FrozenSet[str]]) -> Dict[str, str]:
    """Pick the lex-min hotkey of each cluster as its representative.


    Returns ``{member_hk: representative_hk}`` for every hotkey in any
    non-singleton cluster.  Hotkeys not in the map are implicitly their
    own representative.
    """
    out: Dict[str, str] = {}
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        rep = min(cluster)
        for hk in cluster:
            out[hk] = rep
    return out


def collapse_salience(
    salience: Dict[str, float],
    representatives: Dict[str, str],
) -> Dict[str, float]:
    """Re-attribute every cluster member's salience to its representative.

    Conservation of mass: ``sum(out.values()) == sum(salience.values())``
    up to float-add ordering.

    Hotkeys absent from ``representatives`` are passed through unchanged.
    Cluster non-representatives are dropped (their value is summed into
    the representative's bucket); the representative may have been absent
    from ``salience`` originally and still appears in the output if any
    cluster-mate had positive salience.
    """
    if not representatives or not salience:
        return dict(salience)
    out: Dict[str, float] = {}
    # Iterate in sorted key order so the float-add sequence is itself
    # deterministic (irrelevant to bit-equality of the SUM, but keeps
    # any rounding artefacts identical across machines/runs).
    for hk in sorted(salience.keys()):
        rep = representatives.get(hk, hk)
        out[rep] = out.get(rep, 0.0) + float(salience[hk])
    return out
