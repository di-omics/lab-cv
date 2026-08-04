"""Rollout scoring - how far off was the predicted state, in bench units.

Companion to `eval/metrics.py`. That file scores perception (did you find the
instance, did you name it right); this one scores a *rollout*: a model was given
observations and a commanded action sequence, and predicted where the state would
end up. The unit is the unit of the bench - microlitres, millimetres - not
pixels, because a plate that seats 0.4 mm off is a plate that seats, and a
picture-perfect prediction that misses the tolerance is a failed run.

    terminal_error         per-episode error at the FINAL step
    in_tolerance           per-episode hit, tolerance-normalised, joint over fields
    trajectory_rmse        per-episode RMSE over the whole predicted trajectory
    first_divergence_step  the step at which a prediction left tolerance
    time_to_tolerance_cdf  empirical CDF of steps-to-inside-tolerance

**selective-prediction metrics live in `plr_lr.learn.selective`; do not add a
risk-coverage function here.** Risk-coverage, AURC, E-AURC, ADA, abstention
credit and their intervals have exactly one implementation in this portfolio,
and a second one here would be a second answer to the same question. Measurement
vocabularies drifting across repos is a documented failure here, not a
hypothetical: the moment two files can both produce "the AURC", every published
number needs a provenance note explaining which one it came from.

Three things this module will not do, each of which has produced a confident
wrong number in this portfolio before:

* **No max-over-steps.** Success is read off the final state only. Scanning a
  trajectory for its best step and reporting that is selection, and it turns a
  model that passes through the right answer on its way somewhere else into a
  model that got the right answer.
* **No percentile- or quantile-derived thresholds.** A threshold fitted to the
  data it then judges cannot fail. Every tolerance here is an argument.
* **No mean of a heavy-tailed error.** `terminal_error` returns the per-episode
  vector; summarise it with a median and an interquartile range, which is what
  the harness downstream does. Nothing here averages it for you.

The one max in this file is across *fields* of a single state at a single step -
`max_d |s_hat[d] - s[d]| / tau[d]` - which is the joint tolerance criterion
itself, not a selection over anything.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def _states(a) -> np.ndarray:
    """Coerce (N,) or (N, D) into (N, D). A 1-D state is one field, not one episode."""
    arr = np.asarray(a, float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"expected states shaped (N,) or (N, D), got {arr.shape}")
    return arr


def _trajectories(a) -> np.ndarray:
    """Coerce (N, K) or (N, K, D) into (N, K, D)."""
    arr = np.asarray(a, float)
    if arr.ndim == 2:
        arr = arr.reshape(arr.shape[0], arr.shape[1], 1)
    if arr.ndim != 3:
        raise ValueError(f"expected trajectories shaped (N, K) or (N, K, D), got {arr.shape}")
    return arr


def _tolerance(tol, n_fields: int) -> np.ndarray:
    arr = np.asarray(tol, float).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, n_fields)
    if arr.size != n_fields:
        raise ValueError(f"tolerance has {arr.size} entries for {n_fields} state fields")
    if not np.all(arr > 0):
        raise ValueError("every tolerance must be strictly positive; a zero tolerance makes "
                         "every prediction a miss and every division a divide-by-zero")
    return arr


def terminal_error(pred, truth) -> np.ndarray:
    """Per-episode absolute error at the final state, max over fields, native units.

    Returned per episode rather than summarised. The error distribution on a
    bench task is heavy-tailed - most episodes land close, a few land nowhere -
    so a mean is dominated by the tail and reads as a much worse model than the
    one you have, or a much better one, depending on which tail you drew.
    """
    p, t = _states(pred), _states(truth)
    if p.shape != t.shape:
        raise ValueError(f"prediction {p.shape} and truth {t.shape} must have the same shape")
    return np.max(np.abs(p - t), axis=1)


def in_tolerance(pred, truth, tolerance) -> np.ndarray:
    """Per-episode hit under the joint, per-field tolerance criterion.

        hit_i = max_d ( |s_hat_i[d] - s_i[d]| / tau[d] ) <= 1

    Each field is divided by its own tolerance before the max, so a criterion
    like "0.5 mm AND 0.5 deg" becomes one number with no weighting choice to
    argue about, and a model cannot buy millimetres with degrees.
    """
    p, t = _states(pred), _states(truth)
    if p.shape != t.shape:
        raise ValueError(f"prediction {p.shape} and truth {t.shape} must have the same shape")
    tau = _tolerance(tolerance, p.shape[1])
    return np.max(np.abs(p - t) / tau, axis=1) <= 1.0


def trajectory_rmse(pred, truth) -> np.ndarray:
    """Per-episode RMSE over every predicted step and field.

    Reported alongside `terminal_error`, never instead of it. A model can hold a
    low trajectory RMSE while missing the terminal state - it tracks the shape
    and lands in the wrong place - and the terminal state is the one the robot
    acts on.
    """
    p, t = _trajectories(pred), _trajectories(truth)
    if p.shape != t.shape:
        raise ValueError(f"prediction {p.shape} and truth {t.shape} must have the same shape")
    return np.sqrt(np.mean((p - t) ** 2, axis=(1, 2)))


def first_divergence_step(pred, truth, tolerance) -> List[Optional[int]]:
    """First step index at which an episode's prediction left tolerance.

    `None` for an episode that never left it, which is deliberately not `K` and
    not `-1`: a sentinel integer gets averaged by accident, and "never diverged"
    then reads as "diverged at the last step". Scanned with an explicit loop
    rather than an argmax over a boolean array, because argmax on an all-False
    row silently returns 0 - the most divergent answer possible for the least
    divergent episode.
    """
    p, t = _trajectories(pred), _trajectories(truth)
    if p.shape != t.shape:
        raise ValueError(f"prediction {p.shape} and truth {t.shape} must have the same shape")
    tau = _tolerance(tolerance, p.shape[2])
    out: List[Optional[int]] = []
    for i in range(p.shape[0]):
        step: Optional[int] = None
        for k in range(p.shape[1]):
            if float(np.max(np.abs(p[i, k] - t[i, k]) / tau)) > 1.0:
                step = k
                break
        out.append(step)
    return out


def time_to_tolerance(pred, truth, tolerance) -> List[Optional[int]]:
    """First step index at which an episode's prediction is inside tolerance.

    The mirror of `first_divergence_step`, and `None` for the same reason: an
    episode that never got there is not an episode that got there late.
    """
    p, t = _trajectories(pred), _trajectories(truth)
    if p.shape != t.shape:
        raise ValueError(f"prediction {p.shape} and truth {t.shape} must have the same shape")
    tau = _tolerance(tolerance, p.shape[2])
    out: List[Optional[int]] = []
    for i in range(p.shape[0]):
        step: Optional[int] = None
        for k in range(p.shape[1]):
            if float(np.max(np.abs(p[i, k] - t[i, k]) / tau)) <= 1.0:
                step = k
                break
        out.append(step)
    return out


def time_to_tolerance_cdf(steps: Sequence[Optional[int]],
                          n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Empirical CDF of steps-to-inside-tolerance, over ALL episodes.

    Returns `(k, fraction)` for `k = 0 .. n_steps - 1`, where `fraction[k]` is
    the share of episodes that were inside tolerance by step `k`.

    The denominator is every episode, including the ones that never arrived.
    Dropping them - the obvious convenience - conditions the curve on success and
    turns a model that converges rarely but quickly into a model that converges
    quickly. That is the same mistake as excluding a refusal as missing data
    instead of scoring it, and it is the mistake this whole build exists to
    avoid. The curve therefore need not reach 1.0, and its ceiling is the
    convergence rate, readable straight off the plot.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive; there is no CDF over zero steps")
    total = len(steps)
    if total == 0:
        raise ValueError("refusing to build a CDF over zero episodes; nothing was measured")
    ks = np.arange(n_steps)
    counts = np.zeros(n_steps, float)
    for s in steps:
        if s is None:
            continue
        s = int(s)
        if s < 0:
            raise ValueError(f"negative step index {s} in the time-to-tolerance list")
        if s < n_steps:
            counts[s] += 1.0
    return ks, np.cumsum(counts) / float(total)
