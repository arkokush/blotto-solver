"""Double oracle for two-player zero-sum games.

``double_oracle`` is generic: strategies are any hashable objects, and the game
is reached only through a payoff function and two best-response oracles. The
same loop runs on plain matrix games (tests) and on Blotto (DP oracle).

Convergence is measured by the two best-response values:

    col_br_value = min_c  E_{r ~ p}[u(r, c)]   <= game value <=
    row_br_value = max_r  E_{c ~ q}[u(r, c)]

where (p, q) is the restricted game's equilibrium. gap = row_br_value - col_br_value
is the exploitability of (p, q) in the full game. It is >= 0, and it is 0 exactly
when (p, q) is an equilibrium of the full game. See DECISIONS.md (D7).
"""

from __future__ import annotations
from lp_solver import solve_zero_sum

import time
from dataclasses import dataclass, field
from typing import Callable, Hashable, Sequence

import numpy as np

from best_response import best_response
from payoff import GameConfig, col_weights, payoff, row_weights

# oracle(opponent_strategies, opponent_probs) -> (best_response_strategy, value_to_row_player)
Oracle = Callable[[Sequence[Hashable], np.ndarray], tuple[Hashable, float]]


@dataclass
class IterationRecord:
    iteration: int
    restricted_value: float  # value of the restricted game
    row_br_value: float  # upper bound on the full game value
    col_br_value: float  # lower bound on the full game value
    gap: float  # row_br_value - col_br_value
    n_row: int  # restricted strategy set sizes
    n_col: int
    elapsed: float  # seconds since start


@dataclass
class DOResult:
    row_strats: list
    row_probs: np.ndarray
    col_strats: list
    col_probs: np.ndarray
    value: float
    converged: bool
    history: list[IterationRecord] = field(default_factory=list)


def double_oracle(
    payoff_fn: Callable[[Hashable, Hashable], float],
    row_oracle: Oracle,
    col_oracle: Oracle,
    init_row: Hashable,
    init_col: Hashable,
    tol: float = 1e-6,
    max_iters: int = 1000,
    verbose: bool = False,
    prune: bool = False,
    use_dual: bool = False,
    patience: int = 25,
    on_iteration: Callable[[DOResult], None] | None = None,
    checkpoint_every: int = 25,
) -> DOResult:
    """Run double oracle until gap <= tol or max_iters iterations.

    Args:
        payoff_fn: payoff_fn(r, c) is the row player's payoff (row maximizes).
        row_oracle: given the column player's restricted mix, returns a best
            response row strategy and its expected payoff (to the row player).
        col_oracle: given the row player's restricted mix, returns a best
            response column strategy and the row player's expected payoff against it.
        init_row, init_col: starting strategy for each player, or a list of them.
            Warm-starting from a coarser solve saves a great many iterations.
        tol: stop once gap <= tol.
        max_iters: iteration cap; result.converged reports which condition stopped it.
        verbose: print one line per iteration.
        prune: drop strategies that the restricted equilibrium has given zero weight
            for `patience` consecutive iterations. The LP's cost grows like k^3.5, so
            this is what makes 1-soldier resolution feasible. Dropping a strategy the
            moment it hits zero weight makes the loop cycle (the oracle re-adds it at
            once), hence the patience counter. This gives up the textbook monotone-set
            termination argument, but the reported gap stays a valid certificate: it is
            measured by exact best responses against the mixture actually being played,
            however that mixture was found.
        patience: how many consecutive zero-weight iterations before a strategy is
            dropped.
        on_iteration: called with a snapshot every `checkpoint_every` iterations, so a
            long run can be saved (and inspected, or stopped) before it converges.
        use_dual: read q from the row LP's shadow prices instead of solving a
            second LP. Halves the LP work per iteration.

    Returns:
        DOResult with the final restricted equilibrium and per-iteration history.
    """
    start = time.time()
    row_strats = list(init_row) if isinstance(init_row, list) else [init_row]
    col_strats = list(init_col) if isinstance(init_col, list) else [init_col]
    A = np.array([[payoff_fn(r, c) for c in col_strats] for r in row_strats])
    zero_r = [0] * len(row_strats)
    zero_c = [0] * len(col_strats)
    history = []
    converged = False
    value, p, q = 0.0, np.array([1.0]), np.array([1.0])

    for it in range(max_iters):
        value, p, q = solve_zero_sum(A, use_dual=use_dual)

        row_br, row_br_value = row_oracle(col_strats, q)
        col_br, col_br_value = col_oracle(row_strats, p)

        gap = row_br_value - col_br_value

        history.append(IterationRecord(
            iteration=it, restricted_value=value,
            row_br_value=row_br_value, col_br_value=col_br_value, gap=gap,
            n_row=len(row_strats), n_col=len(col_strats), elapsed=time.time() - start,
        ))

        if verbose:
            print(f"it {it:3d}  value {value:9.4f}  gap {gap:10.6f}  "
                  f"|R|={len(row_strats)} |C|={len(col_strats)}")

        if on_iteration is not None and it % checkpoint_every == 0:
            on_iteration(DOResult(row_strats, p, col_strats, q, value, False, history))

        if gap <= tol:
            converged = True
            break

        zero_r = [0 if p[i] > 1e-9 else zero_r[i] + 1 for i in range(len(row_strats))]
        zero_c = [0 if q[j] > 1e-9 else zero_c[j] + 1 for j in range(len(col_strats))]

        if prune:
            keep_r = [i for i in range(len(row_strats)) if zero_r[i] < patience]
            keep_c = [j for j in range(len(col_strats)) if zero_c[j] < patience]
            if len(keep_r) < len(row_strats) or len(keep_c) < len(col_strats):
                A = A[np.ix_(keep_r, keep_c)]
                row_strats = [row_strats[i] for i in keep_r]
                col_strats = [col_strats[j] for j in keep_c]
                zero_r = [zero_r[i] for i in keep_r]
                zero_c = [zero_c[j] for j in keep_c]
                p, q = p[keep_r], q[keep_c]

        added = False
        if row_br not in row_strats:
            row_strats.append(row_br)
            zero_r.append(0)
            new_row = np.array([[payoff_fn(row_br, c) for c in col_strats]])
            A = np.vstack([A,new_row])
            added = True

        if col_br not in col_strats:
            col_strats.append(col_br)
            zero_c.append(0)
            new_col = np.array([[payoff_fn(r, col_br)] for r in row_strats])
            A = np.hstack([A, new_col])
            added = True

        if not added:
            break
    return DOResult(row_strats, p, col_strats, q, value, converged, history)


def default_allocation(cfg: GameConfig) -> tuple[int, ...]:
    """Spread units as evenly as possible, extra units going to the highest-value towers."""
    base, extra = divmod(cfg.n_units, cfg.n_towers)
    return tuple(base + (1 if i >= cfg.n_towers - extra else 0) for i in range(cfg.n_towers))


def blotto_double_oracle(
    cfg: GameConfig,
    objective: str = "margin",
    init: tuple[int, ...] | list[tuple[int, ...]] | None = None,
    tol: float = 1e-6,
    max_iters: int = 1000,
    verbose: bool = False,
    prune: bool = False,
    use_dual: bool = False,
    patience: int = 25,
    on_iteration: Callable[[DOResult], None] | None = None,
    checkpoint_every: int = 25,
) -> DOResult:
    """Double oracle on Blotto with the DP best response. Strategies are tuples of units."""
    rw, cw = row_weights(objective, cfg), col_weights(objective, cfg)

    def payoff_fn(r, c):
        return payoff(np.array(r), np.array(c), cfg, objective)

    def row_oracle(col_strats, col_probs):
        x, v = best_response(np.array(col_strats), col_probs, cfg, rw)
        return tuple(int(a) for a in x), v

    def col_oracle(row_strats, row_probs):
        y, v = best_response(np.array(row_strats), row_probs, cfg, cw)
        # The column player's utility is minus the row payoff.
        return tuple(int(a) for a in y), -v

    start = init if init is not None else default_allocation(cfg)
    return double_oracle(payoff_fn, row_oracle, col_oracle, start, start, tol, max_iters,
                         verbose, prune=prune, use_dual=use_dual, patience=patience,
                         on_iteration=on_iteration, checkpoint_every=checkpoint_every)
