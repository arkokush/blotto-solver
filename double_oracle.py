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
) -> DOResult:
    """Run double oracle until gap <= tol or max_iters iterations.

    Args:
        payoff_fn: payoff_fn(r, c) is the row player's payoff (row maximizes).
        row_oracle: given the column player's restricted mix, returns a best
            response row strategy and its expected payoff (to the row player).
        col_oracle: given the row player's restricted mix, returns a best
            response column strategy and the row player's expected payoff against it.
        init_row, init_col: starting strategy for each player.
        tol: stop once gap <= tol.
        max_iters: iteration cap; result.converged reports which condition stopped it.
        verbose: print one line per iteration.

    Returns:
        DOResult with the final restricted equilibrium and per-iteration history.
    """
    # STAGE 5: implement. Keep the restricted payoff matrix and grow it by one
    # row/column per new strategy instead of rebuilding it every iteration.
    raise NotImplementedError


def default_allocation(cfg: GameConfig) -> tuple[int, ...]:
    """Spread units as evenly as possible, extra units going to the highest-value towers."""
    base, extra = divmod(cfg.n_units, cfg.n_towers)
    return tuple(base + (1 if i >= cfg.n_towers - extra else 0) for i in range(cfg.n_towers))


def blotto_double_oracle(
    cfg: GameConfig,
    objective: str = "margin",
    init: tuple[int, ...] | None = None,
    tol: float = 1e-6,
    max_iters: int = 1000,
    verbose: bool = False,
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
    return double_oracle(payoff_fn, row_oracle, col_oracle, start, start, tol, max_iters, verbose)
