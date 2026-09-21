"""Exact solution of small Blotto instances by enumerating every pure strategy.

This plays the role Kuhn and Leduc played for the poker solver: a game small
enough to solve directly, so the double oracle output can be checked against it.
"""

from __future__ import annotations
from lp_solver import solve_zero_sum
from payoff import GameConfig, enumerate_allocations, payoff_matrix

from dataclasses import dataclass

import numpy as np

from payoff import GameConfig


@dataclass
class FullSolution:
    value: float
    allocations: np.ndarray  # (n_pure_strategies, n_towers); both players share this set
    row_probs: np.ndarray
    col_probs: np.ndarray


def solve_full_game(cfg: GameConfig, objective: str = "margin") -> FullSolution:
    """Build the full payoff matrix over all allocations and solve it by LP.

    Only feasible when cfg.n_pure_strategies is small (hundreds, maybe a few thousand).
    """
    allocs = enumerate_allocations(cfg)
    A = payoff_matrix(allocs,allocs,cfg,objective)
    value,p,q = solve_zero_sum(A)

    return FullSolution(value=value, allocations=allocs, row_probs=p, col_probs=q)
