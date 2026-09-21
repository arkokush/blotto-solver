"""Game definition and payoffs for the adjacency-bonus Blotto variant.

Allocations are integer numpy arrays of length ``n_towers`` measured in *units*
(``cfg.unit`` soldiers each) and summing to ``cfg.n_units``. Since both players
use the same unit, comparing unit counts is the same as comparing soldier counts.

Every objective we consider is a linear combination of four scores:

    T_me, B_me   tower points and adjacency bonus earned by "me"
    T_opp, B_opp the same for the opponent

Tower points are constant-sum (T_me + T_opp = total tower points), so a
player's utility can always be written as

    tower * T_me + own_bonus * B_me + opp_bonus * B_opp + const

and a single set of ``Weights`` describes it. See DECISIONS.md (D1, D5).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

OBJECTIVES = ("margin", "security")


@dataclass(frozen=True)
class GameConfig:
    """Parameters of one Blotto instance.

    Attributes:
        n_towers: number of towers.
        n_soldiers: soldiers per player.
        unit: soldiers per indivisible unit (discretization).
        bonus: points for each adjacent pair of towers won outright.
        tower_values: points per tower; defaults to 1, 2, ..., n_towers.
    """

    n_towers: int = 10
    n_soldiers: int = 100
    unit: int = 5
    bonus: float = 5.0
    tower_values: tuple[float, ...] | None = None

    def __post_init__(self):
        if self.n_soldiers % self.unit != 0:
            raise ValueError("n_soldiers must be a multiple of unit")
        if self.tower_values is None:
            object.__setattr__(
                self, "tower_values", tuple(float(v) for v in range(1, self.n_towers + 1))
            )
        if len(self.tower_values) != self.n_towers:
            raise ValueError("tower_values must have length n_towers")

    @property
    def n_units(self) -> int:
        return self.n_soldiers // self.unit

    @property
    def total_tower_points(self) -> float:
        return float(sum(self.tower_values))

    @property
    def n_pure_strategies(self) -> int:
        """Number of allocations: stars and bars, C(n_units + n_towers - 1, n_towers - 1)."""
        return comb(self.n_units + self.n_towers - 1, self.n_towers - 1)


@dataclass(frozen=True)
class Weights:
    """Utility = tower*T_me + own_bonus*B_me + opp_bonus*B_opp + const."""

    tower: float
    own_bonus: float
    opp_bonus: float
    const: float = 0.0


def row_weights(objective: str, cfg: GameConfig) -> Weights:
    """Row player's utility; this is also the entry of the payoff matrix.

    margin:   (T_me + B_me) - (T_opp + B_opp) = 2*T_me + B_me - B_opp - total
    security: T_me + B_me   (my own points; the opponent is assumed to minimize them)
    """
    total = cfg.total_tower_points
    if objective == "margin":
        return Weights(tower=2.0, own_bonus=1.0, opp_bonus=-1.0, const=-total)
    if objective == "security":
        return Weights(tower=1.0, own_bonus=1.0, opp_bonus=0.0)
    raise ValueError(f"unknown objective {objective!r}")


def col_weights(objective: str, cfg: GameConfig) -> Weights:
    """Column player's utility, written from the column player's perspective.

    It is always the negative of the row payoff (zero-sum):
    margin:   symmetric, same weights as the row player.
    security: -(T_row + B_row) = -(total - T_me) - B_opp = T_me - B_opp - total
    """
    total = cfg.total_tower_points
    if objective == "margin":
        return row_weights("margin", cfg)
    if objective == "security":
        return Weights(tower=1.0, own_bonus=0.0, opp_bonus=-1.0, const=-total)
    raise ValueError(f"unknown objective {objective!r}")


def score_components(x: np.ndarray, y: np.ndarray, cfg: GameConfig) -> tuple[float, float, float, float]:
    """Score one play of the game.

    Args:
        x, y: allocations (in units) of the two players.
        cfg: game parameters.

    Returns:
        (T_x, B_x, T_y, B_y): tower points and adjacency bonus for x and for y.
        Ties split a tower's points and never count toward a bonus. Each adjacent
        pair (i, i+1) won outright by a player earns that player cfg.bonus, and
        pairs overlap.
    """
    won_prev = 0
    T_x, B_x, T_y, B_y = 0,0,0,0
    bonus = cfg.bonus

    for i in range(len(x)):
        p = cfg.tower_values[i]
        if x[i] > y[i]:
            T_x += p
            B_x += bonus if won_prev == 1 else 0
            won_prev = 1

        elif x[i] < y[i]:
            T_y += p
            B_y += bonus if won_prev == -1 else 0
            won_prev = -1

        else:
            T_x += p/2
            T_y += p/2
            won_prev = 0

    return T_x, B_x, T_y, B_y




def utility(x: np.ndarray, y: np.ndarray, cfg: GameConfig, w: Weights) -> float:
    """Utility to the player holding x against y, under weights w."""
    t_x, b_x, _, b_y = score_components(x, y, cfg)
    return w.tower * t_x + w.own_bonus * b_x + w.opp_bonus * b_y + w.const


def payoff(x: np.ndarray, y: np.ndarray, cfg: GameConfig, objective: str = "margin") -> float:
    """Row payoff when row plays x and column plays y."""
    return utility(x, y, cfg, row_weights(objective, cfg))


def payoff_matrix(
    rows: np.ndarray, cols: np.ndarray, cfg: GameConfig, objective: str = "margin"
) -> np.ndarray:
    """Matrix A with A[i, j] = payoff(rows[i], cols[j]). Row maximizes, column minimizes."""
    w = row_weights(objective, cfg)
    A = np.empty((len(rows), len(cols)))
    for i, x in enumerate(rows):
        for j, y in enumerate(cols):
            A[i, j] = utility(x, y, cfg, w)
    return A


def enumerate_allocations(cfg: GameConfig) -> np.ndarray:
    """All allocations of cfg.n_units into cfg.n_towers towers, shape (n_pure_strategies, n_towers).

    Only feasible for small games: the full 10-tower, 20-unit game has ~10M.
    """
    out = []

    def rec(prefix: list[int], remaining: int, towers_left: int):
        if towers_left == 1:
            out.append(prefix + [remaining])
            return
        for k in range(remaining + 1):
            rec(prefix + [k], remaining - k, towers_left - 1)

    rec([], cfg.n_units, cfg.n_towers)
    return np.array(out, dtype=int)
