"""Best response to a mixed opponent.

``best_response`` is the dynamic program used inside double oracle. It never
enumerates allocations, so it scales to the full game.
``brute_force_best_response`` enumerates every allocation. It is only for
testing the DP on small games.
"""

from __future__ import annotations

import numpy as np

from payoff import GameConfig, Weights, enumerate_allocations, utility


def best_response(
    opp_allocs: np.ndarray, opp_probs: np.ndarray, cfg: GameConfig, w: Weights
) -> tuple[np.ndarray, float]:
    """Allocation maximizing expected utility against a mixed opponent.

    Args:
        opp_allocs: (k, n_towers) opponent allocations (in units) in the support.
        opp_probs: (k,) their probabilities (nonnegative, sum to 1).
        cfg: game parameters.
        w: weights of *my* utility (see payoff.Weights); use row_weights for the
           row player and col_weights for the column player.

    Returns:
        (x, value): an optimal allocation (ints, length n_towers, sum n_units)
        and its expected utility sum_k opp_probs[k] * utility(x, opp_allocs[k], cfg, w).
    """
    tower_value, pair = _build_tables(opp_allocs, opp_probs, cfg, w)
    n, U = cfg.n_towers, cfg.n_units
    f = np.full((n + 1, U + 1, U + 1), -np.inf)
    choice = np.full((n, U + 1, U + 1), -1, dtype=int)
    f[n, 0, :] = 0.0

    for i in range(n - 1, -1, -1):
        for r in range(U + 1):
            for p in range(U + 1):
                best_val, best_a = -np.inf, -1

                for a in range(r + 1):
                    val = tower_value[i,a] + f[i + 1, r - a, a]

                    if i != 0:
                        val += pair[i - 1,p,a]

                    if val > best_val:
                        best_val, best_a = val, a

                f[i, r, p] = best_val
                choice[i,r,p] = best_a

    x = np.zeros(n, dtype=int)
    r, p = U, 0
    for i in range(n):
        a = choice[i, r, p]
        x[i] = a
        r -= a
        p = a

    return x, f[0, U, 0] + w.const




def _build_tables(opp_allocs: np.ndarray, opp_probs: np.ndarray, cfg: GameConfig, w: Weights):
    """Returns (tower_value, pair):
      tower_value[i, a]     expected weighted tower term for a units on tower i
      pair[i, a, b]         expected weighted bonus terms for pair (i, i+1)
    """
    n, U = cfg.n_towers, cfg.n_units
    y, pi = np.asarray(opp_allocs), np.asarray(opp_probs)
    tower_value = np.zeros((n, U + 1))
    pair = np.zeros((n-1,U+1,U+1))

    for i in range(n):
        v = cfg.tower_values[i]
        for a in range(U+1):
            W = (pi * (y[:,i] < a)).sum()
            T = (pi * (y[:,i] == a)).sum()
            tower_value[i,a] = w.tower * v * (W + T/2)

            if i == n - 1:
                continue

            for b in range(U+1):
                WW = (pi * (y[:,i] < a) * (y[:,i + 1] < b)).sum()
                LL = (pi * (y[:,i] > a) * (y[:,i + 1] > b)).sum()
                pair[i,a,b] = cfg.bonus * (w.own_bonus * WW + w.opp_bonus * LL)

    return tower_value, pair



def brute_force_best_response(
    opp_allocs: np.ndarray, opp_probs: np.ndarray, cfg: GameConfig, w: Weights
) -> tuple[np.ndarray, float]:
    """Same contract as best_response, by enumerating every allocation."""
    allocs = enumerate_allocations(cfg)
    values = np.array(
        [sum(p * utility(x, y, cfg, w) for y, p in zip(opp_allocs, opp_probs)) for x in allocs]
    )
    i = int(np.argmax(values))
    return allocs[i], float(values[i])
