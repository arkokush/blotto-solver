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
    # STAGE 4: implement the DP.
    raise NotImplementedError


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
