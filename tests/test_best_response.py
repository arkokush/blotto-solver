"""Stage 4: DP best response, checked against brute force on small games."""

import numpy as np
import pytest

from best_response import best_response, brute_force_best_response
from conftest import FIVE, FOUR, FULL, TINY, random_allocation, random_mixture
from payoff import col_weights, row_weights, utility

WEIGHTS = {
    "margin": lambda cfg: row_weights("margin", cfg),
    "security_row": lambda cfg: row_weights("security", cfg),
    "security_col": lambda cfg: col_weights("security", cfg),
}


def expected_utility(x, allocs, probs, cfg, w):
    return sum(p * utility(x, y, cfg, w) for y, p in zip(allocs, probs))


def assert_valid(x, cfg):
    x = np.asarray(x)
    assert x.shape == (cfg.n_towers,)
    assert (x >= 0).all() and x.sum() == cfg.n_units
    assert np.issubdtype(x.dtype, np.integer)


@pytest.mark.parametrize("cfg", [TINY, FOUR, FIVE], ids=["tiny", "four", "five"])
@pytest.mark.parametrize("weights", list(WEIGHTS), ids=list(WEIGHTS))
@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("seed", range(3))
def test_matches_brute_force(cfg, weights, k, seed):
    rng = np.random.default_rng(seed)
    w = WEIGHTS[weights](cfg)
    allocs, probs = random_mixture(cfg, k, rng)

    x, value = best_response(allocs, probs, cfg, w)
    _, brute_value = brute_force_best_response(allocs, probs, cfg, w)

    assert_valid(x, cfg)
    assert value == pytest.approx(brute_value)
    # The reported value must be what x actually earns.
    assert expected_utility(x, allocs, probs, cfg, w) == pytest.approx(value)


@pytest.mark.parametrize("weights", list(WEIGHTS), ids=list(WEIGHTS))
def test_full_game_beats_random_allocations(weights):
    # No brute force at full size; instead the DP must beat every allocation we try.
    rng = np.random.default_rng(0)
    w = WEIGHTS[weights](FULL)
    allocs, probs = random_mixture(FULL, 10, rng)

    x, value = best_response(allocs, probs, FULL, w)

    assert_valid(x, FULL)
    assert expected_utility(x, allocs, probs, FULL, w) == pytest.approx(value)
    for _ in range(200):
        z = random_allocation(FULL, rng)
        assert expected_utility(z, allocs, probs, FULL, w) <= value + 1e-9
