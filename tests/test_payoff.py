"""Stage 1: payoff function. Hand-computed cases plus structural properties."""

import numpy as np
import pytest

from conftest import FIVE, FOUR, FULL, TINY, random_allocation
from payoff import (
    GameConfig,
    col_weights,
    enumerate_allocations,
    payoff,
    score_components,
    utility,
)


def sc(x, y, cfg):
    return score_components(np.array(x), np.array(y), cfg)


# ---- hand-computed cases -------------------------------------------------------


def test_identical_allocations_split_everything():
    x = [2] * 10
    assert sc(x, x, FULL) == pytest.approx((27.5, 0, 27.5, 0))
    assert payoff(np.array(x), np.array(x), FULL, "margin") == pytest.approx(0)


def test_full_game_nine_consecutive_wins():
    # 10 soldiers on each tower vs everything on tower 10.
    x, y = [2] * 10, [0] * 9 + [20]
    # x wins towers 1-9 (45 pts) and 8 overlapping adjacent pairs (40 bonus).
    assert sc(x, y, FULL) == pytest.approx((45, 40, 10, 0))
    assert payoff(np.array(x), np.array(y), FULL, "margin") == pytest.approx(75)
    assert payoff(np.array(x), np.array(y), FULL, "security") == pytest.approx(85)


def test_tie_splits_points_and_blocks_bonus():
    # tower 1: x wins, tower 2: tie, tower 3: y wins
    assert sc([3, 3, 0], [2, 3, 1], TINY) == pytest.approx((2, 0, 4, 0))


def test_one_adjacent_pair():
    assert sc([2, 2, 2], [1, 1, 4], TINY) == pytest.approx((3, 5, 3, 0))


def test_three_consecutive_wins_give_two_bonuses():
    assert sc([2, 2, 2, 2], [1, 1, 1, 5], FOUR) == pytest.approx((6, 10, 4, 0))


def test_both_players_earn_bonuses():
    # Bonuses are not constant-sum: this is why the literal game is not zero-sum.
    assert sc([3, 3, 1, 1], [0, 0, 4, 4], FOUR) == pytest.approx((3, 5, 7, 5))


def test_tie_in_middle_breaks_chain():
    assert sc([3, 2, 0, 3], [1, 2, 4, 1], FOUR) == pytest.approx((6, 0, 4, 0))


# ---- structural properties on random allocations ------------------------------


@pytest.mark.parametrize("cfg", [TINY, FOUR, FIVE, FULL], ids=["tiny", "four", "five", "full"])
def test_properties(cfg, rng):
    for _ in range(200):
        x, y = random_allocation(cfg, rng), random_allocation(cfg, rng)
        t_x, b_x, t_y, b_y = score_components(x, y, cfg)
        # tower points are constant-sum
        assert t_x + t_y == pytest.approx(cfg.total_tower_points)
        # bonuses are whole multiples of cfg.bonus, at most one per adjacent pair
        for b in (b_x, b_y):
            assert b % cfg.bonus == 0 and 0 <= b <= cfg.bonus * (cfg.n_towers - 1)
        # swapping players swaps the components
        assert score_components(y, x, cfg) == pytest.approx((t_y, b_y, t_x, b_x))
        # margin is antisymmetric
        assert payoff(x, y, cfg, "margin") == pytest.approx(-payoff(y, x, cfg, "margin"))
        # security game is zero-sum: column utility is minus the row payoff
        assert utility(y, x, cfg, col_weights("security", cfg)) == pytest.approx(
            -payoff(x, y, cfg, "security")
        )


# ---- enumeration --------------------------------------------------------------


@pytest.mark.parametrize("cfg", [TINY, FOUR, FIVE], ids=["tiny", "four", "five"])
def test_enumerate_allocations(cfg):
    allocs = enumerate_allocations(cfg)
    assert len(allocs) == cfg.n_pure_strategies
    assert (allocs.sum(axis=1) == cfg.n_units).all()
    assert (allocs >= 0).all()
    assert len({tuple(a) for a in allocs}) == len(allocs)


def test_strategy_counts():
    assert TINY.n_pure_strategies == 28
    assert FIVE.n_pure_strategies == 1001
    assert FULL.n_pure_strategies == 10_015_005


def test_config_validation():
    with pytest.raises(ValueError):
        GameConfig(n_soldiers=100, unit=3)
