"""Stage 3: exact solutions of small Blotto games by full enumeration + LP."""

import pytest

from conftest import FOUR, TINY
from ground_truth import solve_full_game
from payoff import payoff_matrix

TOL = 1e-7
CONFIGS = pytest.mark.parametrize("cfg", [TINY, FOUR], ids=["tiny", "four"])


@CONFIGS
def test_margin_game_value_is_zero(cfg):
    # Symmetric zero-sum game, so neither side can have an edge.
    sol = solve_full_game(cfg, "margin")
    assert sol.value == pytest.approx(0, abs=TOL)


@CONFIGS
@pytest.mark.parametrize("objective", ["margin", "security"])
def test_solution_is_an_equilibrium(cfg, objective):
    sol = solve_full_game(cfg, objective)
    A = payoff_matrix(sol.allocations, sol.allocations, cfg, objective)
    assert len(sol.row_probs) == len(sol.col_probs) == cfg.n_pure_strategies
    assert sol.row_probs.sum() == pytest.approx(1) and sol.col_probs.sum() == pytest.approx(1)
    assert (sol.row_probs @ A >= sol.value - TOL).all()
    assert (A @ sol.col_probs <= sol.value + TOL).all()


@CONFIGS
def test_security_value_is_a_guarantee_on_own_points(cfg):
    # Guaranteed points must lie between 0 and total tower points + max bonus.
    sol = solve_full_game(cfg, "security")
    assert 0 <= sol.value <= cfg.total_tower_points + cfg.bonus * (cfg.n_towers - 1)
