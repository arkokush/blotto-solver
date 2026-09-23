"""Stage 5: double oracle loop, on random matrix games and on small Blotto games."""

import numpy as np
import pytest

from conftest import FOUR, TINY
from double_oracle import blotto_double_oracle, double_oracle
from ground_truth import solve_full_game
from lp_solver import solve_zero_sum

TOL = 1e-6


def run_on_matrix(A, **kwargs):
    """Double oracle on an explicit matrix; strategies are row/column indices."""

    def row_oracle(cols, q):
        values = A[:, cols] @ q
        i = int(np.argmax(values))
        return i, float(values[i])

    def col_oracle(rows, p):
        values = p @ A[rows, :]
        j = int(np.argmin(values))
        return j, float(values[j])

    return double_oracle(lambda i, j: A[i, j], row_oracle, col_oracle, 0, 0, tol=TOL, **kwargs)


def assert_history_consistent(result):
    assert result.history, "history must record every iteration"
    for rec in result.history:
        assert rec.col_br_value <= rec.restricted_value + TOL
        assert rec.restricted_value <= rec.row_br_value + TOL
        assert rec.gap == pytest.approx(rec.row_br_value - rec.col_br_value)
        assert rec.gap >= -TOL
    assert result.history[-1].gap <= TOL


def test_rock_paper_scissors():
    A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    result = run_on_matrix(A)
    assert result.converged
    assert result.value == pytest.approx(0, abs=TOL)
    assert_history_consistent(result)


@pytest.mark.parametrize("shape", [(10, 10), (30, 50), (80, 40)])
@pytest.mark.parametrize("seed", range(5))
def test_random_matrix_matches_full_lp(shape, seed):
    A = np.random.default_rng(seed).normal(size=shape)
    full_value, _, _ = solve_zero_sum(A)

    result = run_on_matrix(A)

    assert result.converged
    assert result.value == pytest.approx(full_value, abs=TOL)
    assert result.row_probs.sum() == pytest.approx(1)
    assert result.col_probs.sum() == pytest.approx(1)
    assert_history_consistent(result)


def test_restricted_sets_have_no_duplicates():
    A = np.random.default_rng(0).normal(size=(40, 40))
    result = run_on_matrix(A)
    assert len(set(result.row_strats)) == len(result.row_strats)
    assert len(set(result.col_strats)) == len(result.col_strats)


@pytest.mark.parametrize("cfg", [TINY, FOUR], ids=["tiny", "four"])
@pytest.mark.parametrize("objective", ["margin", "security"])
def test_blotto_matches_ground_truth(cfg, objective):
    truth = solve_full_game(cfg, objective)
    result = blotto_double_oracle(cfg, objective, tol=TOL)
    assert result.converged
    assert result.value == pytest.approx(truth.value, abs=TOL)
    assert_history_consistent(result)


@pytest.mark.parametrize("shape", [(30, 50), (80, 40)])
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("prune,use_dual", [(True, False), (False, True), (True, True)])
def test_speedups_do_not_change_the_answer(shape, seed, prune, use_dual):
    # Pruning zero-weight strategies and reading q from the LP duals are performance
    # options; they must reach the same value as the plain loop and the full LP.
    A = np.random.default_rng(seed).normal(size=shape)
    full_value, _, _ = solve_zero_sum(A)

    result = run_on_matrix(A, prune=prune, use_dual=use_dual)

    assert result.converged
    assert result.value == pytest.approx(full_value, abs=TOL)
    assert_history_consistent(result)


@pytest.mark.parametrize("cfg", [TINY, FOUR], ids=["tiny", "four"])
def test_blotto_speedups_match_ground_truth(cfg):
    truth = solve_full_game(cfg, "margin")
    result = blotto_double_oracle(cfg, "margin", tol=TOL, prune=True, use_dual=True)
    assert result.converged
    assert result.value == pytest.approx(truth.value, abs=TOL)
