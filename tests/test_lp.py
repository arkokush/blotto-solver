"""Stage 2: LP solver for zero-sum matrix games."""

import numpy as np
import pytest

from lp_solver import solve_zero_sum

TOL = 1e-7


def assert_equilibrium(A, value, p, q):
    """Check the minimax conditions directly, without trusting the solver."""
    assert p.shape == (A.shape[0],) and q.shape == (A.shape[1],)
    assert (p >= -TOL).all() and p.sum() == pytest.approx(1)
    assert (q >= -TOL).all() and q.sum() == pytest.approx(1)
    # p guarantees at least `value` against every column,
    # q concedes at most `value` against every row.
    assert (p @ A >= value - TOL).all()
    assert (A @ q <= value + TOL).all()


def test_rock_paper_scissors():
    A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    value, p, q = solve_zero_sum(A)
    assert value == pytest.approx(0, abs=TOL)
    assert p == pytest.approx(np.full(3, 1 / 3))
    assert q == pytest.approx(np.full(3, 1 / 3))
    assert_equilibrium(A, value, p, q)


def test_two_by_two_mixed():
    # Solved by hand: p = (3/7, 4/7), q = (2/7, 5/7), value = 1/7.
    A = np.array([[3, -1], [-2, 1]], dtype=float)
    value, p, q = solve_zero_sum(A)
    assert value == pytest.approx(1 / 7)
    assert p == pytest.approx([3 / 7, 4 / 7])
    assert q == pytest.approx([2 / 7, 5 / 7])


def test_saddle_point():
    A = np.array([[1, 2], [0, 3]], dtype=float)
    value, p, q = solve_zero_sum(A)
    assert value == pytest.approx(1)
    assert p == pytest.approx([1, 0], abs=TOL)
    assert q == pytest.approx([1, 0], abs=TOL)


def test_all_negative_matrix():
    # Catches the classic bug: linprog bounds every variable to >= 0 by default,
    # so a negative game value comes out wrong unless the value variable is left free.
    A = np.array([[-5, -3], [-4, -6]], dtype=float)
    value, p, q = solve_zero_sum(A)
    assert value < 0
    assert_equilibrium(A, value, p, q)


def test_value_shifts_with_constant():
    A = np.random.default_rng(1).normal(size=(6, 8))
    v0, _, _ = solve_zero_sum(A)
    v1, _, _ = solve_zero_sum(A + 10)
    assert v1 == pytest.approx(v0 + 10)


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1), (7, 7), (20, 35), (60, 15)])
@pytest.mark.parametrize("seed", range(3))
def test_random_games(shape, seed):
    A = np.random.default_rng(seed).normal(size=shape)
    assert_equilibrium(A, *solve_zero_sum(A))
