"""Solve a finite two-player zero-sum matrix game by linear programming."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

def solve_zero_sum(A: np.ndarray, use_dual: bool = False) -> tuple[float, np.ndarray, np.ndarray]:
    """Minimax solution of the zero-sum game with payoff matrix A.

    The row player picks a row and receives A[i, j]; the column player picks a
    column and pays it. The row player maximizes, the column player minimizes.

    Args:
        A: payoff matrix of shape (m, n); entries may be any sign.

    Returns:
        (value, p, q): the game value, the row player's optimal mixed strategy p
        (length m), and the column player's optimal mixed strategy q (length n).
        p and q are nonnegative and each sums to 1, and they satisfy
            min_j (p @ A)[j] = value = max_i (A @ q)[i]
        up to solver tolerance.
    """

    #P1
    m,n = A.shape

    c = np.append(np.zeros(m), -1)
    A_ub = np.hstack([-A.T, np.ones((n,1))])
    b_ub = np.zeros(n)
    A_eq = np.append(np.ones(m),0).reshape(1,-1)
    b_eq = np.array([1.0])
    bounds = [(0, None)] * m + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"row LP failed: {res.message}")

    p, value = res.x[:m], res.x[m]

    if use_dual:
        # The column player's LP is the dual of the row player's, so q is the vector
        # of shadow prices on the column constraints. This halves the LP work, which
        # matters once the restricted game has hundreds of strategies.
        #
        # The duals can come back slightly off when the LP is degenerate (common once
        # the restricted game has many near-equivalent strategies), so the result is
        # checked. A failed check falls through to solving the column LP exactly -
        # slower, never wrong. It must not raise: this runs inside long solves.
        q = -res.ineqlin.marginals
        q = np.clip(q, 0, None)
        if q.sum() > 0:
            q = q / q.sum()
            # Scale the tolerance to the size of the payoffs, not an absolute 1e-6.
            slack = 1e-6 * max(1.0, abs(value), float(np.abs(A).max()))
            if (A @ q).max() <= value + slack:
                return value, p, q

    #P2
    c = np.append(np.zeros(n), 1)
    A_ub = np.hstack([A, -np.ones((m,1))])
    b_ub = np.zeros(m)
    A_eq = np.append(np.ones(n),0).reshape(1,-1)
    b_eq = np.array([1.0])
    bounds = [(0, None)] * n + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if res.status != 0:
        raise RuntimeError(f"row LP failed: {res.message}")

    q, w = res.x[:n], res.x[n]

    if abs(value - w) > 1e-6:
        raise RuntimeError(f"minimax mismatch: v={value}, w={w}")

    return value, p, q

