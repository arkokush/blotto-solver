# blotto-solver

Approximate equilibrium for a Colonel Blotto variant, computed with double oracle.

## The game

Two players each split 100 soldiers across 10 towers worth 1, 2, ..., 10 points.
At each tower the side with more soldiers wins its points, and a tie splits them.
Every pair of *adjacent* towers won outright (ties don't count) is worth 5 bonus
points. Pairs overlap, so three consecutive outright wins give 10 bonus points.

## Approach

- **Objective.** Tower points are constant-sum, but bonuses are not, so "maximize
  your points" is not literally a zero-sum game. The main model uses the point
  *margin* (my points minus the opponent's), which is zero-sum and symmetric. As a
  robustness check, a *security* model maximizes the points I can guarantee
  against any opponent. See [DECISIONS.md](DECISIONS.md).
- **Discretization.** Soldiers are placed in units of 5 (20 units, about 10M pure
  strategies). The final strategy is then checked against 1-soldier deviations.
- **Double oracle.** Solve the game restricted to a small set of strategies by
  LP, add each player's best response to the restricted equilibrium, and repeat
  until neither best response improves on the restricted value. The gap between
  the two best-response values is the exploitability of the current strategy.
- **Best response.** A dynamic program over towers, so the 10M allocations are
  never enumerated.
- **Validation.** Small games (3-5 towers) are solved exactly by enumerating every
  allocation. Double oracle and the DP must match them.

## Layout

| File | Purpose |
|---|---|
| `payoff.py` | game config, scoring, objectives, payoff matrices |
| `lp_solver.py` | zero-sum matrix game by LP (`scipy.optimize.linprog`) |
| `ground_truth.py` | exact solution of small games by full enumeration |
| `best_response.py` | DP best response to a mixed opponent (+ brute-force reference) |
| `double_oracle.py` | double oracle loop with convergence tracking |
| `run.py` | full-size run |
| `DECISIONS.md` | log of design decisions and their reasons |

## Usage

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest                          # all tests
.venv/bin/python run.py --objective margin
```
