# Design decisions

A running log of every modeling and implementation choice and the reason for it.
This file is the source for the application writeup, so it stays current.

Notation: T = tower points, B = adjacency bonus. "Me" is the row player.

---

## D1. Objective: margin (main) and security (robustness check)

**Problem.** The question says "maximize the points you earn." Tower points are
constant-sum: T_me + T_opp = 55 always. Bonuses are not, because both players can
earn them in the same game (e.g. I win towers 1-2 and you win 9-10). So the
literal game is general-sum. The LP and double oracle machinery assume zero-sum.

**Options considered.**

- **General-sum, literal objective (T + B for each player).** This is the most
  faithful model. But general-sum games can have many equilibria with different
  payoffs, finding one is PPAD-hard, and double oracle loses its convergence
  guarantee. Rejected: wrong tool for the deadline, and it gives no single
  number to report.
- **Margin: my points minus the opponent's.** This is zero-sum and symmetric, so
  its value is exactly 0.
- **Security: the most points I can guarantee** (max over my strategy of the min
  over the opponent's strategy of my T + B). Also zero-sum. It answers the literal
  question in its worst case, against an opponent who tries to hold my score down.
- **Win probability.** Also zero-sum, but it answers "beat the opponent," not
  "maximize points." Rejected.

**Decision.** Margin is the main model and security is a robustness check. Both
use the same code, because only the payoff weights differ (D5).

**How margin differs from the literal objective (the key argument).**

    margin = (T_me + B_me) - (T_opp + B_opp)
           = 2*T_me - 55 + B_me - B_opp          (since T_opp = 55 - T_me)
           ∝ T_me + (B_me - B_opp) / 2           (drop the constant, halve)

The literal objective is T_me + B_me. The tower term is identical: maximizing my
tower points and maximizing the tower margin are the same thing, because of
constant sum. The models differ only in how they treat bonuses:

1. My own bonus counts half as much relative to tower points under margin. Taking
   a tower worth v from the opponent moves the margin by 2v, but my bonus does not
   come out of the opponent's score.
2. Under margin, blocking the opponent's bonus is worth exactly as much as earning
   my own. The literal objective does not care about the opponent's bonus.

So the margin equilibrium should value bonuses somewhat less and blocking somewhat
more than a literal points-maximizer would. The security model (weights 1, 1, 0 on
T_me, B_me, B_opp) weights my own bonus fully and serves as the check on this.

**Security game details.** The opponent's utility is -(T_me + B_me) = T_opp - B_me - 55:
they want tower points and want to block my bonus, but get nothing from their
own bonus. The game is asymmetric, so it has no known value to check against.

## D2. The question asks for a single allocation

"How do you allocate your soldiers?" asks for one pure allocation, but the
equilibrium is a mixed strategy. Blotto has no pure equilibrium. *Open: decide
how to present this.* Useful fact: every allocation in the equilibrium support
earns the game value against the equilibrium opponent. So the choice among them
depends on a model of real opponents (e.g. humans over-weight high-value towers).
Candidates: report the mixed strategy and one representative draw, or pick the
support allocation that does best against a plausible "human" opponent.

## D3. Discretize into units of 5 soldiers

The full game has C(109, 9) ≈ 4.3 × 10^12 allocations. With units of 5 there are
20 units and C(29, 9) ≈ 10M allocations. The DP best response never lists them
anyway, but coarser units mean fewer candidate strategies for double oracle to add,
so it converges faster.

**Cost.** In the coarse game both players place multiples of 5. A real opponent
can place 6 where I place 5, and the coarse game cannot see that deviation, so
the coarse gap *understates* true exploitability. **Mitigation:** after solving,
compute a best response at 1-soldier resolution against the final strategy and
report that as the honest exploitability. If it is large, rerun at units of 2 or 1.

## D4. Restricted games are solved by LP (scipy.optimize.linprog, HiGHS)

A finite zero-sum game is exactly an LP (minimax theorem, LP duality). HiGHS is
built into scipy, so no extra dependency is needed. The column strategy comes
from a second LP, or from the dual variables.

## D5. Every objective is expressed as a weight vector

Utility = a*T_me + b*B_me + c*B_opp + const. This works because T_opp = 55 - T_me.

| model | player | (a, b, c) | const |
|---|---|---|---|
| margin | both | (2, 1, -1) | -55 |
| security | row | (1, 1, 0) | 0 |
| security | column | (1, 0, -1) | -55 |

One DP best response handles all three cases, and the tests check that each pair
is zero-sum.

## D6. Double oracle is written generically

The loop only sees a payoff function and two best-response oracles. This lets
the same code be tested on random matrix games (with an enumeration oracle)
against the full LP before it is ever run on Blotto.

## D7. Convergence gap = exploitability

Suppose (p, q) is the restricted equilibrium. Row's best response to q gives an
upper bound U on the game value. Column's best response to p gives a lower bound
L. gap = U - L ≥ 0, and it is 0 exactly when (p, q) is an equilibrium of the full
game. This is the same quantity as exploitability in the poker solver: the sum of
what each player gains by deviating. Under margin the value is 0, so U and -L are
each player's exploitability in points of margin. The full history is recorded
per iteration.

## D8. Validation ladder

Each stage is tested against something that does not rely on the code under test:

1. Payoff: hand-computed cases (ties, overlapping pairs, both players earning
   bonuses), plus properties: tower points sum to 55, margin is antisymmetric,
   security is zero-sum.
2. LP: rock-paper-scissors, a 2×2 game solved by hand, a saddle point, an
   all-negative matrix (catches a value variable wrongly bounded ≥ 0), and
   minimax conditions checked directly on random rectangular games.
3. Ground truth: 3 towers/6 soldiers (28 allocations) and 4 towers/8 soldiers
   (165), solved by full LP. The margin value must be 0.
4. DP best response: must equal brute force on 3-, 4- and 5-tower games (5
   towers/10 soldiers has 1001 allocations; 3 towers alone is too short a chain to
   exercise the DP transitions). At full size it must beat random allocations.
5. Double oracle: must match the full LP on random matrix games and the ground
   truth on the small Blotto games, with valid bounds at every iteration.

## D9. Representation

Allocations are integer arrays in units, and double oracle strategies are tuples
(hashable, so duplicates can be detected). The tower values and bonus are
parameters, so small games are just different `GameConfig`s.

---

## Results log

- **Stage 1 (payoff), 2026-09-21.** `score_components` is a single left-to-right
  pass that remembers the previous tower's outcome (x won / tie / y won). A bonus
  is paid when the current tower is won outright by the same player who won the
  previous one. Tower values and bonus come from the config. 16 tests pass.
- **Stage 2 (LP), 2026-09-21.** Two LPs: the row player maximizes a free variable
  v subject to v ≤ every column's expected payoff, and the column player
  minimizes w subject to every row's expected payoff ≤ w. The function checks
  |v − w| < 1e-6 (the minimax theorem / LP strong duality) as a built-in bug
  detector. 23 tests pass, including rectangular games and an all-negative
  matrix.
