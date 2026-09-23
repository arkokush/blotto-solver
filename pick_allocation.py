"""Choose one allocation to submit from the equilibrium's support (D2).

The question asks for a single allocation, but the equilibrium is a mixture.
Every allocation in the support scores the game value against an equilibrium
opponent, so choosing among them costs nothing against a perfect opponent. This
scores each one against a handful of plausible *human* allocations and reports
both numbers: how it does against that field, and what it scores against the
equilibrium itself.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from payoff import GameConfig, payoff

# Allocations a person might plausibly submit, in soldiers.
HUMAN_OPPONENTS = {
    "even": [10] * 10,
    "top_heavy": [0, 0, 0, 0, 5, 10, 15, 20, 25, 25],
    "proportional": [2, 4, 5, 7, 9, 11, 13, 15, 16, 18],
    "abandon_small": [0, 0, 0, 10, 12, 14, 16, 16, 16, 16],
    "bonus_cluster": [0, 0, 0, 0, 0, 20, 20, 20, 20, 20],
    "top_three_only": [0, 0, 0, 0, 0, 0, 0, 30, 35, 35],
}


def score_support(allocations, probs, objective="margin", humans=None) -> list[dict]:
    """Score every support allocation against the human field and the equilibrium.

    Returns one dict per allocation, sorted by average score against the field:
        alloc, prob, avg, worst, vs_equilibrium, per_opponent
    """
    cfg = GameConfig(unit=1)
    humans = {k: np.asarray(v) for k, v in (humans or HUMAN_OPPONENTS).items()}
    for name, a in humans.items():
        if a.sum() != cfg.n_soldiers:
            raise ValueError(f"{name} does not use {cfg.n_soldiers} soldiers")

    allocations, probs = np.asarray(allocations), np.asarray(probs)
    rows = []
    for x, p in zip(allocations, probs):
        per_opponent = {name: payoff(x, y, cfg, objective) for name, y in humans.items()}
        rows.append({
            "alloc": [int(v) for v in x],
            "prob": float(p),
            "avg": float(np.mean(list(per_opponent.values()))),
            "worst": float(min(per_opponent.values())),
            "vs_equilibrium": float(
                sum(pp * payoff(x, y, cfg, objective) for y, pp in zip(allocations, probs))
            ),
            "per_opponent": per_opponent,
        })
    rows.sort(key=lambda r: -r["avg"])
    return rows


def main(path: str, objective: str = "margin", top: int = 10):
    data = json.load(open(path))
    rows = score_support(data["allocations"], data["probs"], objective)
    names = list(HUMAN_OPPONENTS)

    print(f"{'prob':>7} {'avg':>7} {'worst':>7} {'vs eq':>7}  "
          + " ".join(f"{n[:9]:>9}" for n in names))
    for r in rows[:top]:
        print(f"{r['prob']:7.4f} {r['avg']:7.2f} {r['worst']:7.2f} {r['vs_equilibrium']:7.3f}  "
              + " ".join(f"{r['per_opponent'][n]:9.1f}" for n in names)
              + f"   {r['alloc']}")

    print("\nHuman opponents used:")
    for name, a in HUMAN_OPPONENTS.items():
        print(f"  {name:16s} {a}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "margin")
