"""Command-line entry point for the full-size run.

Example:
    python run.py --objective margin --unit 5
"""

from __future__ import annotations

import argparse

from double_oracle import blotto_double_oracle
from payoff import OBJECTIVES, GameConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--towers", type=int, default=10)
    parser.add_argument("--soldiers", type=int, default=100)
    parser.add_argument("--unit", type=int, default=5)
    parser.add_argument("--objective", choices=OBJECTIVES, default="margin")
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--max-iters", type=int, default=1000)
    args = parser.parse_args()

    cfg = GameConfig(n_towers=args.towers, n_soldiers=args.soldiers, unit=args.unit)
    result = blotto_double_oracle(
        cfg, args.objective, tol=args.tol, max_iters=args.max_iters, verbose=True
    )

    print(f"\nvalue {result.value:.4f}  converged {result.converged}")
    print("row strategy (soldiers per tower):")
    for s, p in sorted(zip(result.row_strats, result.row_probs), key=lambda t: -t[1]):
        if p > 1e-6:
            print(f"  {p:.4f}  {[u * cfg.unit for u in s]}")
    # STAGE 6: save results, and evaluate exploitability at 1-soldier resolution.


if __name__ == "__main__":
    main()
