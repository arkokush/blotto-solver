"""Command-line entry point for a full-size run.

Examples:
    python run.py --objective margin --unit 5 --fine-check

    # long run, saving a usable partial result every 10 iterations
    python run.py --objective margin --unit 1 --tol 0.02 --max-iters 6000 \
        --use-dual --warm-start results/margin_unit5.json \
        --checkpoint-every 10 --tag _overnight --fine-check
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from double_oracle import blotto_double_oracle
from exploitability import best_response_value_to_row
from payoff import OBJECTIVES, GameConfig

RESULTS_DIR = Path(__file__).parent / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--towers", type=int, default=10)
    parser.add_argument("--soldiers", type=int, default=100)
    parser.add_argument("--unit", type=int, default=5, help="soldiers per indivisible unit")
    parser.add_argument("--objective", choices=OBJECTIVES, default="margin")
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument(
        "--fine-check",
        action="store_true",
        help="also measure exploitability at 1-soldier resolution (D3)",
    )
    parser.add_argument("--prune", action="store_true",
                        help="drop zero-weight strategies to keep the LP small (needed at unit 1)")
    parser.add_argument("--use-dual", action="store_true",
                        help="read q from the LP duals instead of solving a second LP")
    parser.add_argument("--warm-start", metavar="RESULTS.json",
                        help="seed the restricted game with a coarser run's support")
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        help="write a partial result every N iterations")
    parser.add_argument("--tag", default="", help="suffix for output files, so concurrent runs never collide")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    cfg = GameConfig(n_towers=args.towers, n_soldiers=args.soldiers, unit=args.unit)
    print(f"{args.objective}: {cfg.n_towers} towers, {cfg.n_soldiers} soldiers, "
          f"units of {cfg.unit} ({cfg.n_pure_strategies:,} allocations)")

    init = None
    if args.warm_start:
        warm = json.loads(Path(args.warm_start).read_text())
        init = [tuple(v // cfg.unit for v in a) for a in warm["allocations"]]
        bad = [a for a in warm["allocations"] if any(v % cfg.unit for v in a)]
        if bad:
            raise SystemExit(f"{args.warm_start} has allocations that are not multiples of {cfg.unit}")
        print(f"warm start: {len(init)} allocations from {args.warm_start}")

    partial_path = RESULTS_DIR / f"{args.objective}_unit{cfg.unit}{args.tag}_partial.json"

    def checkpoint(snapshot):
        """Save a usable partial result, so a long run is never wasted.

        Written to a temporary file and then renamed, which is atomic: killing the
        run mid-write can never leave a half-written (unreadable) checkpoint.
        """
        if args.no_save:
            return
        RESULTS_DIR.mkdir(exist_ok=True)
        rec = snapshot.history[-1]
        payload = {
            "objective": args.objective, "unit": cfg.unit, "converged": False,
            "tol": args.tol, "prune": bool(args.prune),
            "iterations": len(snapshot.history), "value": float(snapshot.value),
            "gap": float(rec.gap), "solve_seconds": round(rec.elapsed, 1),
            "support": int((snapshot.row_probs > 1e-9).sum()),
            "allocations": [[u * cfg.unit for u in s] for s in snapshot.row_strats],
            "probs": [float(v) for v in snapshot.row_probs],
            "history": [asdict(r) for r in snapshot.history],
        }
        tmp = partial_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        tmp.replace(partial_path)
        print(f"    checkpoint: iteration {len(snapshot.history)}, gap {rec.gap:.4f} "
              f"-> {partial_path.name}", flush=True)

    t0 = time.time()
    result = blotto_double_oracle(
        cfg, args.objective, tol=args.tol, max_iters=args.max_iters, verbose=True,
        prune=args.prune, use_dual=args.use_dual, init=init, on_iteration=checkpoint,
        checkpoint_every=args.checkpoint_every,
    )
    solve_seconds = time.time() - t0

    # Keep the support only, and convert units back to soldiers for reporting.
    keep = result.row_probs > 1e-9
    allocs = np.array([[u * cfg.unit for u in s] for s in result.row_strats])[keep]
    probs = result.row_probs[keep]
    probs = probs / probs.sum()

    print(f"\nvalue {result.value:.4f}  converged {result.converged}  "
          f"iterations {len(result.history)}  support {keep.sum()}  {solve_seconds:.0f}s")
    print("most frequent allocations (soldiers per tower):")
    for p, x in sorted(zip(probs, allocs.tolist()), key=lambda t: -t[0])[:10]:
        print(f"  {p:.4f}  {x}")

    out = {
        "objective": args.objective,
        "unit": cfg.unit,
        "tol": args.tol,
        "prune": bool(args.prune),
        "value": float(result.value),
        "converged": bool(result.converged),
        "iterations": len(result.history),
        "support": int(keep.sum()),
        "solve_seconds": round(solve_seconds, 1),
        "allocations": allocs.tolist(),
        "probs": probs.tolist(),
        "history": [asdict(rec) for rec in result.history],
    }

    if args.fine_check:
        # The coarse game cannot represent an opponent who adds a single soldier,
        # so replay this strategy in the 1-soldier game and see what it concedes.
        t0 = time.time()
        y, br_value = best_response_value_to_row(allocs, probs, args.objective, unit=1)
        out["fine_br_value"] = float(br_value)
        out["fine_exploitability"] = float(result.value - br_value)
        out["fine_br_alloc"] = y.tolist()
        print(f"\n1-soldier best response holds this strategy to {br_value:.4f} "
              f"(value was {result.value:.4f}, so it gives up {result.value - br_value:.4f}) "
              f"[{time.time() - t0:.0f}s]")
        print(f"  exploiting allocation: {y.tolist()}")

    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / f"{args.objective}_unit{cfg.unit}{args.tag}.json"
        path.write_text(json.dumps(out, indent=1))
        print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
