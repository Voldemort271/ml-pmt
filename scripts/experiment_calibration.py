"""Calibration diagnostic for the pair model.

Does a leave-one-bug-out prediction pass, then measures how well the model's
output probabilities match the empirical flip rate: expected calibration error
(ECE), a binned reliability table, Brier score vs. base rate, and the
over/under-confidence signature.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_pmt import OutcomeChangeModel

N_BINS = 10


def main() -> int:
    pairs = pd.read_csv(ROOT / "data" / "pairs.csv")
    buggy = sorted(
        v for v in pairs["source_version"].unique() if not v.endswith("-fixed")
    )

    probs_all: list[float] = []
    labels_all: list[int] = []
    for i, vid in enumerate(buggy, 1):
        bug_key = vid.removesuffix("-fixed")
        train = pairs[~pairs["source_version"].str.removesuffix("-fixed").eq(bug_key)]
        cand = pairs[pairs["source_version"] == vid]
        model = OutcomeChangeModel(random_state=42).fit(train)
        probs = model.predict_probabilities(cand)
        probs_all.extend(probs.tolist())
        labels_all.extend(cand["outcome_changed"].astype(int).tolist())
        print(f"[{i}/{len(buggy)}] {vid} predicted", flush=True)

    p = np.asarray(probs_all)
    y = np.asarray(labels_all)
    base_rate = y.mean()

    # Brier score vs constant-predictor (base rate) Brier
    brier = float(np.mean((p - y) ** 2))
    brier_base = float(base_rate * (1 - base_rate))

    # ECE + reliability table
    edges = np.linspace(0, 1, N_BINS + 1)
    print(f"\nn = {len(p)}, base rate = {base_rate:.4f}")
    print(f"Brier = {brier:.4f}  |  constant-predictor Brier = {brier_base:.4f}")
    print(f"Brier gap (calibration+resolution) = {brier - brier_base:+.4f} "
          f"(negative = better than predicting the mean)")
    print()
    print(f"{'bin':>10s} {'n':>7s} {'mean_pred':>10s} {'frac_pos':>10s} {'gap':>8s}")
    ece = 0.0
    for i in range(N_BINS):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi)
        if i == N_BINS - 1:
            mask = (p >= lo) & (p <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        mean_pred = float(p[mask].mean())
        frac_pos = float(y[mask].mean())
        gap = mean_pred - frac_pos
        ece += (n / len(p)) * abs(gap)
        print(f"[{lo:.1f},{hi:.1f}) {n:7d} {mean_pred:10.4f} {frac_pos:10.4f} {gap:+8.4f}")
    print(f"\nECE = {ece:.4f}  (lower is better)")
    over = p.mean() - base_rate
    print(f"mean predicted = {p.mean():.4f} vs base rate {base_rate:.4f} "
          f"({'overconfident' if over > 0 else 'underconfident'} by {over:+.4f})")
    print(f"fraction p > 0.5 = {(p > 0.5).mean():.4f} (vs true rate {base_rate:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
