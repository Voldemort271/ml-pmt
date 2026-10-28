"""Measure calibration's effect on both calibration quality and localization.

For each calibration mode (None, sigmoid, isotonic), do a leave-one-bug-out
prediction pass and report ECE + max-formula MRR (mean aggregation).
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
from scripts.evaluate_localization import rank_metrics

MODES = [None, "sigmoid", "isotonic"]
N_BINS = 10


def _ece(p: np.ndarray, y: np.ndarray) -> float:
    edges = np.linspace(0, 1, N_BINS + 1)
    ece = 0.0
    for i in range(N_BINS):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi)
        if i == N_BINS - 1:
            mask = (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(p)) * abs(p[mask].mean() - y[mask].mean())
    return ece


def _rank_lines(preds: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    has_file = "file" in preds.columns
    group_cols = (["file"] if has_file else []) + ["line_number", "test_id"]
    collapsed = (
        preds.groupby(group_cols, as_index=False)["probability"].mean()
        .rename(columns={"probability": "prob"})
    )
    failing = {t for t, r in baseline.items() if r == "fail"}
    passing = {t for t, r in baseline.items() if r == "pass"}
    F = len(failing)
    rows = []
    line_cols = (["file", "line_number"] if has_file else ["line_number"])
    for key, g in collapsed.groupby(line_cols):
        f = g[g["test_id"].isin(failing)]["prob"].sum()
        p = g[g["test_id"].isin(passing)]["prob"].sum()
        denom = np.sqrt(F * (f + p))
        score = f / denom if denom > 0 else 0.0
        row = {"line_number": int(key[1] if has_file else key), "suspiciousness": float(score)}
        if has_file:
            row["file"] = str(key[0])
        rows.append(row)
    out = pd.DataFrame(rows)
    sort_cols = (["file"] if has_file else []) + ["suspiciousness", "line_number"]
    asc = ([True] if has_file else []) + [False, True]
    return out.sort_values(sort_cols, ascending=asc, ignore_index=True)


def main() -> int:
    pairs = pd.read_csv(ROOT / "data" / "pairs.csv")
    faults_df = pd.read_csv(ROOT / "data" / "fault_labels.csv")
    buggy = sorted(v for v in pairs["source_version"].unique() if not v.endswith("-fixed"))

    print(f"{'calibration':12s} {'ECE':>7s} {'MRR':>7s} {'top1':>6s} {'top5':>6s}")
    for mode in MODES:
        all_p, all_y = [], []
        mrrs, top1s, top5s = [], [], []
        for i, vid in enumerate(buggy, 1):
            bug_key = vid.removesuffix("-fixed")
            train = pairs[~pairs["source_version"].str.removesuffix("-fixed").eq(bug_key)]
            cand = pairs[pairs["source_version"] == vid]
            model = OutcomeChangeModel(random_state=42, calibration=mode).fit(train)
            probs = model.predict_probabilities(cand)
            all_p.extend(probs.tolist())
            all_y.extend(cand["outcome_changed"].astype(int).tolist())

            preds = cand[["file", "line_number", "test_id"]].copy()
            preds["probability"] = probs
            baseline = dict(zip(cand["test_id"], cand["baseline_test_result"]))
            faults = [(str(r.file), int(r.fault_line)) for _, r in faults_df[faults_df.source_version == vid].iterrows()]
            m = rank_metrics(_rank_lines(preds, baseline), faults)
            mrrs.append(m["mrr"]); top1s.append(m["top1"]); top5s.append(m["top5"])
            print(f"  [{mode}] {i}/{len(buggy)} {vid}", flush=True)
        p = np.asarray(all_p); y = np.asarray(all_y)
        print(f"{str(mode):12s} {_ece(p, y):7.4f} {np.mean(mrrs):7.3f} {np.mean(top1s):6.2f} {np.mean(top5s):6.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
