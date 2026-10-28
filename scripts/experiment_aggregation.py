"""Compare line-scoring aggregations on the predicted pair probabilities.

The max formula currently collapses mutants per (file, line, test) with
``max()`` (localization.py:81), then sums over failing tests (Metallaxis).
Denser sampling inflates ``max`` on every line, so we test alternatives:
max (current), mean, median, soft-OR (1 - prod(1-p)), and top-2 mean.

Reuses one leave-one-bug-out prediction pass; only the aggregation changes.
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
from scripts.build_bugs import fault_lines as _fault_lines
from scripts.evaluate_localization import rank_metrics


def _collapse(group: pd.Series, agg: str) -> float:
    p = group.to_numpy(dtype=float)
    if agg == "max":
        return float(p.max())
    if agg == "mean":
        return float(p.mean())
    if agg == "median":
        return float(np.median(p))
    if agg == "softor":
        return float(1.0 - np.prod(1.0 - p))
    if agg == "top2":
        return float(np.mean(np.sort(p)[-2:]) if len(p) >= 2 else p.mean())
    if agg == "top3":
        return float(np.mean(np.sort(p)[-3:]) if len(p) >= 3 else p.mean())
    raise ValueError(agg)


def _rank_lines(preds: pd.DataFrame, baseline: dict, agg: str) -> pd.DataFrame:
    has_file = "file" in preds.columns
    group_cols = (["file"] if has_file else []) + ["line_number", "test_id"]
    collapsed = (
        preds.groupby(group_cols, as_index=False)["probability"]
        .agg(lambda s: _collapse(s, agg))
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

    variants = sorted(set(pairs["source_version"]))
    buggy = [v for v in variants if not v.endswith("-fixed")]

    aggs = ["max", "mean", "median", "softor", "top2", "top3"]
    totals = {a: [] for a in aggs}

    for i, vid in enumerate(buggy, 1):
        bug_key = vid.removesuffix("-fixed")
        train = pairs[~pairs["source_version"].str.removesuffix("-fixed").eq(bug_key)]
        candidates = pairs[pairs["source_version"] == vid]
        model = OutcomeChangeModel(random_state=42).fit(train)
        probs = model.predict_probabilities(candidates)
        preds = candidates[["file", "line_number", "test_id"]].copy()
        preds["probability"] = probs
        baseline = dict(zip(candidates["test_id"], candidates["baseline_test_result"]))

        faults = [
            (str(r.file), int(r.fault_line))
            for _, r in faults_df[faults_df.source_version == vid].iterrows()
        ]
        for agg in aggs:
            ranking = _rank_lines(preds, baseline, agg)
            m = rank_metrics(ranking, faults)
            totals[agg].append(m)
        print(f"[{i}/{len(buggy)}] {vid} done", flush=True)

    print(f"\n{'aggregation':12s} {'MRR':>7s} {'top1':>6s} {'top5':>6s} {'top10':>6s} {'rank':>7s}")
    for agg in aggs:
        m = totals[agg]
        mrr = np.mean([x["mrr"] for x in m])
        t1 = np.mean([x["top1"] for x in m])
        t5 = np.mean([x["top5"] for x in m])
        t10 = np.mean([x["top10"] for x in m])
        rank = np.mean([x["rank"] for x in m])
        print(f"{agg:12s} {mrr:7.3f} {t1:6.2f} {t5:6.2f} {t10:6.2f} {rank:7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
