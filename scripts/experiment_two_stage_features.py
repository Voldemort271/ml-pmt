"""Quick B4 diagnostic: do richer line features help the two-stage ranker?

Single leave-one-bug-out pair predictions (cheap proxy for the nested LOO), then
an LOO RandomForest ranker trained on either the OLD 13 line features or the NEW
18 (which add probability-distribution stats + operator diversity). Compares MRR.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_pmt import OutcomeChangeModel
from ml_pmt.ranker import line_aggregates

OLD_FEATURES = [
    "f_killed", "p_killed", "n_mutants", "n_failing_cover", "n_passing_cover",
    "max_failing_prob", "mean_failing_prob", "frac_failing_high",
    "n_failing", "n_passing", "ochiai", "tarantula", "dstar",
]
NEW_FEATURES = OLD_FEATURES + [
    "var_failing_prob", "median_failing_prob", "p90_failing_prob",
    "mean_passing_prob", "n_operator_types",
]


def main() -> int:
    pairs = pd.read_csv(ROOT / "data" / "pairs.csv")
    faults = pd.read_csv(ROOT / "data" / "fault_labels.csv")
    buggy = sorted(v for v in pairs["source_version"].unique() if not v.endswith("-fixed"))

    # Single LOO: pair model trained on all-other-bugs, predict this bug.
    all_features = []
    for i, vid in enumerate(buggy, 1):
        bug_key = vid.removesuffix("-fixed")
        train = pairs[~pairs["source_version"].str.removesuffix("-fixed").eq(bug_key)]
        cand = pairs[pairs["source_version"] == vid]
        model = OutcomeChangeModel(random_state=42).fit(train)
        probs = model.predict_probabilities(cand)
        feats = line_aggregates(cand, probs)
        fault_lines = set(faults[faults["source_version"] == vid]["fault_line"].astype(int))
        feats["is_fault"] = feats["line_number"].isin(fault_lines).astype(int)
        feats["source_version"] = vid
        all_features.append(feats)
        print(f"[{i}/{len(buggy)}] {vid} predicted", flush=True)

    data = pd.concat(all_features, ignore_index=True)

    for name, cols in [("old (13)", OLD_FEATURES), ("new (18)", NEW_FEATURES)]:
        mrrs, top1s = [], []
        for vid in buggy:
            held = data[data["source_version"] == vid]
            train_rows = data[data["source_version"] != vid]
            rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
            rf.fit(train_rows[cols], train_rows["is_fault"])
            proba = rf.predict_proba(held[cols])
            held = held.copy()
            held["suspiciousness"] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
            ranked = held.sort_values(["suspiciousness", "line_number"], ascending=[False, True])
            faults_vid = faults[faults["source_version"] == vid]
            flines = [int(x) for x in faults_vid["fault_line"]]
            lines = ranked["line_number"].tolist()
            ranks = [lines.index(l) + 1 for l in flines if l in lines]
            rank = min(ranks) if ranks else len(ranked) + 1
            mrrs.append(1.0 / rank)
            top1s.append(rank == 1)
        print(f"{name:12s} MRR={np.mean(mrrs):.3f}  top1={np.mean(top1s):.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
