"""Experiment: does training the pair model on the repair (fail->pass) signal
close the gap to real MBFL on the expressible subset?

Variants (all leave-one-bug-out, mean per-(line,test) collapse):
  A  full-train   + standard Metallaxis (f / sqrt(F*(f+p)))   == current ML-PMT
  B  full-train   + failonly          (f / sqrt(F*f))
  C  fail-only-train + failonly       (predict failing rows only)

Reports aggregate + expressible-subset MRR / top-1 / top-5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_pmt.classifier import OutcomeChangeModel
from scripts.evaluate_localization import rank_metrics


def score_and_rank(
    preds: pd.DataFrame,
    baseline: dict[str, str],
    formula: str,
) -> pd.DataFrame:
    """preds has [file?, line_number, test_id, probability]. Collapse mean."""
    has_file = "file" in preds.columns
    group_cols = (["file"] if has_file else []) + ["line_number", "test_id"]
    collapsed = (
        preds.groupby(group_cols, as_index=False)["probability"].mean()
    )
    F = sum(1 for r in baseline.values() if r == "fail")
    rows = []
    line_cols = (["file", "line_number"] if has_file else ["line_number"])
    for key, grp in collapsed.groupby(line_cols):
        if has_file:
            file, line = key
        else:
            line = key
            file = None
        probs = dict(zip(grp["test_id"], grp["probability"]))
        f = sum(probs.get(t, 0.0) for t, r in baseline.items() if r == "fail")
        p = sum(probs.get(t, 0.0) for t, r in baseline.items() if r == "pass")
        denom = np.sqrt(F * (f + p)) if formula == "standard" else np.sqrt(F * f)
        score = f / denom if denom > 0 else 0.0
        row = {"line_number": int(line), "f_killed": f, "p_killed": p,
               "suspiciousness": float(score)}
        if has_file:
            row["file"] = file
        rows.append(row)
    sort_cols = (["file"] if has_file else []) + [
        "suspiciousness", "f_killed", "p_killed", "line_number"
    ]
    asc = ([True] if has_file else []) + [False, False, True, True]
    return pd.DataFrame(rows).sort_values(sort_cols, ascending=asc, ignore_index=True)


def is_expressible(variant: pd.DataFrame, fault_lines: list[int]) -> bool:
    failing = set(variant[variant.baseline_test_result == "fail"]["test_id"])
    flips = variant[
        (variant.outcome_changed == 1) & (variant.test_id.isin(failing))
    ]
    return any(
        (flips["line_number"].astype(int) == int(line)).any()
        for line in fault_lines
    )


def main() -> None:
    pairs = pd.read_csv(ROOT / "data" / "pairs.csv")
    faults = pd.read_csv(ROOT / "data" / "fault_labels.csv")

    variants = ["A_full+standard", "B_full+failonly", "C_failtrain+failonly"]
    records = []
    for variant_id, group in faults.groupby("source_version"):
        variant_id = str(variant_id)
        fault_lines = sorted((str(f), int(line)) for f, line in zip(group["file"], group["fault_line"]))
        bug_key = variant_id.removesuffix("-fixed")
        train = pairs[
            ~pairs["source_version"].str.removesuffix("-fixed").eq(bug_key)
        ]
        candidates = pairs[pairs["source_version"] == variant_id]
        baseline = dict(
            zip(candidates["test_id"], candidates["baseline_test_result"], strict=True)
        )
        expr = is_expressible(candidates, fault_lines)

        full_model = OutcomeChangeModel().fit(train)
        failing_train = train[train["baseline_test_result"] == "fail"]
        fail_model = OutcomeChangeModel().fit(failing_train)

        # A: full train, standard formula
        preds = candidates.loc[:, ["file", "line_number", "test_id"]].copy()
        preds["probability"] = full_model.predict_probabilities(candidates)
        rA = score_and_rank(preds, baseline, "standard")

        # B: full train, failonly formula
        rB = score_and_rank(preds, baseline, "failonly")

        # C: failing-only train, predict failing rows, failonly
        fcand = candidates[candidates["baseline_test_result"] == "fail"].copy()
        fpreds = fcand.loc[:, ["file", "line_number", "test_id"]].copy()
        fpreds["probability"] = fail_model.predict_probabilities(fcand)
        rC = score_and_rank(fpreds, baseline, "failonly")

        for name, ranking in (("A_full+standard", rA),
                              ("B_full+failonly", rB),
                              ("C_failtrain+failonly", rC)):
            m = rank_metrics(ranking, fault_lines)
            m["variant"] = name
            m["source_version"] = variant_id
            m["expressible"] = expr
            records.append(m)
        print(f"done {variant_id} (expressible={expr})", flush=True)

    df = pd.DataFrame(records)
    print("\n=== Aggregate (all bugs) ===")
    print(df.groupby("variant").agg(
        top1=("top1", "mean"), top5=("top5", "mean"),
        mrr=("mrr", "mean"), exam=("exam", "mean"),
    ).round(3).sort_values("mrr", ascending=False).to_string())

    print("\n=== Expressible subset only ===")
    edf = df[df["expressible"]]
    print(edf.groupby("variant").agg(
        top1=("top1", "mean"), top5=("top5", "mean"),
        mrr=("mrr", "mean"), exam=("exam", "mean"),
    ).round(3).sort_values("mrr", ascending=False).to_string())

    print("\n=== Per-bug MRR (expressible bugs) ===")
    piv = edf.pivot_table(index="source_version", columns="variant",
                          values="mrr", aggfunc="first")
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
