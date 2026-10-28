"""Fusion sweep: can we beat the fixed max(SBFL, ML-PMT) hybrid?

For each bug we compute, once (cached):
  - the SBFL spectrum over the fault-file line universe (ochiai, a, b)
  - ML-PMT mutation scores per line (max and mean mutant collapse)
  - a coverage-diffusion signal (fraction of covered lines hit by a failing test)

Then we evaluate fusion rules:
  max        : max(norm(ochiai), norm(mut))            (current hybrid)
  weighted   : alpha*norm(mut) + (1-alpha)*norm(ochiai) for a grid of alpha
  adaptive   : alpha = coverage diffusion (per bug)
  rank_avg   : mean of normalized SBFL/mutation ranks
plus references: SBFL-only and ML-PMT-only.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_pmt import OutcomeChangeModel
from ml_pmt.ranker import line_aggregates
from scripts.build_bugs import prepare_checkout
from scripts.evaluate_baselines import sbfl_ranking, variant_environment
from scripts.evaluate_localization import rank_metrics

CACHE = ROOT / "artifacts" / "fusion_cache.pkl"
SEED = 42


def build_bug(pairs, faults, variant_id):
    group = faults[faults.source_version == variant_id]
    fault_lines = sorted(
        (str(f), int(l)) for f, l in zip(group["file"], group["fault_line"])
    )
    fault_files = sorted(set(group["file"].astype(str)))
    variant = pairs[pairs.source_version == variant_id]
    baseline = dict(
        zip(variant["test_id"], variant["baseline_test_result"], strict=True)
    )
    tests = sorted(set(variant["test_id"]))
    checkout, python, packages = variant_environment(variant_id)
    # Newer thefuck bugs need the same import-compat patches the corpus build
    # applies inside its sandboxes; apply them to the raw checkout so SBFL
    # coverage can run.
    prepare_checkout(checkout, variant_id.split("-")[0])
    spectrum = sbfl_ranking(
        checkout, python, tests, fault_files, packages, baseline
    )["_spectrum"].copy()
    if spectrum.empty or "line_number" not in spectrum.columns:
        return None

    bug_key = variant_id.removesuffix("-fixed")
    train = pairs[~pairs["source_version"].str.removesuffix("-fixed").eq(bug_key)]
    model = OutcomeChangeModel(random_state=SEED).fit(train)
    probs = model.predict_probabilities(variant)
    failing_count = sum(1 for r in baseline.values() if r == "fail")
    for collapse in ("max", "mean"):
        agg = line_aggregates(variant, probs, collapse=collapse)
        mut = agg["f_killed"] / np.sqrt(
            np.maximum(failing_count, 1) * (agg["f_killed"] + agg["p_killed"])
        )
        spectrum[f"mut_{collapse}"] = (
            spectrum["line_number"].map(dict(zip(agg["line_number"], mut))).fillna(0.0)
        )
    diffusion = float((spectrum["a"] > 0).mean()) if len(spectrum) else 0.0
    return {
        "spectrum": spectrum,
        "fault_lines": fault_lines,
        "diffusion": diffusion,
    }


def _norm(s: pd.Series) -> pd.Series:
    spread = s.max() - s.min()
    return (s - s.min()) / spread if spread > 0 else s * 0.0


def fuse(spec, collapse, rule, alpha=None, diffusion=0.0):
    o = _norm(spec["ochiai"].astype(float))
    m = _norm(spec[f"mut_{collapse}"].astype(float))
    if rule == "max":
        score = np.maximum(o, m)
    elif rule == "weighted":
        score = alpha * m + (1 - alpha) * o
    elif rule == "adaptive":
        score = diffusion * m + (1 - diffusion) * o
    elif rule == "rank_avg":
        n = len(o)
        ro = o.rank(ascending=False)
        rm = m.rank(ascending=False)
        no = 1 - (ro - 1) / max(n - 1, 1)
        nm = 1 - (rm - 1) / max(n - 1, 1)
        score = (no + nm) / 2
    elif rule == "ochiai":
        score = o
    elif rule == "mut":
        score = m
    else:
        raise ValueError(rule)
    out = spec[["line_number"]].copy()
    out["suspiciousness"] = np.asarray(score, dtype=float)
    return out.sort_values(
        ["suspiciousness", "line_number"], ascending=[False, True], ignore_index=True
    )


def variants():
    v = [("max", "max", None), ("mean", "max", None),
         ("max", "ochiai", None), ("max", "mut", None),
         ("max", "rank_avg", None), ("mean", "rank_avg", None),
         ("max", "adaptive", None), ("mean", "adaptive", None)]
    for collapse in ("max", "mean"):
        for alpha in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            v.append((collapse, "weighted", alpha))
    return v


def label(collapse, rule, alpha):
    if rule == "weighted":
        return f"{collapse}-w{alpha}"
    return f"{collapse}-{rule}"


def main() -> None:
    pairs = pd.read_csv(ROOT / "data" / "pairs.csv")
    faults = pd.read_csv(ROOT / "data" / "fault_labels.csv")

    cache = {}
    if CACHE.exists():
        cache = pickle.loads(CACHE.read_bytes())
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    for variant_id in faults["source_version"].unique():
        variant_id = str(variant_id)
        if variant_id in cache:
            continue
        print(f"building {variant_id}", flush=True)
        result = build_bug(pairs, faults, variant_id)
        if result is None:
            print(f"  SKIP {variant_id}: empty SBFL spectrum", flush=True)
        cache[variant_id] = result
        CACHE.write_bytes(pickle.dumps(cache))

    # expressibility per bug
    expr = {}
    for variant_id in faults["source_version"].unique():
        variant_id = str(variant_id)
        bug = variant_id.removesuffix("-fixed")
        v = pairs[pairs.source_version == bug]
        failing = set(v[v.baseline_test_result == "fail"]["test_id"])
        flips = v[(v.outcome_changed == 1) & (v.test_id.isin(failing))]
        group = faults[faults.source_version == variant_id]
        expr[variant_id] = any(
            (
                (flips["file"].astype(str) == str(r.file))
                & (flips["line_number"].astype(int) == int(r.fault_line))
            ).any()
            for r in group.itertuples()
        )

    records = []
    skipped = [v for v, d in cache.items() if d is None]
    for variant_id, data in cache.items():
        if data is None:
            continue
        for collapse, rule, alpha in variants():
            ranking = fuse(
                data["spectrum"], collapse, rule, alpha, data["diffusion"]
            )
            m = rank_metrics(ranking, data["fault_lines"])
            m["variant"] = label(collapse, rule, alpha)
            m["source_version"] = variant_id
            m["expressible"] = expr[variant_id]
            records.append(m)
    df = pd.DataFrame(records)

    print(f"\nskipped (empty SBFL spectrum): {len(skipped)} bugs: {skipped}")
    print(f"\n=== Aggregate ({df['source_version'].nunique()} bugs with SBFL data) ===")
    agg = df.groupby("variant").agg(
        top1=("top1", "mean"), top5=("top5", "mean"),
        mrr=("mrr", "mean"), exam=("exam", "mean"),
    ).round(3).sort_values("mrr", ascending=False)
    print(agg.head(20).to_string())

    print("\n=== Expressible subset (10 bugs) ===")
    e = df[df["expressible"]]
    eagg = e.groupby("variant").agg(
        top1=("top1", "mean"), top5=("top5", "mean"),
        mrr=("mrr", "mean"), exam=("exam", "mean"),
    ).round(3).sort_values("mrr", ascending=False)
    print(eagg.head(20).to_string())


if __name__ == "__main__":
    main()
