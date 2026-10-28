"""Compare all methods (ML-PMT, MBFL-real, SBFL) split by fault-file size."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_pmt import OutcomeChangeModel, predict_and_rank
from scripts.build_bugs import WORK, BUGS
from scripts.evaluate_baselines import metallaxis_ranking, sbfl_ranking
from scripts.evaluate_localization import rank_metrics

pairs = pd.read_csv(ROOT / "data" / "pairs.csv")
faults = pd.read_csv(ROOT / "data" / "fault_labels.csv")
SPEC = {(f"{s['project']}-{s['bug']}"): s for s in BUGS}


def fault_loc(vid: str) -> int:
    proj, bug = vid.split("-", 1)
    chk = WORK / f"{proj}_{bug}" / proj
    loc = 0
    for f in sorted(set(faults[faults["source_version"] == vid]["file"])):
        p = chk / f
        if p.exists():
            loc += sum(1 for _ in p.open())
    return loc


def main() -> int:
    buggy = sorted(v for v in pairs["source_version"].unique() if not v.endswith("-fixed"))
    results = {m: [] for m in ("mlpmt", "mbfl", "sbfl")}
    sizes = {}

    for i, vid in enumerate(buggy, 1):
        spec = SPEC[vid]
        checkout = WORK / f"{spec['project']}_{spec['bug']}" / spec["project"]
        python = WORK / spec["venv"] / "venv" / "bin" / "python"
        packages = spec["packages"]

        loc = fault_loc(vid)
        sizes[vid] = loc
        fl = [(str(r.file), int(r.fault_line)) for _, r in faults[faults["source_version"] == vid].iterrows()]
        cand = pairs[pairs["source_version"] == vid]
        baseline = dict(zip(cand["test_id"], cand["baseline_test_result"]))
        tests = sorted(set(cand["test_id"]))
        fault_files = sorted(set(faults[faults["source_version"] == vid]["file"]))

        # ML-PMT (max formula, mean aggregation)
        bug_key = vid.removesuffix("-fixed")
        train = pairs[~pairs["source_version"].str.removesuffix("-fixed").eq(bug_key)]
        model = OutcomeChangeModel(random_state=42).fit(train)
        results["mlpmt"].append(rank_metrics(predict_and_rank(model, cand, baseline), fl))

        # MBFL-real
        results["mbfl"].append(rank_metrics(metallaxis_ranking(pairs, vid), fl))

        # SBFL-ochiai
        sbfl = sbfl_ranking(checkout, python, tests, fault_files, packages, baseline)
        results["sbfl"].append(rank_metrics(sbfl["ochiai"], fl))

        print(f"[{i}/{len(buggy)}] {vid} loc={loc}", flush=True)

    small = [v for v in buggy if sizes[v] < 500]
    large = [v for v in buggy if sizes[v] >= 500]

    print(f"\n{'method':8s} | {'small(<500) n=' + str(len(small)):>18s} | {'large(>=500) n=' + str(len(large)):>18s}")
    for m, label in (("mlpmt", "ML-PMT"), ("mbfl", "MBFL-real"), ("sbfl", "SBFL")):
        for subset_name, vids in (("small", small), ("large", large)):
            mrrs = [results[m][buggy.index(v)]["mrr"] for v in vids]
            top1s = [results[m][buggy.index(v)]["top1"] for v in vids]
            part = f"MRR {np.mean(mrrs):.3f} / top1 {np.mean(top1s):.2f}"
            row = f"{label:8s} | {part:>18s}" if subset_name == "small" else part
            if subset_name == "small":
                print(f"{label:8s} | {part:>18s}", end="")
            else:
                print(f" | {part}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
