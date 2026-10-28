"""Close the loop: hybrid on the size split + de-confound size vs coverage.

For each buggy bug compute fault-file LOC, a direct coverage-diffusion ratio
(fraction of the fault file covered by >=1 failing test), and MRR for ML-PMT
(mean), MBFL-real, SBFL-ochiai, and the hybrid. Then report splits by LOC and
by diffusion, plus a within-project gradient (tqdm/black).
"""

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
from scripts.evaluate_baselines import hybrid_ranking, metallaxis_ranking, sbfl_ranking
from scripts.evaluate_localization import rank_metrics

pairs = pd.read_csv(ROOT / "data" / "pairs.csv")
faults = pd.read_csv(ROOT / "data" / "fault_labels.csv")
SPEC = {f"{s['project']}-{s['bug']}": s for s in BUGS}


def main() -> int:
    buggy = sorted(v for v in pairs["source_version"].unique() if not v.endswith("-fixed"))
    records = []
    for i, vid in enumerate(buggy, 1):
        spec = SPEC[vid]
        checkout = WORK / f"{spec['project']}_{spec['bug']}" / spec["project"]
        python = WORK / spec["venv"] / "venv" / "bin" / "python"
        packages = spec["packages"]
        fl = [(str(r.file), int(r.fault_line)) for _, r in faults[faults["source_version"] == vid].iterrows()]
        cand = pairs[pairs["source_version"] == vid]
        baseline = dict(zip(cand["test_id"], cand["baseline_test_result"]))
        tests = sorted(set(cand["test_id"]))
        fault_files = sorted(set(faults[faults["source_version"] == vid]["file"]))

        # fault-file LOC
        loc = 0
        for f in fault_files:
            p = checkout / f
            if p.exists():
                loc += sum(1 for _ in p.open())

        # ML-PMT (mean)
        bug_key = vid.removesuffix("-fixed")
        train = pairs[~pairs["source_version"].str.removesuffix("-fixed").eq(bug_key)]
        model = OutcomeChangeModel(random_state=42).fit(train)
        m_mlpmt = rank_metrics(predict_and_rank(model, cand, baseline), fl)

        # MBFL-real
        m_mbfl = rank_metrics(metallaxis_ranking(pairs, vid), fl)

        # SBFL + hybrid
        sbfl = sbfl_ranking(checkout, python, tests, fault_files, packages, baseline)
        m_sbfl = rank_metrics(sbfl["ochiai"], fl)
        m_hyb = rank_metrics(
            hybrid_ranking(pairs, vid, checkout=checkout, python=python,
                           test_args=tests, fault_files=fault_files,
                           source_packages=packages, baseline=baseline), fl)

        # coverage diffusion: fraction of fault-file lines covered by >=1 failing test
        spectrum = sbfl["_spectrum"]
        n_fail_covered = int((spectrum["a"] > 0).sum()) if not spectrum.empty else 0
        diffusion = n_fail_covered / loc if loc > 0 else 0.0

        records.append({
            "vid": vid, "project": vid.split("-")[0], "loc": loc,
            "diffusion": diffusion,
            "mlpmt": m_mlpmt["mrr"], "mbfl": m_mbfl["mrr"],
            "sbfl": m_sbfl["mrr"], "hybrid": m_hyb["mrr"],
        })
        print(f"[{i}/{len(buggy)}] {vid} loc={loc} diff={diffusion:.2f}", flush=True)

    df = pd.DataFrame(records)
    df.to_csv("/tmp/opencode/size_split.csv", index=False)

    def report(sub, label):
        m = {k: np.mean(sub[k]) for k in ("mlpmt", "mbfl", "sbfl", "hybrid")}
        print(f"{label:34s} n={len(sub):2d} | ML-PMT {m['mlpmt']:.3f} | MBFL {m['mbfl']:.3f} | SBFL {m['sbfl']:.3f} | Hybrid {m['hybrid']:.3f}")

    print("\n=== split by LOC ===")
    report(df[df.loc < 500], "small (<500 LOC)")
    report(df[df.loc >= 500], "large (>=500 LOC)")

    print("\n=== split by coverage diffusion ===")
    med = df.diffusion.median()
    report(df[df.diffusion <= med], f"discriminative (diff <= {med:.2f})")
    report(df[df.diffusion > med], f"diffuse (diff > {med:.2f})")

    print("\n=== within-project gradient (tqdm) ===")
    tq = df[df.project == "tqdm"]
    report(tq[tq.loc < 500], "tqdm small (<500)")
    report(tq[tq.loc >= 500], "tqdm large (>=500)")

    print("\n=== within-project gradient (black) ===")
    bk = df[df.project == "black"]
    report(bk[bk.loc < 500], "black small (<500)")
    report(bk[bk.loc >= 500], "black large (>=500)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
