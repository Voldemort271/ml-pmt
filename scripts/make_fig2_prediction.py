"""Create Figure 2: ROC curves for all pairs and failing-test pairs."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "revision_v7"
FIG = ROOT / "docs" / "figures"


def _write_curve(path: Path, labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    if len(fpr) > 400:
        index = np.unique(np.linspace(0, len(fpr) - 1, 400).astype(int))
        fpr, tpr = fpr[index], tpr[index]
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(path, index=False, header=False)
    return float(roc_auc_score(labels, scores))


def main() -> int:
    predictions = pd.read_csv(OUT / "pair_predictions.csv")
    labels = predictions["outcome_changed"].astype(int).to_numpy()
    full = predictions["probability"].to_numpy()
    coverage = predictions["test_covers_mutation"].to_numpy()
    failing = (predictions["baseline_test_result"] == "fail").to_numpy()

    panels = {
        "all": np.ones(len(predictions), dtype=bool),
        "failing": failing,
    }
    auc_values: dict[tuple[str, str], float] = {}
    for panel, mask in panels.items():
        auc_values[(panel, "full")] = _write_curve(
            OUT / f"roc_{panel}_full.dat", labels[mask], full[mask]
        )
        auc_values[(panel, "coverage")] = _write_curve(
            OUT / f"roc_{panel}_coverage.dat", labels[mask], coverage[mask]
        )

    script = f'''set datafile separator comma
set terminal pdfcairo size 7.0,2.7 font "Helvetica,9" enhanced
set output "{FIG / 'fig2_prediction.pdf'}"
set multiplot layout 1,2 margins 0.08,0.98,0.17,0.92 spacing 0.10
set border linewidth 0.6
set grid ytics lt 0 lc rgb "#dddddd"
set xrange [0:1]
set yrange [0:1]
set xtics 0,0.25,1
set ytics 0,0.25,1
set xlabel "False positive rate"
set ylabel "True positive rate"
set key bottom right font ",7" spacing 0.9
set title "(a) All test--mutant pairs" font ",9"
plot "{OUT / 'roc_all_full.dat'}" using 1:2 with lines lw 1.8 lc rgb "#1f4e79" title "Full model (AUC {auc_values[('all','full')]:.3f})", \\
     "{OUT / 'roc_all_coverage.dat'}" using 1:2 with lines lw 1.8 lc rgb "#c55a11" title "Coverage only (AUC {auc_values[('all','coverage')]:.3f})", \\
     x with lines lw 0.9 lc rgb "#888888" dt 2 title "Random"
unset title
set title "(b) Failing-test pairs" font ",9"
plot "{OUT / 'roc_failing_full.dat'}" using 1:2 with lines lw 1.8 lc rgb "#1f4e79" title "Full model (AUC {auc_values[('failing','full')]:.3f})", \\
     "{OUT / 'roc_failing_coverage.dat'}" using 1:2 with lines lw 1.8 lc rgb "#c55a11" title "Coverage only (AUC {auc_values[('failing','coverage')]:.3f})", \\
     x with lines lw 0.9 lc rgb "#888888" dt 2 title "Random"
unset multiplot
set datafile separator comma
set terminal pngcairo size 1600,620 font "Helvetica,16" enhanced
set output "{FIG / 'fig2_prediction.png'}"
set multiplot layout 1,2 margins 0.08,0.98,0.17,0.92 spacing 0.10
set border linewidth 0.8
set grid ytics lt 0 lc rgb "#dddddd"
set xrange [0:1]
set yrange [0:1]
set xtics 0,0.25,1
set ytics 0,0.25,1
set xlabel "False positive rate"
set ylabel "True positive rate"
set key bottom right font ",11" spacing 0.9
set title "(a) All test--mutant pairs" font ",14"
plot "{OUT / 'roc_all_full.dat'}" using 1:2 with lines lw 2.4 lc rgb "#1f4e79" title "Full model (AUC {auc_values[('all','full')]:.3f})", \\
     "{OUT / 'roc_all_coverage.dat'}" using 1:2 with lines lw 2.4 lc rgb "#c55a11" title "Coverage only (AUC {auc_values[('all','coverage')]:.3f})", \\
     x with lines lw 1.2 lc rgb "#888888" dt 2 title "Random"
unset title
set title "(b) Failing-test pairs" font ",14"
plot "{OUT / 'roc_failing_full.dat'}" using 1:2 with lines lw 2.4 lc rgb "#1f4e79" title "Full model (AUC {auc_values[('failing','full')]:.3f})", \\
     "{OUT / 'roc_failing_coverage.dat'}" using 1:2 with lines lw 2.4 lc rgb "#c55a11" title "Coverage only (AUC {auc_values[('failing','coverage')]:.3f})", \\
     x with lines lw 1.2 lc rgb "#888888" dt 2 title "Random"
unset multiplot
'''
    script_path = FIG / "fig2_prediction.gp"
    script_path.write_text(script)
    subprocess.run(["gnuplot", str(script_path)], check=True)
    print((FIG / "fig2_prediction.pdf").as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
