"""Reproducible analyses for the AST 2027 short-paper revision.

Computes the expressibility funnel, leave-one-bug-out pair metrics on
failing-test pairs, tie-aware localization metrics, aggregation ablations,
an SBFL-only learned-ranker baseline, and execution-cost counts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_pmt import OutcomeChangeModel
from ml_pmt.localization import rank_suspicious_lines
from mutation_engine.generator import candidate_lines

OUT = ROOT / "artifacts" / "revision_v7"
PAIRS = ROOT / "data" / "pairs.csv"
FAULTS = ROOT / "data" / "fault_labels.csv"
CACHE = ROOT / "artifacts" / "fusion_cache.pkl"
WORK = ROOT / "benchmarks" / "work"


def _match_file(left: object, right: object) -> bool:
    left = str(left)
    right = str(right)
    if left in ("nan", "None", "<NA>") or right in ("nan", "None", "<NA>"):
        return False
    return (
        left == right
        or left.endswith("/" + right)
        or right.endswith("/" + left)
    )


def _fault_pairs(faults: pd.DataFrame, version: str) -> list[tuple[str, int]]:
    group = faults[faults["source_version"] == version]
    return [(str(row.file), int(row.fault_line)) for row in group.itertuples()]


def _rank_interval(
    frame: pd.DataFrame,
    score_column: str,
    faults: list[tuple[str, int]],
    *,
    line_column: str = "line_number",
) -> tuple[float, float, float]:
    """Return best, mid, and worst rank across tied lines."""
    size = len(frame)
    values: list[tuple[float, float, float]] = []
    for fault_file, fault_line in faults:
        mask = np.array(
            [
                _match_file(row_file, fault_file) and int(row_line) == fault_line
                for row_file, row_line in zip(frame["file"], frame[line_column])
            ]
        )
        if not mask.any():
            values.append((size + 1, size + 1, size + 1))
            continue
        score = frame.loc[mask, score_column].iloc[0]
        greater = float((frame[score_column] > score).sum())
        greater_equal = float((frame[score_column] >= score).sum())
        values.append(
            (1.0 + greater, (1.0 + greater + greater_equal) / 2.0, greater_equal)
        )
    return (
        min(value[0] for value in values),
        min(value[1] for value in values),
        min(value[2] for value in values),
    )


def _normalize(series: pd.Series) -> pd.Series:
    spread = series.max() - series.min()
    return (series - series.min()) / spread if spread > 0 else series * 0.0


def _tie_row(
    method: str, interval: tuple[float, float, float], size: int
) -> dict:
    best, mid, worst = interval
    return {
        "method": method,
        "top1_best": float(best == 1),
        "top1_mid": float(mid == 1),
        "top1_worst": float(worst == 1),
        "top5_best": float(best <= 5),
        "top5_mid": float(mid <= 5),
        "top5_worst": float(worst <= 5),
        "mrr_best": 1.0 / best,
        "mrr_mid": 1.0 / mid,
        "mrr_worst": 1.0 / worst,
        "rank_best": best,
        "rank_mid": mid,
        "rank_worst": worst,
        "exam_mid": mid / max(size, 1),
    }


def _metallaxis_scores(pairs: pd.DataFrame, version: str, collapse: str) -> pd.DataFrame:
    variant = pairs[pairs["source_version"] == version]
    baseline = dict(zip(variant["test_id"], variant["baseline_test_result"], strict=True))
    failing = {test for test, result in baseline.items() if result == "fail"}
    passing = {test for test, result in baseline.items() if result == "pass"}
    total_failing = len(failing)
    grouped = variant.groupby(
        ["file", "line_number", "test_id"], as_index=False
    )["outcome_changed"].agg(collapse)
    rows = []
    for (file, line), group in grouped.groupby(["file", "line_number"]):
        values = dict(zip(group["test_id"], group["outcome_changed"]))
        failing_mass = sum(float(values.get(test, 0.0)) for test in failing)
        passing_mass = sum(float(values.get(test, 0.0)) for test in passing)
        denominator = np.sqrt(total_failing * (failing_mass + passing_mass))
        score = failing_mass / denominator if denominator > 0 else 0.0
        rows.append(
            {
                "file": str(file),
                "line_number": int(line),
                "suspiciousness": float(score),
            }
        )
    return pd.DataFrame(rows)


def _ptyes(labels: np.ndarray, scores: np.ndarray) -> dict:
    result = {
        "n": int(len(labels)),
        "positive_rate": float(labels.mean()) if len(labels) else float("nan"),
        "brier": float(brier_score_loss(labels, scores)) if len(labels) else float("nan"),
    }
    if len(labels) and len(np.unique(labels)) > 1:
        result["auc"] = float(roc_auc_score(labels, scores))
        result["ap"] = float(average_precision_score(labels, scores))
    else:
        result["auc"] = float("nan")
        result["ap"] = float("nan")
    return result


def expressibility(pairs: pd.DataFrame, faults: pd.DataFrame) -> pd.DataFrame:
    existing = OUT / "expressibility_bugs.csv"
    if existing.exists() and not WORK.exists():
        return pd.read_csv(existing)

    cache: dict[tuple[str, str, str], set[int]] = {}

    def mutatable(project: str, bug: str, file: str) -> bool:
        key = (project, bug, file)
        if key not in cache:
            path = WORK / f"{project}_{bug}" / project / file
            try:
                cache[key] = candidate_lines(path.read_text())
            except Exception:
                cache[key] = set()
        return key in cache

    rows = []
    for row in faults.itertuples():
        project, bug = str(row.source_version).split("-", 1)
        mut_lines = cache.get((project, bug, row.file))
        if mut_lines is None:
            path = WORK / f"{project}_{bug}" / project / row.file
            try:
                mut_lines = candidate_lines(path.read_text())
            except Exception:
                mut_lines = set()
            cache[(project, bug, row.file)] = mut_lines
        subset = pairs[
            (pairs["source_version"] == row.source_version)
            & (pairs["file"] == row.file)
            & (pairs["line_number"] == int(row.fault_line))
        ]
        sampled = len(subset) > 0
        flips = (
            (subset["outcome_changed"] == 1)
            & (subset["baseline_test_result"] == "fail")
        ).sum() > 0
        rows.append(
            {
                "source_version": row.source_version,
                "bug": str(row.source_version).removesuffix("-fixed"),
                "project": project,
                "file": row.file,
                "fault_line": int(row.fault_line),
                "mutatable": int(int(row.fault_line) in mut_lines),
                "sampled": int(sampled),
                "flips": int(flips),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "expressibility_fault_lines.csv", index=False)
    bug = (
        frame.groupby(["bug", "project"], as_index=False)
        .agg(
            fault_lines=("fault_line", "size"),
            mutatable_lines=("mutatable", "sum"),
            sampled_lines=("sampled", "sum"),
            flipping_lines=("flips", "sum"),
        )
    )
    bug["level"] = "mutatable_only"
    bug.loc[bug["sampled_lines"] > 0, "level"] = "sampled_no_flip"
    bug.loc[bug["flipping_lines"] > 0, "level"] = "flipping"
    bug.to_csv(OUT / "expressibility_bugs.csv", index=False)
    return bug


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: tuple[str, ...],
    categorical: tuple[str, ...],
    numeric: tuple[str, ...],
) -> np.ndarray:
    features = ColumnTransformer(
        [
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ),
                list(categorical),
            ),
            ("numeric", "passthrough", list(numeric)),
        ],
        remainder="drop",
    )
    classifier = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_depth=3,
        random_state=42,
    )
    pipeline = Pipeline([("features", features), ("classifier", classifier)])
    x_train = train.loc[:, feature_columns].copy()
    x_test = test.loc[:, feature_columns].copy()
    for column in categorical:
        x_train[column] = x_train[column].astype("string").fillna("__missing__").astype(str)
        x_test[column] = x_test[column].astype("string").fillna("__missing__").astype(str)
    labels = train["outcome_changed"].astype(int).to_numpy()
    counts = np.bincount(labels)
    weights = np.array([len(labels) / (2 * counts[label]) for label in labels])
    pipeline.fit(x_train, labels, classifier__sample_weight=weights)
    probabilities = pipeline.predict_proba(x_test)
    classes = pipeline.named_steps["classifier"].classes_
    positive = np.flatnonzero(classes == 1)
    if positive.size == 0:
        return np.zeros(len(x_test), dtype=float)
    return probabilities[:, positive[0]]


def pair_metrics(pairs: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    bugs = sorted(
        pairs.loc[~pairs["source_version"].str.endswith("-fixed"), "source_version"].unique()
    )
    full_categorical = (
        "mutation_type",
        "parent_node_type",
        "baseline_test_result",
        "mutated_operator",
        "replacement_operator",
    )
    full_numeric = ("ast_depth", "test_covers_mutation")
    full_features = (
        "mutation_type",
        "ast_depth",
        "parent_node_type",
        "test_covers_mutation",
        "baseline_test_result",
        "mutated_operator",
        "replacement_operator",
    )
    configs = {
        "full": (full_features, full_categorical, full_numeric),
        "no_coverage": (
            tuple(column for column in full_features if column != "test_covers_mutation"),
            full_categorical,
            ("ast_depth",),
        ),
        "coverage_baseline": (
            ("test_covers_mutation", "baseline_test_result"),
            ("baseline_test_result",),
            ("test_covers_mutation",),
        ),
    }

    results = []
    line_scores: dict[str, pd.DataFrame] = {}
    for name, (feature_columns, categorical, numeric) in configs.items():
        collected = []
        for version in bugs:
            bug_key = str(version).removesuffix("-fixed")
            train = pairs[
                ~pairs["source_version"].str.removesuffix("-fixed").eq(bug_key)
            ]
            test = pairs[pairs["source_version"] == version].copy()
            test["probability"] = _fit_predict(
                train, test, feature_columns, categorical, numeric
            )
            collected.append(test)
            if name == "full":
                baseline = dict(
                    zip(test["test_id"], test["baseline_test_result"], strict=True)
                )
                for aggregation in ("mean", "max"):
                    ranking = rank_suspicious_lines(
                        test[["file", "line_number", "test_id", "probability"]],
                        baseline,
                        aggregation=aggregation,
                    )
                    line_scores[f"ML-PMT-{aggregation}::{version}"] = ranking
        all_rows = pd.concat(collected, ignore_index=True)
        if name == "full":
            all_rows.loc[
                :,
                [
                    "source_version",
                    "baseline_test_result",
                    "outcome_changed",
                    "test_covers_mutation",
                    "probability",
                ],
            ].to_csv(OUT / "pair_predictions.csv", index=False)
        labels = all_rows["outcome_changed"].astype(int).to_numpy()
        scores = all_rows["probability"].to_numpy()
        for subset, mask in (
            ("all", np.ones(len(all_rows), dtype=bool)),
            ("failing", (all_rows["baseline_test_result"] == "fail").to_numpy()),
            ("passing", (all_rows["baseline_test_result"] == "pass").to_numpy()),
            ("covered", (all_rows["test_covers_mutation"] == 1).to_numpy()),
        ):
            metrics = _ptyes(labels[mask], scores[mask])
            metrics.update({"config": name, "subset": subset})
            results.append(metrics)
        if name == "full":
            coverage = _ptyes(labels, all_rows["test_covers_mutation"].to_numpy())
            coverage.update({"config": "coverage_only", "subset": "all"})
            results.append(coverage)

    result = pd.DataFrame(results)
    result.to_csv(OUT / "pair_metrics.csv", index=False)
    return result, line_scores

def failing_diagnostics(pairs: pd.DataFrame) -> None:
    predictions = pd.read_csv(OUT / "pair_predictions.csv")
    rows = []
    for version, group in predictions.groupby("source_version"):
        group = group[group["baseline_test_result"] == "fail"]
        labels = group["outcome_changed"].astype(int).to_numpy()
        if len(np.unique(labels)) > 1:
            rows.append(
                {
                    "source_version": version,
                    "n": len(group),
                    "positives": int(labels.sum()),
                    "auc": float(roc_auc_score(labels, group["probability"])),
                    "ap": float(average_precision_score(labels, group["probability"])),
                }
            )
        else:
            rows.append(
                {
                    "source_version": version,
                    "n": len(group),
                    "positives": int(labels.sum()),
                    "auc": float("nan"),
                    "ap": float("nan"),
                }
            )
    per_bug = pd.DataFrame(rows)
    per_bug.to_csv(OUT / "failing_per_bug.csv", index=False)

    full_features = (
        "mutation_type",
        "ast_depth",
        "parent_node_type",
        "test_covers_mutation",
        "baseline_test_result",
        "mutated_operator",
        "replacement_operator",
    )
    categorical = (
        "mutation_type",
        "parent_node_type",
        "baseline_test_result",
        "mutated_operator",
        "replacement_operator",
    )
    numeric = ("ast_depth", "test_covers_mutation")
    collected = []
    for version in sorted(faults_versions(pairs)):
        key = str(version).removesuffix("-fixed")
        train = pairs[
            ~pairs["source_version"].str.removesuffix("-fixed").eq(key)
        ]
        train = train[train["baseline_test_result"] == "fail"]
        test = pairs[pairs["source_version"] == version]
        test = test[test["baseline_test_result"] == "fail"].copy()
        if test.empty:
            continue
        test["probability"] = _fit_predict(
            train, test, full_features, categorical, numeric
        )
        collected.append(test)
    pooled = pd.concat(collected, ignore_index=True)
    labels = pooled["outcome_changed"].astype(int).to_numpy()
    metrics = _ptyes(labels, pooled["probability"].to_numpy())
    metrics["config"] = "failing_only"
    per_bug_metrics = []
    for version, group in pooled.groupby("source_version"):
        y = group["outcome_changed"].astype(int).to_numpy()
        if len(np.unique(y)) > 1:
            per_bug_metrics.append(roc_auc_score(y, group["probability"]))
    pd.DataFrame(
        [
            {
                "config": "failing_only",
                **metrics,
                "per_bug_mean_auc": float(np.mean(per_bug_metrics)),
                "per_bug_with_both_classes": len(per_bug_metrics),
            }
        ]
    ).to_csv(OUT / "failing_only_metrics.csv", index=False)

    calibration_rows = []
    for name in ("isotonic", "sigmoid"):
        scores = []
        labels = []
        for version in sorted(faults_versions(pairs)):
            key = str(version).removesuffix("-fixed")
            train = pairs[
                ~pairs["source_version"].str.removesuffix("-fixed").eq(key)
            ]
            test = pairs[pairs["source_version"] == version]
            model = OutcomeChangeModel(calibration=name, random_state=42).fit(train)
            scores.append(model.predict_probabilities(test))
            labels.append(test["outcome_changed"].astype(int).to_numpy())
        y = np.concatenate(labels)
        s = np.concatenate(scores)
        calibration_rows.append(
            {
                "calibration": name,
                "auc": roc_auc_score(y, s),
                "ap": average_precision_score(y, s),
                "brier": brier_score_loss(y, s),
            }
        )
    pd.DataFrame(calibration_rows).to_csv(
        OUT / "calibration_metrics.csv", index=False
    )


def faults_versions(pairs: pd.DataFrame) -> list[str]:
    return sorted(
        str(version)
        for version in pairs.loc[
            ~pairs["source_version"].str.endswith("-fixed"), "source_version"
        ].unique()
    )

def tie_metrics(
    pairs: pd.DataFrame,
    faults: pd.DataFrame,
    line_scores: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    import pickle

    cache = pickle.loads(CACHE.read_bytes())
    bugs = sorted(faults["source_version"].unique())
    records = []
    for version in bugs:
        version = str(version)
        fault_list = _fault_pairs(faults, version)
        variant = pairs[pairs["source_version"] == version]
        mutant_keys = set(
            zip(variant["file"].astype(str), variant["line_number"].astype(int))
        )
        if version in cache and cache[version] is not None:
            spectrum = cache[version]["spectrum"].copy()
            for method, column in (("SBFL-Ochiai", "ochiai"), ("SBFL-Tarantula", "tarantula")):
                interval = _rank_interval(spectrum, column, fault_list)
                records.append(
                    {"source_version": version, **_tie_row(method, interval, len(spectrum))}
                )
            for collapse in ("mut_max", "mut_mean"):
                spectrum["hybrid"] = np.maximum(
                    _normalize(spectrum["ochiai"].astype(float)),
                    _normalize(spectrum[collapse].astype(float)),
                )
                interval = _rank_interval(spectrum, "hybrid", fault_list)
                records.append(
                    {
                        "source_version": version,
                        **_tie_row(f"Hybrid-{collapse}", interval, len(spectrum)),
                    }
                )
            spectrum["mbfl_sbfl"] = spectrum["mut_max"].astype(float) + 1e-6 * _normalize(
                spectrum["ochiai"].astype(float)
            )
            interval = _rank_interval(spectrum, "mbfl_sbfl", fault_list)
            records.append(
                {
                    "source_version": version,
                    **_tie_row("MBFL+SBFL", interval, len(spectrum)),
                }
            )
        for collapse in ("mean", "max"):
            key = f"ML-PMT-{collapse}::{version}"
            if key in line_scores:
                ranking = line_scores[key]
                interval = _rank_interval(ranking, "suspiciousness", fault_list)
                records.append(
                    {
                        "source_version": version,
                        **_tie_row(f"ML-PMT-{collapse}", interval, len(ranking)),
                    }
                )
            mbfl = _metallaxis_scores(pairs, version, collapse)
            mbfl = mbfl[
                [
                    (file, line) in mutant_keys
                    for file, line in zip(mbfl["file"], mbfl["line_number"])
                ]
            ]
            interval = _rank_interval(mbfl, "suspiciousness", fault_list)
            records.append(
                {
                    "source_version": version,
                    **_tie_row(f"MBFL-real-{collapse}", interval, len(mbfl)),
                }
            )
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "tie_metrics_per_bug.csv", index=False)
    aggregate = (
        frame.groupby("method")
        .agg(
            top1_best=("top1_best", "mean"),
            top1_mid=("top1_mid", "mean"),
            top1_worst=("top1_worst", "mean"),
            mrr_best=("mrr_best", "mean"),
            mrr_mid=("mrr_mid", "mean"),
            mrr_worst=("mrr_worst", "mean"),
            exam_mid=("exam_mid", "mean"),
        )
        .reset_index()
    )
    aggregate.to_csv(OUT / "tie_metrics.csv", index=False)
    return aggregate


def sbfl_only_ranker(pairs: pd.DataFrame, faults: pd.DataFrame) -> pd.DataFrame:
    bugs = sorted(faults["source_version"].unique())
    feature_rows = []
    for version in bugs:
        version = str(version)
        variant = pairs[pairs["source_version"] == version]
        failing = set(variant.loc[variant["baseline_test_result"] == "fail", "test_id"])
        passing = set(variant.loc[variant["baseline_test_result"] == "pass", "test_id"])
        total_failing = len(failing)
        total_passing = len(passing)
        fault_set = set(_fault_pairs(faults, version))
        for (file, line), group in variant.groupby(["file", "line_number"]):
            covering = set(group.loc[group["test_covers_mutation"] == 1, "test_id"])
            a = len(covering & failing)
            b = len(covering & passing)
            c = total_failing - a
            e = total_passing - b
            ochiai_den = np.sqrt((a + b) * (a + c))
            positive = a / (a + c) if a + c else 0.0
            negative = b / (b + e) if b + e else 0.0
            tarantula_den = positive + negative
            dstar_den = b + c
            feature_rows.append(
                {
                    "source_version": version,
                    "file": str(file),
                    "line_number": int(line),
                    "a": a,
                    "b": b,
                    "n_failing": total_failing,
                    "n_passing": total_passing,
                    "n_mutants": int(group["mutant_id"].nunique()),
                    "ochiai": a / ochiai_den if ochiai_den else 0.0,
                    "tarantula": positive / tarantula_den if tarantula_den else 0.0,
                    "dstar": (a * a / dstar_den) if dstar_den else (float(a) if a else 0.0),
                    "is_fault": int(
                        any(
                            _match_file(file, fault_file) and int(line) == fault_line
                            for fault_file, fault_line in fault_set
                        )
                    ),
                }
            )
    frame = pd.DataFrame(feature_rows)
    columns = [
        "a",
        "b",
        "n_failing",
        "n_passing",
        "n_mutants",
        "ochiai",
        "tarantula",
        "dstar",
    ]
    records = []
    for version in bugs:
        version = str(version)
        train = frame[frame["source_version"] != version]
        test = frame[frame["source_version"] == version].copy()
        model = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        )
        model.fit(train[columns], train["is_fault"])
        test["suspiciousness"] = model.predict_proba(test[columns])[:, 1]
        fault_list = _fault_pairs(faults, version)
        interval = _rank_interval(test, "suspiciousness", fault_list)
        records.append({"source_version": version, **_tie_row("SBFL-only-ranker", interval, len(test))})
    result = pd.DataFrame(records)
    result.to_csv(OUT / "sbfl_only_ranker_per_bug.csv", index=False)
    aggregate = pd.DataFrame(
        [
            {
                "method": "SBFL-only-ranker",
                "top1_best": result["top1_best"].mean(),
                "top1_mid": result["top1_mid"].mean(),
                "mrr_best": result["mrr_best"].mean(),
                "mrr_mid": result["mrr_mid"].mean(),
                "mrr_worst": result["mrr_worst"].mean(),
                "exam_mid": result["exam_mid"].mean(),
            }
        ]
    )
    aggregate.to_csv(OUT / "sbfl_only_ranker.csv", index=False)
    return aggregate


def cost(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    buggy = pairs[~pairs["source_version"].str.endswith("-fixed")]
    for version, variant in buggy.groupby("source_version"):
        tests = variant["test_id"].nunique()
        mutants = variant["mutant_id"].nunique()
        full_exec = len(variant)
        cover_exec = int(
            variant.groupby("mutant_id").apply(
                lambda group: (
                    (group["baseline_test_result"] == "fail")
                    | (group["test_covers_mutation"] == 1)
                ).sum(),
                include_groups=False,
            ).sum()
        )
        row = {
            "source_version": version,
            "tests": tests,
            "mutants": mutants,
            "mbfl_full_test_executions": full_exec,
            "mbfl_covering_test_executions": cover_exec,
            "mlpmt_one_run_test_executions": tests,
            "mlpmt_two_runs_test_executions": 2 * tests,
            "sbfl_test_executions": tests,
            "mbfl_suite_runs": mutants,
            "mlpmt_suite_runs": 1,
            "sbfl_suite_runs": 1,
        }
        for budget in (20, 30):
            selected = variant[
                variant["mutant_id"].isin(sorted(variant["mutant_id"].unique())[:budget])
            ]
            row[f"mbfl_{budget}_suite_runs"] = min(budget, mutants)
            row[f"mbfl_{budget}_test_executions"] = len(selected)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "cost_per_bug.csv", index=False)
    aggregate = frame.drop(columns=["source_version"]).mean().to_dict()
    pd.DataFrame([aggregate]).to_csv(OUT / "cost_mean.csv", index=False)
    return frame


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(PAIRS)
    faults = pd.read_csv(FAULTS)
    bug_level = expressibility(pairs, faults)
    metrics, line_scores = pair_metrics(pairs)
    failing_diagnostics(pairs)
    ties = tie_metrics(pairs, faults, line_scores)
    sbfl = sbfl_only_ranker(pairs, faults)
    costs = cost(pairs)
    summary = {
        "expressibility": bug_level["level"].value_counts().to_dict(),
        "fault_lines": {
            "total": int(
                pd.read_csv(OUT / "expressibility_fault_lines.csv").shape[0]
            ),
            "mutatable": int(
                pd.read_csv(OUT / "expressibility_fault_lines.csv")["mutatable"].sum()
            ),
            "sampled": int(
                pd.read_csv(OUT / "expressibility_fault_lines.csv")["sampled"].sum()
            ),
            "flipping": int(
                pd.read_csv(OUT / "expressibility_fault_lines.csv")["flips"].sum()
            ),
        },
        "cost_mean": costs.drop(columns=["source_version"]).mean().round(2).to_dict(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("\nPair metrics:")
    print(metrics.round(3).to_string(index=False))
    print("\nTie-aware metrics:")
    print(ties.round(3).to_string(index=False))
    print("\nSBFL-only ranker:")
    print(sbfl.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
