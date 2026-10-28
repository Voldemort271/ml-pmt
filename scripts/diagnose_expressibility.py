"""Measure the TRUE expressibility ceiling of each non-expressible bug.

For every bug, exhaustively mutate the fault *line(s)* (no sampling budget) and
execute each fault-line mutant against the bug's failing tests. A bug is truly
expressible if at least one fault-line mutant flips a failing test's outcome —
this separates the "sampling gap" (deep fault lines never mutated under the
300-mutant cap) from the "operator gap" (mutated but no flip).

This is a diagnostic: it does NOT feed back into the corpus or model features.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.pytest_runner import run_pytest
from mutation_engine.generator import generate_mutants
from scripts.build_bugs import (
    BUGS,
    BUGS_ROOT,
    WORK,
    fault_lines,
    make_sandbox,
    select_tests,
)

NON_EXPRESSIBLE = [
    "black-10", "black-11", "black-12", "black-15", "black-16", "black-17",
    "black-18", "black-20", "black-21", "black-23", "black-4", "black-5",
    "black-6", "black-7", "black-8", "black-9",
    "thefuck-1", "thefuck-2", "thefuck-25", "thefuck-26", "thefuck-27",
    "thefuck-28", "thefuck-29", "thefuck-31", "thefuck-32", "thefuck-6",
    "thefuck-8",
    "tqdm-1", "tqdm-4", "tqdm-5", "tqdm-6", "tqdm-8", "tqdm-9",
]


def _diagnose(spec: dict) -> dict:
    project, bug = spec["project"], spec["bug"]
    variant_id = f"{project}-{bug}"
    python = WORK / spec["venv"] / "venv" / "bin" / "python"
    bug_dir = BUGS_ROOT / project / "bugs" / str(bug)
    checkout = WORK / f"{project}_{bug}" / project
    sandbox = make_sandbox(checkout, f"{project}_{bug}_exprdiag", project)
    tests, failing = select_tests(sandbox, bug_dir, python)
    faults = fault_lines((bug_dir / "bug_patch.txt").read_text())

    if not failing:
        return {"variant": variant_id, "status": "no-failing-tests",
                "n_fault_mutants": 0, "flip": False}

    # Execute only failing tests to keep the diagnostic cheap.
    baseline = run_pytest(sandbox, list(failing), python)

    total_mutants = 0
    flipped = False
    flip_detail = ""
    for relative_file, lines in faults.items():
        path = sandbox / relative_file
        if not path.exists():
            continue
        original = path.read_text()
        fault_mutants = generate_mutants(original, limit=None, only_lines=set(lines))
        for mutant in fault_mutants:
            total_mutants += 1
            path.write_text(mutant["source"])
            try:
                results = run_pytest(sandbox, list(failing), python)
            finally:
                path.write_text(original)
            for test_id in failing:
                if results.get(test_id, "fail") != baseline.get(test_id, "fail"):
                    flipped = True
                    flip_detail = (
                        f"{relative_file}:{mutant['line']} "
                        f"{mutant['mutation_type']}/{mutant['detail']}"
                    )
                    break
            if flipped:
                break
        if flipped:
            break

    return {
        "variant": variant_id,
        "status": "expressible" if flipped else "inexpressible",
        "n_fault_mutants": total_mutants,
        "flip": flipped,
        "flip_detail": flip_detail,
    }


def main() -> int:
    from concurrent.futures import ProcessPoolExecutor

    bugs = [s for s in BUGS if f"{s['project']}-{s['bug']}" in NON_EXPRESSIBLE]
    print(f"Diagnosing {len(bugs)} non-expressible bugs (exhaustive fault-line mutation)...",
          flush=True)

    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(_diagnose, spec) for spec in bugs]
        results = []
        for i, future in enumerate(futures, 1):
            r = future.result()
            results.append(r)
            print(
                f"[{i}/{len(bugs)}] {r['variant']:12s} -> {r['status']:14s} "
                f"({r['n_fault_mutants']} fault-line mutants)"
                + (f"  via {r['flip_detail']}" if r["flip"] else ""),
                flush=True,
            )

    df = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(df[["variant", "status", "n_fault_mutants"]].to_string(index=False))
    expr = df[df.status == "expressible"]
    print(f"\nTrue expressibility: {len(expr)}/{len(bugs)} of the previously "
          f"non-expressible bugs are expressible under exhaustive fault-line mutation.")
    print(f"Combined with the original 5, ceiling = {len(expr) + 5}/38 bugs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
