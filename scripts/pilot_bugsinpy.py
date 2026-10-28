"""Pilot new BugsInPy projects to check they slot into the pipeline.

For each (project, bug): do a clean buggy checkout + overlay the fixed test
file(s), run select_tests, then exhaustive fault-line mutation to measure
expressibility. Reuses one git clone per project.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_bugs import fault_lines, make_sandbox, read_info, select_tests
from execution.pytest_runner import run_pytest
from mutation_engine.generator import generate_mutants

WORK = ROOT / "benchmarks" / "work"
BUGS_ROOT = ROOT / "benchmarks" / "BugsInPy" / "projects"

VENV_DEPS = ["pytest==9.1.1", "pytest-cov==7.1.0", "coverage==7.16.2"]

PILOT_BUGS = [
    ("youtube-dl", 3, "string"),
    ("youtube-dl", 10, "string"),
    ("youtube-dl", 20, "string"),
    ("youtube-dl", 25, "string"),
    ("youtube-dl", 15, "numeric"),
    ("youtube-dl", 26, "numeric"),
    ("youtube-dl", 5, "condition"),
    ("youtube-dl", 7, "condition"),
    ("tornado", 2, "string"),
    ("tornado", 8, "string"),
    ("tornado", 12, "string"),
    ("tornado", 13, "string"),
    ("tornado", 1, "condition"),
    ("tornado", 3, "condition"),
    ("tornado", 9, "condition"),
    ("tornado", 14, "condition"),
]


def _clone(project: str) -> Path:
    url = _project_url(project)
    repo = WORK / f"{project}_pilot" / project
    if not (repo / ".git").exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", url, str(repo)], check=True)
    return repo


def _project_url(project: str) -> str:
    info = (BUGS_ROOT / project / "project.info").read_text()
    for line in info.splitlines():
        if line.startswith("github_url="):
            return line.split('"')[1]
    raise RuntimeError(f"no github_url for {project}")


def _venv(project: str) -> Path:
    venv = WORK / f"{project}_pilot" / "venv"
    if not (venv / "bin" / "python").exists():
        subprocess.run(["uv", "venv", str(venv), "--python", "3.11"], check=True)
        subprocess.run(
            ["uv", "pip", "install", "--python", str(venv / "bin" / "python"), *VENV_DEPS],
            check=True,
        )
    return venv / "bin" / "python"


def _checkout(repo: Path, project: str, bug: int) -> Path:
    info = read_info(BUGS_ROOT / project / "bugs" / str(bug))
    buggy, fixed = info["buggy_commit_id"], info["fixed_commit_id"]
    subprocess.run(["git", "-C", str(repo), "checkout", "--", "."], check=False)
    subprocess.run(["git", "-C", str(repo), "reset", "--hard", buggy], check=True)
    test_files = [t for t in info["test_file"].split(";") if t.strip().endswith(".py")]
    if test_files:
        subprocess.run(
            ["git", "-C", str(repo), "checkout", fixed, "--", *test_files], check=True
        )
    return repo


def _check_expressible(sandbox: Path, faults, failing, python) -> tuple[bool, str]:
    baseline = run_pytest(sandbox, list(failing), python)
    for f, lines in faults.items():
        p = sandbox / f
        if not p.exists():
            continue
        orig = p.read_text()
        for m in generate_mutants(orig, limit=None, only_lines=set(lines)):
            p.write_text(m["source"])
            try:
                res = run_pytest(sandbox, list(failing), python)
            finally:
                p.write_text(orig)
            for t in failing:
                if res.get(t, "fail") != baseline.get(t, "fail"):
                    return True, f"{f}:{m['line']} {m['mutation_type']}/{m['detail']}"
    return False, ""


def main() -> int:
    print(f"Piloting {len(PILOT_BUGS)} bugs", flush=True)
    repos: dict[str, Path] = {}
    venvs: dict[str, Path] = {}
    for i, (project, bug, ftype) in enumerate(PILOT_BUGS, 1):
        try:
            if project not in repos:
                repos[project] = _clone(project)
                venvs[project] = _venv(project)
            repo = _checkout(repos[project], project, bug)
            python = venvs[project]
            bug_dir = BUGS_ROOT / project / "bugs" / str(bug)
            sandbox = make_sandbox(repo, f"{project}_{bug}_pilot", project)
            tests, failing = select_tests(sandbox, bug_dir, python)
            faults = fault_lines((bug_dir / "bug_patch.txt").read_text())
            expr, via = _check_expressible(sandbox, faults, failing, python)
            status = "EXPRESSIBLE" if expr else "inexpressible"
            print(
                f"[{i}/{len(PILOT_BUGS)}] {project}-{bug:2d} ({ftype:9s}) "
                f"{status:12s} tests={len(tests)} failing={len(failing)}"
                + (f"  via {via}" if expr else ""),
                flush=True,
            )
        except Exception as e:
            print(f"[{i}/{len(PILOT_BUGS)}] {project}-{bug:2d} FAILED: {type(e).__name__}: {e}",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
