#!/usr/bin/env python3
"""Verify frozen results, then reproduce all numeric outputs in a temporary copy.

Run from any directory: python research/verify_release.py
The archived experiment, outputs and figures are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parent
ATOL = 1e-10
RTOL = 1e-10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(left, right, where, counts):
    """Exact structure/text/integers; tolerant finite floating-point values."""
    counts["values"] += 1
    if isinstance(left, dict):
        require(isinstance(right, dict) and left.keys() == right.keys(), where)
        for key in left:
            compare(left[key], right[key], f"{where}.{key}", counts)
    elif isinstance(left, list):
        require(isinstance(right, list) and len(left) == len(right), where)
        for i, (a, b) in enumerate(zip(left, right)):
            compare(a, b, f"{where}[{i}]", counts)
    elif isinstance(left, bool) or left is None or isinstance(left, int):
        require(type(left) is type(right) and left == right, where)
    else:
        try:
            a, b = float(left), float(right)
        except (ValueError, TypeError):
            require(left == right, f"{where}: {left!r} != {right!r}")
            return
        if math.isnan(a) or math.isnan(b):
            require(math.isnan(a) and math.isnan(b), where)
        else:
            require(math.isfinite(a) and math.isfinite(b), where)
            require(math.isclose(a, b, abs_tol=ATOL, rel_tol=RTOL),
                    f"{where}: {a} != {b}")
            counts["max_absolute_numeric_difference"] = max(
                counts["max_absolute_numeric_difference"], abs(a-b))


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def validate_inventory():
    config = json.loads((ROOT / "outputs/configuration.json").read_text())
    rows = read_csv(ROOT / "outputs/all_raw.csv")
    expected = {(c["name"], f, str(s), str(t), m)
                for c in config["conditions"] for f in config["families"]
                for s in config["seeds"] for t in range(c["tasks"])
                for m in config["methods"]}
    identity = lambda r: tuple(r[k] for k in ("condition", "family", "seed", "task", "method"))
    keys = [identity(r) for r in rows]
    require(len(keys) == len(set(keys)) and set(keys) == expected,
            "Missing, duplicated or unexpected policy rows")
    for row in rows:
        for field in ("executed", "certified", "invalid_executed", "invalid_certified"):
            require(row[field] in ("True", "False"), f"Invalid boolean: {field}")
        executed = row["executed"] == "True"
        certified = row["certified"] == "True"
        invalid = row["invalid_executed"] == "True"
        require(certified == (executed and row["method"] != "ungated_no_refresh"),
                "Inconsistent gate flags")
        require((row["invalid_certified"] == "True") == (certified and invalid),
                "Inconsistent invalid certificate flag")
        require(not invalid or executed, "Invalid action counted without execution")
        require(executed or (row["action"] == "-1" and math.isnan(float(row["utility_regret"]))),
                "Deferred task has an action or measured regret")
    for condition in config["conditions"]:
        subset = [r for r in rows if r["condition"] == condition["name"]]
        require(subset == read_csv(ROOT / "outputs" / (condition["name"] + "_raw.csv")),
                f"Condition CSV differs: {condition['name']}")
    return len(rows), len(expected) // len(config["methods"])


def run(script, cwd, *args):
    result = subprocess.run([sys.executable, str(script), *args], cwd=cwd,
                            capture_output=True, text=True, check=True)
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-secondary-drift", action="store_true",
                        help="Report, but do not fail on, rerun drift limited to full_order and timed_h3 and their summaries.")
    args = parser.parse_args()
    row_count, task_count = validate_inventory()
    replay = json.loads(run(ROOT / "audit_replay.py", ROOT))
    for key in ("maximum_margin_error", "maximum_regret_error",
                "maximum_selected_summary_error", "maximum_interval_error"):
        require(math.isfinite(replay[key]) and replay[key] <= ATOL,
                f"Replay discrepancy exceeds tolerance: {key}={replay[key]}")
    require(replay["rows"] == row_count and replay["episodes"] == task_count,
            "Replay counts differ")
    counts = {"values": 0, "max_absolute_numeric_difference": 0.0}
    checked = []
    differences = []
    unexpected = []
    with tempfile.TemporaryDirectory(prefix="certifying-decisions-") as directory:
        work = Path(directory)
        for name in ("experiments.py", "plot_results.py"):
            shutil.copy2(ROOT / name, work / name)
        run(work / "experiments.py", work, "--output", "outputs")
        run(work / "plot_results.py", work)
        for source in sorted((ROOT / "outputs").iterdir()):
            if source.suffix not in (".csv", ".json") or source.name == "provenance.json":
                continue
            reproduced = work / "outputs" / source.name
            if source.suffix == ".csv":
                left, right = read_csv(source), read_csv(reproduced)
            else:
                left, right = json.loads(source.read_text()), json.loads(reproduced.read_text())
            if isinstance(left, list) and isinstance(right, list):
                require(len(left) == len(right), f"{source.name}: row count differs")
                for i, (a, b) in enumerate(zip(left, right)):
                    try:
                        compare(a, b, f"{source.name}[{i}]", counts)
                    except ValueError as error:
                        secondary = a.get("method", a.get("baseline")) in ("full_order", "timed_h3")
                        identity_fields = ("condition", "family", "seed", "task", "method", "baseline", "metric")
                        identity = {k: a[k] for k in identity_fields if k in a}
                        require(all(b.get(k) == v for k, v in identity.items()),
                                f"{source.name}[{i}]: identity differs")
                        change = {"file": source.name, "row": i, "identity": identity,
                                  "first_difference": str(error)}
                        differences.append(change)
                        if not secondary:
                            unexpected.append(change)
            else:
                compare(left, right, source.name, counts)
            checked.append(source.name)
        original_provenance = json.loads((ROOT / "outputs/provenance.json").read_text())
        new_provenance = json.loads((work / "outputs/provenance.json").read_text())
        for key in ("script_sha256", "numpy_version", "tasks", "policy_evaluations", "actual_data"):
            require(original_provenance[key] == new_provenance[key], f"Provenance mismatch: {key}")
        for name in ("latency_sensitivity", "bound_misspecification", "two_source_example"):
            for suffix in (".pdf", ".png"):
                require((work / "figures" / (name + suffix)).stat().st_size > 0,
                        f"Missing regenerated figure: {name}{suffix}")
    report = {
        "status": ("failed" if unexpected or (differences and not args.allow_secondary_drift)
                   else "archive_and_primary_passed_secondary_drift" if differences else "passed"),
        "policy_rows": row_count, "tasks": task_count,
        "absolute_tolerance": ATOL, "relative_tolerance": RTOL,
        "compared_output_files": checked, "comparison": counts,
        "rerun_differences": differences,
        "unexpected_difference_count": len(unexpected),
        "secondary_drift_allowed": args.allow_secondary_drift,
        "original_results_replay": replay,
        "experiment_sha256": hashlib.sha256((ROOT / "experiments.py").read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "scipy": scipy.__version__,
                        "matplotlib": matplotlib.__version__},
        "scope": "Full rerun of the supplied simulator, exact row inventory, numeric output comparisons and separate arithmetic replay. Not an independent simulator implementation or external peer review.",
        "figures": "All six figure files regenerated; plotted numeric summaries compared. Image bytes are not required to match across platforms.",
    }
    print(json.dumps(report, indent=2, allow_nan=False))
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
