#!/usr/bin/env python3
"""Read-only replay audit. Uses the experiment's deterministic task generator.

Run after experiments.py: python3 audit_replay.py
Prints checks and main matched-subset diagnostics; does not modify any files.
"""
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


def main():
    base = Path(__file__).resolve().parent
    out = base / "outputs"
    spec = importlib.util.spec_from_file_location("audited_experiment", base / "experiments.py")
    exp = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = exp
    spec.loader.exec_module(exp)
    with (out / "all_raw.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    episodes = defaultdict(list)
    for row in rows:
        episodes[row["condition"], row["family"], row["seed"], row["task"]].append(row)
    conditions = {c["name"]: c for c in exp.CONFIG["conditions"]}
    margin_error = regret_error = 0.0
    for key, episode_rows in episodes.items():
        condition, family, seed, task = key
        info, env = exp.generate_task(int(seed), exp.CONFIG["families"].index(family),
                                      int(task), conditions[condition])
        for row in episode_rows:
            order = [int(value) for value in row["query_order"].split("-") if value]
            assert len(order) == len(set(order)) == int(row["calls"])
            cached, sampled = info.y.copy(), info.tau.copy()
            elapsed = 0.0
            for source in order:
                elapsed += float(info.latency[source])
                cached[source] = env.value(source, elapsed)
                sampled[source] = elapsed
            commit = elapsed + exp.H
            assert abs(float(row["tool_latency"]) - elapsed) < 1e-9
            assert abs(float(row["commit_time"]) - commit) < 1e-9
            assert commit <= exp.DEADLINE + 1e-9
            nominal = info.b + info.W @ cached
            action = int(np.argmax(nominal))
            actual = info.b + info.W @ env.state_for_evaluation(commit)
            radii = info.v * (commit - sampled)
            margins = [float(nominal[action] - nominal[other] - sum(
                abs(info.W[action, j] - info.W[other, j]) * radii[j]
                for j in range(len(cached))))
                for other in range(len(nominal)) if other != action]
            minimum = min(margins)
            margin_error = max(margin_error, abs(minimum - float(row["final_margin"])))
            if row["executed"] == "True":
                assert int(row["action"]) == action
                regret = float(max(actual) - actual[action])
                regret_error = max(regret_error, abs(regret - float(row["utility_regret"])))
                assert (regret > exp.EVAL_TOL) == (row["invalid_executed"] == "True")
            if row["certified"] == "True":
                assert minimum >= -exp.TOL - 1e-12
            if conditions[condition]["actual_bound_multiplier"] == 1:
                assert np.all(abs(env.state_for_evaluation(commit) - cached) <= radii + 1e-10)

    summaries = json.loads((out / "summary.json").read_text())
    summary_error = 0.0
    for summary in summaries:
        selected = [r for r in rows if r["condition"] == summary["condition"]
                    and r["method"] == summary["method"]
                    and (summary["family"] == "all" or r["family"] == summary["family"])]
        count = len(selected)
        certified = sum(r["certified"] == "True" for r in selected)
        executed = sum(r["executed"] == "True" for r in selected)
        invalid = sum(r["invalid_executed"] == "True" for r in selected)
        invalid_certified = sum(r["invalid_certified"] == "True" for r in selected)
        calls = sum(int(r["calls"]) for r in selected)
        delay = sum(float(r["tool_latency"]) for r in selected)
        check = dict(n=count, executed=executed, certified=certified,
                     invalid_executed=invalid, invalid_certified=invalid_certified,
                     execution_coverage=executed/count, certificate_coverage=certified/count,
                     calls_per_arrival=calls/count, latency_per_arrival=delay/count,
                     total_calls_per_certificate=calls/certified if certified else None,
                     total_latency_per_certificate=delay/certified if certified else None)
        for field, value in check.items():
            if value is None:
                assert summary[field] is None
            else:
                summary_error = max(summary_error, abs(value - summary[field]))

    seeds = defaultdict(list)
    for row in rows:
        seeds[row["condition"], row["family"], row["seed"], row["method"]].append(row)
        seeds[row["condition"], "all", row["seed"], row["method"]].append(row)

    def seed_metric(selected, metric):
        certificates = sum(r["certified"] == "True" for r in selected)
        if metric == "certificate_coverage":
            return certificates / len(selected)
        field = "calls" if "calls" in metric else "tool_latency"
        total = sum(float(r[field]) for r in selected)
        if "per_certificate" in metric:
            return total / certificates if certificates else None
        return total / len(selected)

    interval_error = 0.0
    with (out / "paired_seed_intervals.csv").open() as stream:
        intervals = list(csv.DictReader(stream))
    for interval in intervals:
        differences = []
        for seed in map(str, exp.CONFIG["seeds"]):
            key = (interval["condition"], interval["family"], seed)
            proposed = seed_metric(seeds[key + ("timed_h2",)], interval["metric"])
            baseline = seed_metric(seeds[key + (interval["baseline"],)], interval["metric"])
            if proposed is not None and baseline is not None:
                differences.append(proposed - baseline)
        assert int(interval["seeds"]) == len(differences)
        mean = np.mean(differences)
        half = student_t.ppf(.975, len(differences)-1) * np.std(
            differences, ddof=1) / math.sqrt(len(differences))
        interval_error = max(interval_error, abs(float(interval["difference"]) - mean),
                             abs(float(interval["ci_low"]) - (mean-half)),
                             abs(float(interval["ci_high"]) - (mean+half)))

    provenance = json.loads((out / "provenance.json").read_text())
    source_hash = hashlib.sha256((base / "experiments.py").read_bytes()).hexdigest()
    assert provenance["script_sha256"] == source_hash
    timed = {(r["family"], r["seed"], r["task"]): r for r in rows
             if r["condition"] == "main" and r["method"] == "timed_h2"}
    static = {(r["family"], r["seed"], r["task"]): r for r in rows
              if r["condition"] == "main" and r["method"] == "static_h2"}
    common = [k for k in timed if timed[k]["certified"] == "True"
              and static[k]["certified"] == "True"]
    matched = {}
    for field in ["calls", "tool_latency"]:
        a = float(np.mean([float(timed[k][field]) for k in common]))
        b = float(np.mean([float(static[k][field]) for k in common]))
        matched[field] = dict(timed_h2=a, static_h2=b, relative_reduction=1-a/b)
    print(json.dumps(dict(
        rows=len(rows), episodes=len(episodes), interval_rows=len(intervals),
        maximum_margin_error=margin_error, maximum_regret_error=regret_error,
        maximum_selected_summary_error=summary_error, maximum_interval_error=interval_error,
        source_hash_matches=True, source_sha256=source_hash,
        initially_certified=sum(r["certified"] == "True" and r["calls"] == "0"
                                for r in timed.values()),
        common_certificates=len(common), matched_subset=matched,
        timed_only_certificates=sum(timed[k]["certified"] == "True"
                                    and static[k]["certified"] == "False" for k in timed),
        static_only_certificates=sum(timed[k]["certified"] == "False"
                                     and static[k]["certified"] == "True" for k in timed)), indent=2))


if __name__ == "__main__":
    main()
