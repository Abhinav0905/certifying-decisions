#!/usr/bin/env python3
"""Synthetic serial-query study. No policy is given an unqueried true state.

All settings are fixed in CONFIG before simulation. The implementation is
deliberately small enough to audit. Run: python3 experiments.py --output outputs
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


CONFIG = {
    "seeds": list(range(100, 120)),
    "main_tasks_per_seed_per_family": 100,
    "supplement_tasks_per_seed_per_family": 25,
    "sources": 5,
    "actions": 4,
    "families": ["independent_option", "dense_shared", "sparse_mixed"],
    "initial_age_uniform": [0.0, 8.0],
    "velocity_bound_uniform": [0.03, 0.15],
    "query_latency_uniform": [0.2, 1.8],
    "nominal_gap_uniform": [0.1, 1.5],
    "commit_gap": 0.5,
    "deadline": 12.0,
    "ttl_age_threshold": 3.0,
    "certificate_tolerance": 1e-10,
    "evaluation_tolerance": 1e-8,
    "query_limit": "at most once per source per episode",
    "coefficient_normalization": "each action row has L1 norm 1",
    "dense_weights": "independent standard normal entries, row L1 normalization",
    "independent_weights": "own source coefficient 0.5; fifth shared source coefficient 0.5",
    "sparse_weights": "two distinct sources per action, normal entries, row L1 normalization",
    "initial_state": "independent standard normal coordinates",
    "motion": "x0_i + bound_i * rho_i * t; rho uniform[-1,1]",
    "sinusoidal_motion": "x0_i + bound_i*rho_i/omega_i*(sin(omega_i*t+phase_i)-sin(phase_i)); omega uniform[0.12,0.65], phase uniform[0,2*pi]",
    "offset_construction": "random target action offset makes its cached-center advantage over every competitor at least sampled gap, with exactly one competitor at sampled gap",
    "forecast": "keep cached centers fixed, update all timestamps under serial completions and actual commit gap; replan after one returned observation",
    "static_forecast_ablation": "same horizon-2 search, but records unqueried by a forecast sequence keep their current age plus commit gap; queried records age through later forecast queries",
    "full_order": "maximize minimum final forecast margin among all full permutations; break ties lexicographically; execute this fixed order with correct gate after every return; skip reads that cannot fit deadline",
    "ttl": "initial age > 3, increasing source-index order; correct gate after every return; skip infeasible reads",
    "weighted_age": "max competitor coefficient contrast * stated bound * current age / query latency; among feasible unqueried sources",
    "oldest_first": "greatest current age among feasible unqueried sources; index tie break",
    "conditions": [
        {"name": "main", "motion": "linear", "latency_multiplier": 1.0, "actual_bound_multiplier": 1.0, "tasks": 100},
        {"name": "latency_quarter", "motion": "linear", "latency_multiplier": 0.25, "actual_bound_multiplier": 1.0, "tasks": 25},
        {"name": "latency_double", "motion": "linear", "latency_multiplier": 2.0, "actual_bound_multiplier": 1.0, "tasks": 25},
        {"name": "sinusoidal", "motion": "sinusoidal", "latency_multiplier": 1.0, "actual_bound_multiplier": 1.0, "tasks": 25},
        {"name": "bound_double", "motion": "linear", "latency_multiplier": 1.0, "actual_bound_multiplier": 2.0, "tasks": 25},
        {"name": "bound_fivefold", "motion": "linear", "latency_multiplier": 1.0, "actual_bound_multiplier": 5.0, "tasks": 25},
    ],
    "methods": ["ungated_no_refresh", "full_order", "ttl", "weighted_age", "oldest_first", "static_h2", "timed_h2", "timed_h3"],
    "confidence_intervals": "paired seed-level differences, Student t 95% CI, 20 independent seeds; no multiple-comparison correction",
    "study_status": "synthetic computational experiment; not a production benchmark or an evaluation of LLM extraction",
}

TOL = CONFIG["certificate_tolerance"]
EVAL_TOL = CONFIG["evaluation_tolerance"]
H = CONFIG["commit_gap"]
DEADLINE = CONFIG["deadline"]


@dataclass(frozen=True)
class PolicyInput:
    W: np.ndarray
    b: np.ndarray
    y: np.ndarray
    tau: np.ndarray
    v: np.ndarray
    latency: np.ndarray


@dataclass(frozen=True)
class Environment:
    x0: np.ndarray
    actual_velocity: np.ndarray
    omega: np.ndarray
    phase: np.ndarray
    motion: str

    def value(self, source: int, at: float) -> float:
        if self.motion == "linear":
            return float(self.x0[source] + self.actual_velocity[source] * at)
        return float(self.x0[source] + self.actual_velocity[source] / self.omega[source]
                     * (np.sin(self.omega[source] * at + self.phase[source]) - np.sin(self.phase[source])))

    def state_for_evaluation(self, at: float) -> np.ndarray:
        return np.array([self.value(i, at) for i in range(len(self.x0))])


def generate_task(seed: int, family_index: int, task_index: int, condition: dict):
    # A task identity is reused across conditions, giving paired exogenous draws.
    rng = np.random.default_rng(np.random.SeedSequence([seed, family_index, task_index]))
    n, k = CONFIG["sources"], CONFIG["actions"]
    x0 = rng.normal(size=n)
    v = rng.uniform(*CONFIG["velocity_bound_uniform"], size=n)
    rho = rng.uniform(-1.0, 1.0, size=n)
    ages = rng.uniform(*CONFIG["initial_age_uniform"], size=n)
    latency = rng.uniform(*CONFIG["query_latency_uniform"], size=n) * condition["latency_multiplier"]
    omega = rng.uniform(0.12, 0.65, size=n)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n)
    W = np.zeros((k, n))
    if family_index == 0:
        W[np.arange(k), np.arange(k)] = 0.5
        W[:, -1] = 0.5
    elif family_index == 1:
        W = rng.normal(size=(k, n))
        W /= np.abs(W).sum(axis=1, keepdims=True)
    else:
        for a in range(k):
            ids = rng.choice(n, size=2, replace=False)
            W[a, ids] = rng.normal(size=2)
        W /= np.abs(W).sum(axis=1, keepdims=True)
    env = Environment(x0, v * rho * condition["actual_bound_multiplier"], omega, phase, condition["motion"])
    tau = -ages
    y = np.array([env.value(i, tau[i]) for i in range(n)])
    target = int(rng.integers(k))
    gap = float(rng.uniform(*CONFIG["nominal_gap_uniform"]))
    center_scores = W @ y
    b = np.zeros(k)
    b[target] = max(np.delete(center_scores, target)) + gap - center_scores[target]
    info = PolicyInput(W, b, y, tau, v, latency)
    return info, env


def certificate(info: PolicyInput, y: np.ndarray, tau: np.ndarray, commit: float):
    """Exact worst-case affine comparison over a Cartesian uncertainty box."""
    scores = info.b + info.W @ y
    a = int(np.argmax(scores))  # stable index tie break
    others = np.arange(len(scores)) != a
    C = np.abs(info.W[a] - info.W[others])
    radii = info.v * np.maximum(0.0, commit - tau)
    margins = scores[a] - scores[others] - C @ radii
    minimum = float(margins.min())
    return minimum >= -TOL, a, minimum


SEQUENCES = {}
for mask in range(1 << CONFIG["sources"]):
    remaining = [i for i in range(CONFIG["sources"]) if mask & (1 << i)]
    for horizon in (2, 3, 5):
        seqs = []
        lengths = range(1, min(horizon, len(remaining)) + 1) if horizon != 5 else [len(remaining)]
        for length in lengths:
            if length:
                seqs.extend(itertools.permutations(remaining, length))
        SEQUENCES[mask, horizon] = seqs


def predicted_margins(info, y, tau, elapsed, sequences, freeze_unqueried=False):
    """Vectorized forecast. This never sees Environment or future true values."""
    if not sequences:
        return np.empty(0), np.empty(0)
    completion = np.array([sum(info.latency[j] for j in seq) for seq in sequences])
    candidate_tau = np.broadcast_to(tau, (len(sequences), len(tau))).copy()
    queried = np.zeros_like(candidate_tau, dtype=bool)
    for row, seq in enumerate(sequences):
        finish = elapsed
        for j in seq:
            finish += info.latency[j]
            candidate_tau[row, j] = finish
            queried[row, j] = True
    future_commit = elapsed + completion + H
    ages = future_commit[:, None] - candidate_tau
    if freeze_unqueried:
        ages = np.where(queried, ages, elapsed + H - tau)
    radius = info.v[None, :] * np.maximum(0.0, ages)
    scores = info.b + info.W @ y
    a = int(np.argmax(scores))
    others = np.arange(len(scores)) != a
    C = np.abs(info.W[a] - info.W[others])
    margins = (scores[a] - scores[others])[None, :] - radius @ C.T
    return margins.min(axis=1), completion


def forecast_next(info, y, tau, elapsed, remaining, horizon, freeze_unqueried):
    mask = sum(1 << i for i in remaining)
    sequences = SEQUENCES[mask, horizon]
    margins, times = predicted_margins(info, y, tau, elapsed, sequences, freeze_unqueried)
    feasible = [i for i in range(len(sequences)) if elapsed + times[i] + H <= DEADLINE + TOL]
    if not feasible:
        return None
    certifiable = [i for i in feasible if margins[i] >= -TOL]
    if certifiable:
        idx = min(certifiable, key=lambda i: (float(times[i]), len(sequences[i]), sequences[i]))
        return sequences[idx][0]
    current = certificate(info, y, tau, elapsed + H)[2]
    improving = [i for i in feasible if margins[i] - current > TOL]
    if not improving:
        return None
    idx = min(improving, key=lambda i: (-(margins[i] - current) / times[i], float(times[i]), len(sequences[i]), sequences[i]))
    return sequences[idx][0]


def run_policy(info: PolicyInput, query, method: str):
    """The only environment access available to a policy is query(i, return_time)."""
    y, tau = info.y.copy(), info.tau.copy()
    elapsed = 0.0
    called = []
    remaining = list(range(len(y)))
    if method == "ungated_no_refresh":
        _, a, margin = certificate(info, y, tau, H)
        return dict(executed=True, certified=False, action=a, commit_time=H,
                    calls=0, tool_latency=0.0, final_margin=margin, query_order="")
    fixed_order = None
    while True:
        certified, a, margin = certificate(info, y, tau, elapsed + H)
        if certified:
            return dict(executed=True, certified=True, action=a, commit_time=elapsed + H,
                        calls=len(called), tool_latency=elapsed, final_margin=margin,
                        query_order="-".join(map(str, called)))
        feasible = [i for i in remaining if elapsed + info.latency[i] + H <= DEADLINE + TOL]
        if not feasible:
            break
        if method in ("full_order", "ttl"):
            if fixed_order is None:
                if method == "full_order":
                    all_sequences = SEQUENCES[(1 << len(y)) - 1, 5]
                    margins, _ = predicted_margins(info, y, tau, elapsed, all_sequences)
                    idx = min(range(len(all_sequences)), key=lambda q: (-float(margins[q]), all_sequences[q]))
                    fixed_order = list(all_sequences[idx])
                else:
                    fixed_order = [i for i in remaining if -info.tau[i] > CONFIG["ttl_age_threshold"]]
            candidates = [i for i in fixed_order if i in feasible]
            if not candidates:
                break
            source = candidates[0]
        elif method == "oldest_first":
            source = min(feasible, key=lambda i: (-(elapsed - tau[i]), i))
        elif method == "weighted_age":
            contrasts = np.abs(info.W[a] - info.W).max(axis=0)
            scores = contrasts * info.v * (elapsed - tau) / info.latency
            source = min(feasible, key=lambda i: (-float(scores[i]), i))
        elif method in ("timed_h2", "timed_h3", "static_h2"):
            source = forecast_next(info, y, tau, elapsed, remaining,
                                   3 if method == "timed_h3" else 2, method == "static_h2")
            if source is None:
                break
        else:
            raise ValueError(method)
        elapsed += float(info.latency[source])
        y[source] = query(source, elapsed)
        tau[source] = elapsed
        called.append(source)
        remaining.remove(source)
    return dict(executed=False, certified=False, action=-1, commit_time=elapsed + H,
                calls=len(called), tool_latency=elapsed, final_margin=margin,
                query_order="-".join(map(str, called)))


def evaluate(info, env, policy_result):
    result = dict(policy_result)
    result["invalid_executed"] = False
    result["invalid_certified"] = False
    result["utility_regret"] = math.nan
    if result["executed"]:
        actual_scores = info.b + info.W @ env.state_for_evaluation(result["commit_time"])
        regret = float(actual_scores.max() - actual_scores[result["action"]])
        result["utility_regret"] = max(0.0, regret)
        result["invalid_executed"] = regret > EVAL_TOL
        result["invalid_certified"] = bool(result["certified"] and result["invalid_executed"])
    return result


def exact_checks():
    rng = np.random.default_rng(90421)
    max_error = 0.0
    checks = 0
    for n in range(1, 6):
        corners = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        for _ in range(100):
            W = rng.normal(size=(4, n))
            b = rng.normal(size=4)
            y = rng.normal(size=n)
            r = rng.uniform(0.0, 2.0, size=n)
            a = int(np.argmax(b + W @ y))
            exact = []
            analytic = []
            for c in range(4):
                if c == a:
                    continue
                contrast = W[a] - W[c]
                corner_values = (b[a] - b[c]) + (y + corners * r) @ contrast
                exact.append(float(corner_values.min()))
                analytic.append(float((b[a] - b[c]) + contrast @ y - np.abs(contrast) @ r))
            err = float(np.max(np.abs(np.array(exact) - analytic)))
            max_error = max(max_error, err)
            assert err < 1e-11
            checks += 3
    info = PolicyInput(np.array([[1., 1.], [0., 0.]]), np.zeros(2),
                       np.array([.2, .8]), np.array([-8., -2.]), np.array([.1, .2]), np.array([5., 1.]))
    before = certificate(info, info.y, info.tau, H)
    fmargin, ftime = predicted_margins(info, info.y, info.tau, 0., [(0,)], True)
    tmargin, _ = predicted_margins(info, info.y, info.tau, 0., [(0,)], False)
    commit = float(ftime[0] + H)
    actual_state = np.array([.2, .8 - .2 * (commit + 2.)])
    actual_gap = float(np.sum(actual_state))
    assert before[2] < 0 and fmargin[0] > 0 and tmargin[0] < 0 and actual_gap < 0
    two_step_info = PolicyInput(
        np.array([[1., 1., 0., 0., 0.], [0., 0., 0., 0., 0.]]), np.zeros(2),
        np.array([.325, .325, 0., 0., 0.]), np.array([-4., -4., 0., 0., 0.]),
        np.array([.1, .1, 0., 0., 0.]), np.array([5., 5., 20., 20., 20.]))
    start_margin = certificate(two_step_info, two_step_info.y, two_step_info.tau, H)[2]
    joint_margins, _ = predicted_margins(two_step_info, two_step_info.y, two_step_info.tau,
                                       0., [(0,), (1,), (0, 1), (1, 0)])
    first = forecast_next(two_step_info, two_step_info.y, two_step_info.tau, 0., list(range(5)), 2, False)
    assert joint_margins[0] < start_margin and joint_margins[1] < start_margin
    assert joint_margins[2] > 0 and joint_margins[3] > 0 and first == 0
    return {
        "corner_comparisons": checks, "maximum_absolute_error": max_error,
        "two_step_check": {"initial_margin": start_margin,
                           "one_query_margins": joint_margins[:2].tolist(),
                           "two_query_margins": joint_margins[2:].tolist(),
                           "selected_first_query": first,
                           "purpose": "A two-query improvement is not discarded when both immediate one-query gains are negative."},
        "two_source_counterexample": {
            "y": info.y.tolist(), "timestamps": info.tau.tolist(), "bounds": info.v.tolist(),
            "W": info.W.tolist(), "b": info.b.tolist(), "query_source": 0,
            "query_latency": 5., "commit_gap": H, "initial_minimum_margin": before[2],
            "frozen_age_forecast_margin": float(fmargin[0]), "correct_forecast_margin": float(tmargin[0]),
            "actual_commit_time": commit, "actual_commit_state": actual_state.tolist(),
            "actual_utility_advantage_of_nominal_action": actual_gap,
            "true_motion": "source 0 constant .2; source 1 .8 - .2*(t+2)",
        },
    }


def metrics(rows):
    n = len(rows)
    executed = sum(r["executed"] for r in rows)
    certified = sum(r["certified"] for r in rows)
    invalid = sum(r["invalid_executed"] for r in rows)
    invalid_certified = sum(r["invalid_certified"] for r in rows)
    calls = sum(r["calls"] for r in rows)
    latency = sum(r["tool_latency"] for r in rows)
    regrets = [r["utility_regret"] for r in rows if r["executed"]]
    return dict(n=n, executed=executed, certified=certified, invalid_executed=invalid,
                invalid_certified=invalid_certified,
                execution_coverage=executed / n, certificate_coverage=certified / n,
                abort_rate=1.0-executed/n,
                invalid_rate_given_execution=invalid/executed if executed else None,
                invalid_rate_given_certificate=invalid_certified/certified if certified else None,
                calls_per_arrival=calls/n, latency_per_arrival=latency/n,
                total_calls_per_certificate=calls/certified if certified else None,
                total_latency_per_certificate=latency/certified if certified else None,
                calls_given_certificate=float(np.mean([r["calls"] for r in rows if r["certified"]])) if certified else None,
                latency_given_certificate=float(np.mean([r["tool_latency"] for r in rows if r["certified"]])) if certified else None,
                regret_given_execution=float(np.mean(regrets)) if regrets else None,
                maximum_regret_given_execution=float(max(regrets)) if regrets else None)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, out):
    grouped = {}
    for r in rows:
        for family in [r["family"], "all"]:
            grouped.setdefault((r["condition"], family, r["method"]), []).append(r)
    summaries = []
    for (condition, family, method), group in grouped.items():
        summaries.append(dict(condition=condition, family=family, method=method, **metrics(group)))
    write_csv(out / "summary.csv", summaries)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False))

    seed_grouped = {}
    for r in rows:
        for family in [r["family"], "all"]:
            seed_grouped.setdefault((r["condition"], family, r["seed"], r["method"]), []).append(r)
    seed_metrics = {key: metrics(group) for key, group in seed_grouped.items()}
    paired = []
    for condition in [c["name"] for c in CONFIG["conditions"]]:
        for family in CONFIG["families"] + ["all"]:
            for baseline in ("static_h2", "weighted_age", "full_order"):
                for measure in ("certificate_coverage", "calls_per_arrival", "latency_per_arrival", "total_calls_per_certificate", "total_latency_per_certificate"):
                    differences = []
                    for seed in CONFIG["seeds"]:
                        p = seed_metrics[condition, family, seed, "timed_h2"][measure]
                        b = seed_metrics[condition, family, seed, baseline][measure]
                        if p is not None and b is not None:
                            differences.append(p-b)
                    values = np.array(differences)
                    mean = float(values.mean())
                    se = float(values.std(ddof=1)/np.sqrt(len(values)))
                    half = float(student_t.ppf(.975, len(values)-1) * se)
                    paired.append(dict(condition=condition, family=family, proposed="timed_h2", baseline=baseline,
                                       metric=measure, seeds=len(values), difference=mean, ci_low=mean-half, ci_high=mean+half))
    write_csv(out / "paired_seed_intervals.csv", paired)
    # A paired survivor analysis is descriptive, not a causal effect estimate.
    # Its purpose is to expose how much a cost difference comes from abstention.
    diagnostics = []
    for condition in [c["name"] for c in CONFIG["conditions"]]:
        p = {(r["family"], r["seed"], r["task"]): r for r in rows
             if r["condition"] == condition and r["method"] == "timed_h2"}
        b = {(r["family"], r["seed"], r["task"]): r for r in rows
             if r["condition"] == condition and r["method"] == "static_h2"}
        shared = [key for key in p if p[key]["certified"] and b[key]["certified"]]
        diagnostics.append({
            "condition": condition, "tasks": len(p),
            "initially_certified": sum(r["certified"] and r["calls"] == 0 for r in p.values()),
            "shared_successes": len(shared),
            "timed_only_successes": sum(p[key]["certified"] and not b[key]["certified"] for key in p),
            "static_only_successes": sum(b[key]["certified"] and not p[key]["certified"] for key in p),
            "timed_calls_on_shared_successes": float(np.mean([p[key]["calls"] for key in shared])),
            "static_calls_on_shared_successes": float(np.mean([b[key]["calls"] for key in shared])),
            "timed_latency_on_shared_successes": float(np.mean([p[key]["tool_latency"] for key in shared])),
            "static_latency_on_shared_successes": float(np.mean([b[key]["tool_latency"] for key in shared])),
            "interpretation": "The shared-success subset is selected by both algorithms' outcomes; it is a descriptive diagnostic. Primary cost metrics include all arrivals and all abstentions.",
        })
    (out / "paired_success_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    return summaries, paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "configuration.json").write_text(json.dumps(CONFIG, indent=2))
    checks = exact_checks()
    (out / "verification.json").write_text(json.dumps(checks, indent=2))
    print("Verification:", json.dumps(checks), flush=True)
    if args.checks_only:
        return
    start = time.monotonic()
    rows = []
    for condition in CONFIG["conditions"]:
        condition_start = time.monotonic()
        for family_index, family in enumerate(CONFIG["families"]):
            for seed in CONFIG["seeds"]:
                for task in range(condition["tasks"]):
                    info, env = generate_task(seed, family_index, task, condition)
                    for method in CONFIG["methods"]:
                        result = evaluate(info, env, run_policy(info, env.value, method))
                        assert result["commit_time"] <= DEADLINE + TOL
                        assert result["calls"] <= CONFIG["sources"]
                        if condition["actual_bound_multiplier"] == 1.0:
                            assert not result["invalid_certified"], (condition, family, seed, task, method, result)
                        rows.append(dict(condition=condition["name"], family=family, seed=seed, task=task,
                                         method=method, **result))
        subset = [r for r in rows if r["condition"] == condition["name"]]
        write_csv(out / (condition["name"] + "_raw.csv"), subset)
        print(f"Completed {condition['name']}: {len(subset)} policy evaluations in {time.monotonic()-condition_start:.1f}s", flush=True)
    write_csv(out / "all_raw.csv", rows)
    summaries, paired = aggregate(rows, out)
    duration = time.monotonic()-start
    provenance = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__, "wall_seconds": duration,
        "tasks": len(rows)//len(CONFIG["methods"]), "policy_evaluations": len(rows),
        "actual_data": "all values in raw CSVs were computed by this run; no external data, model API, or private source was used",
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print("MAIN SUMMARY", flush=True)
    for row in summaries:
        if row["condition"] == "main" and row["family"] == "all":
            print(json.dumps(row), flush=True)
    print("FINISHED", json.dumps(provenance), flush=True)


if __name__ == "__main__":
    main()
