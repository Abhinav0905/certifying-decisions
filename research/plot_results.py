#!/usr/bin/env python3
"""Render manuscript figures from the saved raw rows, with seed-level CIs."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "lines.linewidth": 1.4,
})

METHODS = ["full_order", "weighted_age", "static_h2", "timed_h2"]
LABELS = {"full_order": "Full order", "weighted_age": "Weighted age",
          "static_h2": "Static two-read", "timed_h2": "Timed two-read",
          "ungated_no_refresh": "No refresh, ungated"}
COLORS = {"full_order": "#777777", "weighted_age": "#278067",
          "static_h2": "#BB5A24", "timed_h2": "#235A9F",
          "ungated_no_refresh": "#1C1C1C"}
MARKERS = {"full_order": "s", "weighted_age": "^", "static_h2": "D", "timed_h2": "o", "ungated_no_refresh": "x"}

rows = list(csv.DictReader((OUT / "all_raw.csv").open()))
for r in rows:
    r["seed"] = int(r["seed"])
    r["task"] = int(r["task"])
    for key in ("executed", "certified", "invalid_executed"):
        r[key] = r[key] == "True"
    for key in ("tool_latency", "calls"):
        r[key] = float(r[key])


def estimate(subset, metric):
    groups = defaultdict(list)
    for r in subset:
        groups[r["seed"]].append(r)
    vals = []
    for seed in sorted(groups):
        group = groups[seed]
        if metric == "coverage":
            vals.append(100 * np.mean([r["certified"] for r in group]))
        elif metric == "latency":
            vals.append(np.mean([r["tool_latency"] for r in group]))
        elif metric == "invalid":
            vals.append(100 * sum(r["invalid_executed"] for r in group) / sum(r["executed"] for r in group))
        else:
            raise ValueError(metric)
    vals = np.array(vals)
    mean = float(vals.mean())
    half = float(student_t.ppf(.975, len(vals)-1) * vals.std(ddof=1) / np.sqrt(len(vals)))
    return mean, half


def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight", pad_inches=.025)
    fig.savefig(FIGS / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=.025)
    plt.close(fig)


# Balanced latency sensitivity: the same first 25 tasks per seed/family at all x.
conditions = ["latency_quarter", "main", "latency_double"]
x = np.array([.25, 1., 2.])
fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.8))
fig.subplots_adjust(left=.12, right=.985, bottom=.32, top=.89, wspace=.55)
latency_summary = []
for method in METHODS:
    for ax, metric in zip(axes, ["coverage", "latency"]):
        stats = []
        for condition in conditions:
            subset = [r for r in rows if r["condition"] == condition and r["method"] == method and r["task"] < 25]
            mean, half = estimate(subset, metric)
            stats.append((mean, half))
            latency_summary.append({"condition": condition, "method": method, "metric": metric,
                                    "tasks": len(subset), "seed_mean": mean,
                                    "ci_low": mean-half, "ci_high": mean+half})
        ax.errorbar(x, [s[0] for s in stats], yerr=[s[1] for s in stats],
                    color=COLORS[method], marker=MARKERS[method], markersize=3.5,
                    capsize=2.0, label=LABELS[method])
        ax.grid(axis="y", alpha=.18)
        ax.set_xticks(x, ["0.25", "1", "2"])
        ax.set_xlabel("Latency multiplier")
axes[0].set_title("(a) Coverage")
axes[0].set_ylabel("Certified decisions (%)")
axes[0].set_ylim(82, 100)
axes[1].set_title("(b) Query cost")
axes[1].set_ylabel("Latency per arriving task")
axes[1].set_ylim(bottom=0)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(.5, -.005), columnspacing=1., handlelength=1.5)
save(fig, "latency_sensitivity")
with (OUT / "balanced_latency_summary.csv").open("w", newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(latency_summary[0])); w.writeheader(); w.writerows(latency_summary)


# Drift stress: same latent draws, with recalculated cache values/intercepts.
fig, ax = plt.subplots(figsize=(4.8, 2.8))
fig.subplots_adjust(left=.14, right=.97, bottom=.38, top=.92)
stress_summary = []
for method in ["ungated_no_refresh"] + METHODS:
    stats = []
    for condition in ["main", "bound_double", "bound_fivefold"]:
        subset = [r for r in rows if r["condition"] == condition and r["method"] == method and r["task"] < 25]
        mean, half = estimate(subset, "invalid")
        stats.append((mean, half))
        stress_summary.append({"condition": condition, "method": method, "tasks": len(subset),
                               "seed_mean_percent": mean, "ci_low": mean-half, "ci_high": mean+half})
    ax.errorbar([1,2,5], [s[0] for s in stats], yerr=[s[1] for s in stats],
                color=COLORS[method], marker=MARKERS[method], markersize=4,
                capsize=2., label=LABELS[method])
ax.set_xlabel("Motion multiplier")
ax.set_ylabel("Invalid actions (%)")
ax.set_xticks([1,2,5]); ax.set_ylim(bottom=-.3)
ax.grid(axis="y", alpha=.18)
fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(.5,-.015), columnspacing=1., handlelength=1.5)
save(fig,"bound_misspecification")
with (OUT / "stress_plot_summary.csv").open("w", newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(stress_summary[0])); w.writeheader(); w.writerows(stress_summary)


# Exact two-source example, not a sampled result.
fig, ax = plt.subplots(figsize=(4.8, 2.4))
fig.subplots_adjust(left=.13, right=.97, bottom=.23, top=.93)
duration=np.linspace(0,5,101)
ax.plot(duration,np.full_like(duration,.45),label="Frozen-age forecast",color=COLORS["static_h2"],linestyle="--")
ax.plot(duration,.45-.2*duration,label="Correct lower bound",color=COLORS["timed_h2"])
ax.plot(duration,.5-.2*duration,label="Actual utility gap",color="#333333",linestyle=":")
ax.axhline(0,color="#888888",linewidth=.7)
ax.set_xlabel("Source 1 query duration")
ax.set_ylabel("Action 1 advantage")
ax.grid(axis="y",alpha=.18)
ax.legend(frameon=False,loc="lower left",handlelength=2.)
save(fig,"two_source_example")


# Family-level values support an optional table without expanding the manuscript.
family_summary=[]
for family in ["independent_option","dense_shared","sparse_mixed"]:
    for method in METHODS:
        subset=[r for r in rows if r["condition"]=="main" and r["family"]==family and r["method"]==method]
        for metric in ["coverage","latency"]:
            mean,half=estimate(subset,metric)
            family_summary.append({"family":family,"method":method,"metric":metric,"tasks":len(subset),
                                   "seed_mean":mean,"ci_low":mean-half,"ci_high":mean+half})
with (OUT / "family_plot_summary.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(family_summary[0]));w.writeheader();w.writerows(family_summary)

print("Saved",", ".join(p.name for p in sorted(FIGS.iterdir())))
