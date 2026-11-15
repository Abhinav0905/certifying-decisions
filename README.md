# Certifying Decisions When Tool Reads Take Time

Code and synthetic results by **Kumar Abhinav**. Release **1.0.0**, 15 November 2026.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24253199.svg)](https://doi.org/10.5281/zenodo.24253199)

This archive studies decisions whose inputs keep changing while serial tool reads take time. Four actions have affine scores over five sources. A shared certificate checks whether the selected action remains optimal within the uncertainty box at execution time. Eight acquisition policies choose which sources to refresh.

The experiment contains **13,500 synthetic tasks and 108,000 policy evaluations**, with 20 seeds, three coefficient families and six conditions. All time values are simulated units. No model API, production service, private dataset or human participant was used.

## Main finding

On 6,000 main tasks, the timed two-read forecast spends less query time than the frozen-age two-read forecast, with a small loss of coverage.

| Metric | Timed two-read | Frozen-age two-read |
| --- | ---: | ---: |
| Certified tasks | 5,469 | 5,496 |
| Certification coverage | 91.15% | 91.60% |
| Calls per arriving task | 0.613833 | 0.763667 |
| Query delay per arriving task | 0.567601 | 0.735528 |
| Total query delay / certified task | 0.622711 | 0.802978 |

Query delay per arrival falls **22.83%** while coverage falls **0.45 percentage points**. Costs include reads on tasks that are later deferred. This is a cost-coverage tradeoff, not a speedup for equal completions.

Both methods certify 5,452 tasks. Within that selected subset, mean delay is 0.373860 versus 0.392308, a descriptive reduction of 4.70%. There are 17 timed-only and 44 frozen-age-only successes. The initial cache certifies 3,936 tasks without a read.

All gated policies use the same execution-time certificate. Their zero invalid accepted actions under valid bounds support the gate's arithmetic, not a unique safety benefit of the timed heuristic. Deliberate bound violations are retained: timed two-read accepts 20 invalid actions out of 1,426 accepted actions at 2x slopes and 162 out of 1,473 at 5x slopes.

## Verification status and portability

The original simulator, result files and figures are frozen. Their checksums match the supplied package. The manuscript is excluded from this release.

The 15 November release review replayed all 108,000 saved policy rows, checked the full task inventory and recomputed certificate margins, regret, selected aggregates and all 360 paired confidence intervals. The primary `timed_h2` / `static_h2` comparison also reproduced in a fresh run with the pinned dependencies.

**The full Linux x86_64 rerun passed.** GitHub Actions reproduced all raw policy rows, summaries, diagnostics, confidence intervals and numeric plot summaries within tolerance, with a maximum accepted numeric difference of 8.89e-16. See [the Linux report](release/verification-linux-x86_64.json) and [the verified workflow run](https://github.com/Abhinav0905/certifying-decisions/actions/runs/43744120026). Use this tested platform for full reproduction.

**The full simulator is not exactly reproducible across the tested numerical environments.** On macOS ARM64 with Apple's Accelerate BLAS, 48 policy rows changed query order in the secondary `full_order` and `timed_h3` baselines. Two `timed_h3` rows changed whether they certified, in opposite directions. The main pooled coverage count stayed the same. Some secondary costs and intervals changed. The original results have not been replaced.

See [the release review](RELEASE_REVIEW.md) and [the machine-readable report](release/verification-macos-arm64.json). The strict checker fails on these differences. An explicit option accepts only drift assigned to those two secondary methods and their summaries; every difference remains in its report. It does not waive the saved-result replay, provenance or primary-comparison checks.

## Reproduce and check

Use Python 3.12.14 and the pinned NumPy, SciPy and Matplotlib versions:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r research/requirements.txt
```

Run the release check without overwriting any archived outputs. It creates a temporary copy, reruns the simulator and plotting code, compares every numeric output file and replays the saved decisions:

```sh
python research/verify_release.py > verification.json
```

Exit code 0 means all checks passed. A strict full-rerun mismatch returns exit code 1. On environments with the documented secondary tie sensitivity, use this explicit reporting mode:

```sh
python research/verify_release.py --allow-secondary-drift > verification.json
```

Its status is `archive_and_primary_passed_secondary_drift` when such differences occur, and `passed` when the rerun matches. Unexplained differences still fail. Floating-point comparisons use absolute and relative tolerances of `1e-10`; row identities and boolean outcomes are exact.

For the original arithmetic replay alone:

```sh
python research/audit_replay.py
```

The original replay prints discrepancies. The added release checker enforces error thresholds and verifies missing or duplicate rows as well.

To regenerate the archived output paths deliberately, work in a disposable copy:

```sh
cd research
python experiments.py --output outputs
python plot_results.py
```

These two commands replace generated files. The original figures can differ in font rendering or PDF metadata across platforms; numeric plot summaries are compared by the release check.

## Study design

| Condition | Tasks | Policy evaluations |
| --- | ---: | ---: |
| Main linear motion | 6,000 | 48,000 |
| Quarter read duration | 1,500 | 12,000 |
| Double read duration | 1,500 | 12,000 |
| Bounded sinusoid | 1,500 | 12,000 |
| Double actual slopes | 1,500 | 12,000 |
| Fivefold actual slopes | 1,500 | 12,000 |

Seeds are 100 through 119. Families are independent-option scores with a shared cancelling source, dense normal coefficients and two-source sparse coefficients. Score rows have unit L1 norm. The execution gap is 0.5, the deadline is 12, and each source can be queried at most once per task. Complete settings are in [configuration.json](research/outputs/configuration.json).

Intervals use the 20 paired seed-level differences and a Student t interval. They do not treat 108,000 policy evaluations as independent trials. No multiple-comparison correction was applied. The main coverage difference has a 95% interval of [-0.6942, -0.2058] percentage points; the delay difference has a 95% interval of [-0.182348, -0.153505] simulated units.

The latency plot uses the same first 25 tasks per seed and family at all three latency multipliers: 1,500 tasks per point. Stress plots report means of seed-level rates; these can differ from pooled rates in `summary.csv`.

## Files

| Path | Contents |
| --- | --- |
| `research/experiments.py` | Frozen generator, policies, exact checks and aggregations |
| `research/plot_results.py` | Figure generation and numeric plot summaries |
| `research/audit_replay.py` | Original separate arithmetic replay using the same generator |
| `research/verify_release.py` | Added release verification and full-rerun comparison |
| `research/outputs/` | Raw policy rows, summaries, paired intervals, configuration and provenance |
| `research/figures/` | Three figures, each as PDF and PNG |
| `research/audit_report.txt` | Historical 13 November review, with its original scope and source hashes |
| `release/` | Current verification evidence and frozen-input checksums |
| `RELEASE_REVIEW.md` | Findings, portability issue and interpretation limits |
| `CITATION.cff`, `CITATION.bib` | Citation metadata for this software/results archive |
| `SHA256SUMS` | Checksums of the published files, excluding the checksum file itself |

Raw-row identity is `(condition, family, seed, task, method)`. `executed` records whether an action is taken; `certified` records whether the common gate accepts it. `tool_latency` includes all paid reads, and `commit_time` adds the fixed execution gap. `query_order` uses zero-based source indices. `action=-1` and a `nan` regret mean the task was deferred. See the CSV headers for the remaining fields.

## Limits and attribution

The guarantee assumes valid source-change bounds, exact observations, truthful sample times and a complete fixed action set. Read values are sampled at completion. The study does not test action feasibility, authorization, reservation or atomic commit. Planning computation is not charged. The heuristic holds cached centers fixed in its forecast and may abstain even when another acquisition policy would succeed.

The review covers code and recorded numerical evidence. It does not assess the manuscript, establish novelty or constitute external peer review. Parameter selection history and preregistration cannot be established from this archive alone. ChatGPT assisted with the research, mathematical checks, code, figures and documentation, as disclosed in the supplied project notes.

## Citation and license

Cite the version-specific archive DOI: **[10.5281/zenodo.24253199](https://doi.org/10.5281/zenodo.24253199)**. This identifies the supporting code and synthetic results, not the conference paper.

Kumar Abhinav. (2026). *Certifying Decisions When Tool Reads Take Time: Code and Synthetic Results* (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.24253199

Python code is licensed under MIT. Original results, figures and documentation are licensed under CC BY 4.0. See [LICENSE.md](LICENSE.md).
