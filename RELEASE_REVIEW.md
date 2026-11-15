# Release review: 15 November 2026

## Scope

This review checks the supplied code and synthetic results for release 1.0.0. The manuscript was not reviewed or edited and is excluded from the public artifact. The original experiment source, outputs, figures and historical audit report are unchanged.

The certificate computes each candidate-versus-competitor affine score difference at the cached center, then subtracts the sum of absolute coefficient contrasts multiplied by source uncertainty radii. For a Cartesian box, this is the exact minimum of that affine difference. Shared coefficients cancel before uncertainty is charged. The simulator uses timestamps at read completion and evaluates the gate at elapsed time plus the execution gap.

The policies' implemented forecast functions receive cached observations, coefficients, declared rates and durations. They do not receive the environment's future true state. This is a code-level information boundary, not a security sandbox for arbitrary Python callbacks.

## Checks and evidence

- All 29 supplied research files match their original SHA-256 checksums.
- All 108,000 raw policy rows are present with unique identities, across 13,500 tasks and eight methods. Condition-specific files match the corresponding combined rows.
- Saved query schedules, paid times, deadlines, certificates and regret were replayed. The largest margin error was 1.78e-15 and the largest regret error was 2.00e-15.
- Selected pooled and family-level aggregates matched exactly. All 360 paired interval rows matched within 4.45e-16.
- A complete simulator rerun and regeneration of all six figure files completed under Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0 and Matplotlib 3.10.8 on macOS ARM64.
- All primary two-read rows and summaries reproduced. The common-success diagnostic and its 5,452 shared certificates reproduced too.
- Four deliberate invalid comparisons were rejected by the added verifier: changed numeric value, changed boolean, changed row count and a missing-value mismatch.

The source hash is `aa6142fafd5c4ac77b8fea9d2a4169e269f190de94d53bc1119fd4bfda1db444`. The full evidence is in `release/verification-macos-arm64.json`. Its explicit status is `archive_and_primary_passed_secondary_drift`, not an exact full-rerun pass.

## Finding: numerical ties affect secondary policies

The original implementation uses raw floating-point values to rank predicted margins and cumulative durations. A mathematically tied value can differ at machine precision across numerical backends. That difference takes precedence over the intended lexicographic tie break.

The macOS rerun changed query orders in 29 `full_order` rows and 19 `timed_h3` rows, 48 out of 108,000 evaluations. Five `timed_h3` rows and one `full_order` row changed call count. Two main `timed_h3` tasks changed certification outcome in opposite directions, so the main pooled coverage count remained unchanged. Some secondary delay summaries changed. Eight paired interval rows comparing against `full_order` and one balanced latency plot summary also differed.

For example, in the independent-option family at seed 101, task 59, several full-read permutations share the same computed best margin on this Mac. The archived first query was source 4; the rerun's lexicographic choice starts with source 0. At seed 107, task 73, several certifiable three-read permutations have the same cumulative duration on this Mac. Their first read determines the subsequent replanning path.

The numerical backend is a plausible cause supported by the tied rankings. The original preparation environment's operating system and BLAS build were not recorded, so the exact original backend has not been identified. Pinned package versions alone do not settle this issue.

Release 1.0.0 preserves the source and results supporting the current paper. It does not silently replace secondary results or change ranking rules. A future version should specify tolerance-aware ranking or a canonical accumulation rule, rerun every condition and update any affected paper tables. That would be a distinct scientific artifact.

## Finding: the original replay did not enforce every discrepancy

`audit_replay.py` accumulated maximum errors for margins, regret, selected summaries and intervals, then printed them without asserting numerical thresholds. It also lacked an exact expected-row inventory. A successful process exit was therefore insufficient evidence of agreement.

`verify_release.py` adds explicit error bounds, inventory and flag checks, comparison of condition files, provenance verification and a complete temporary rerun. Strict mode fails on any output difference outside numeric tolerance. The optional `--allow-secondary-drift` mode reports every mismatch and accepts only rows attributed to the two documented secondary policies or their summaries. Primary-method, diagnostic, configuration and saved-replay failures remain fatal.

## Interpretation

The supported primary result is lower query cost with slightly lower coverage: 22.83% less query delay per arrival and a 0.45-percentage-point loss of certification. The 4.70% reduction on common successes is descriptive because the subset is selected by both policies' outcomes.

Correctness belongs to the shared certificate under its stated assumptions. The acquisition forecast is a heuristic. A forecast's predicted margin does not certify a future observation. The `static_h2` ablation freezes unqueried ages only during planning; it uses the same correct execution-time gate.

The generated initial gap is controlled by construction. The study does not estimate production failure rates. Bound-misspecification conditions reuse latent draws but regenerate cached values and target offsets, so they are not identical decisions with only a changed physical slope. All reported times exclude policy computation.

The separate replay shares the generator and environment with the simulator. It is useful arithmetic verification but cannot expose every shared modelling error. The finite-box checks are numerical checks of small examples, not a substitute for a formal proof. The review makes no claim about conference acceptance, novelty or external peer review.
