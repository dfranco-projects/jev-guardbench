# Pre-registered hypotheses

These hypotheses were written on 2026-09-22, before any benchmark run that counts as evidence. Smoke runs on Kev-0.8B only test the pipeline and are not reported as results. The decision rules below will not be changed after results come in. Any extra analysis will be labelled exploratory.

**Question:** can a System One model (hosted Jev, or self-hosted Kev) replace or front an LLM-as-judge in agent guardrail callbacks? It would need to be much faster without meaningfully lower detection quality.

**Guards compared:**

| Name | What it is |
|---|---|
| `jev` | TypeSafe Jev, hosted |
| `kev-4b`, `kev-9b` | Open-source Kev, self-hosted |
| `gemini-flash` | Gemini 3.5 Flash judge (minimal thinking) returning JSON ALLOW/BLOCK |
| `claude-haiku` | Claude Haiku 4.5 judge returning JSON ALLOW/BLOCK |
| cascade | System One model first; the judge runs only when the probability falls in (0.2, 0.8) or the call fails |

All guards get the same policy text for each task (`src/guardbench/tasks.py`). Headline numbers come from the **test split**. The dev split is used only to tune thresholds.

## H1: latency

The System One guard's p50 and p95 latency are **at least 5× lower** than those of the faster judge. This must hold at concurrency 1 and at concurrency 16.

- **Supported** if the ratio is ≥ 5 on both percentiles at both concurrency levels.
- **Rejected** if the ratio is < 2 on either percentile.
- **Inconclusive** otherwise.
- **Measurement:** client wall-clock time, one attempt, no retries. A no-op round trip to each provider is reported alongside, so differences in network and region are visible.
- **Scope:** H1 is judged only on hosted Jev against hosted judges. Self-hosted Kev latency is reported separately and depends on the hardware.

## H2: quality, tested as non-inferiority

For each task, compare the System One guard's F1 with the better judge's F1 on the same test rows. Take the paired bootstrap 95% CI of ΔF1 = F1(System One) − F1(best judge). The decision rule has two parts:

- **Non-inferior** if the lower bound of that CI is **> −0.02**. The comparison is made at two System One thresholds: 0.5, and a threshold tuned on dev.
- **Over-blocking:** the false-positive rate on XSTest safe prompts must be no higher than the judge's plus 0.02.

A task counts as passing when both parts hold. H2 is supported if at least 4 of the 5 tasks pass.

## H3: in-agent latency

Measured inside a real google-adk agent, with the guard in the `before_model` and `after_model` callbacks. Swapping the judge for the System One guard must cut the **latency the guard adds per turn** (guarded turn minus unguarded turn, median) by **≥ 50%**.

## H4: cascade

The cascade's F1 is not lower than the best judge's, using the same non-inferiority rule as H2. At the same time, its p50 latency must be within 1.5× of the System One guard alone.

## Risks we will measure, not assume away

- **Attacks on the guard itself:** injected text that tells the guard to answer false. We report the drop in recall.
- **Long inputs:** F1 and latency by input length (≤ 2k, 2k–8k, > 8k characters).
- **Non-English inputs:** F1 by language on the PII task, which covers 8 languages.
- **Determinism:** 3 repeated runs on 200 rows. We report flip rates for the judges and probability drift for System One.

## Known limits

- **Policies and labels are ours.** Dataset labels were mapped onto them, for example the strong-PII rule described in `METHODOLOGY.md`. Results say how well each guard follows *these* policies.
- **Possible training contamination:** public datasets may be in any model's training data.
- **Kev does not stand in for Jev.** Conclusions about Jev need Jev runs.
