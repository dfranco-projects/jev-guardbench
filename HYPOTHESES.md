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

## Amendment 1 (2026-09-22, before any counted run)

TypeSafe paused new signups, so no Jev key can be obtained, and H1 as first written could not be decided. The only results so far are Kev-0.8B smoke runs, which are not evidence. This amendment was made before any counted run and replaces the matching rules above.

- **Primary System One guard: `kev-9b`**, self-hosted on a rented GPU. The GPU model and region are named in the report. It is reached through a forwarded port, so it pays a network hop like the hosted judges do.
- **H1, H2 and H4 are judged on `kev-9b`.** The H1 "Scope" line above no longer applies. Hosted `jev` (pinned to `jev-1.13.0`) is a secondary arm, reported with the same rules when a key is available. Results for one model do not carry over to the other.
- **Secondary arm, `*-decomposed`:** each policy is asked as single-condition Nouls (`Task.parts`), and a row is flagged by the highest part probability. TypeSafe's documentation recommends this style. It gets the same H2 rule, reported separately. The primary arm keeps the shared single-question policy.
- **Framing for System One:** the question names the judged field (`content`) when context is present, and says to treat it as data. This mirrors the data instruction the judges already get.
- Every result records the model version that answered, as reported by the provider.
- **H2 task count:** `prompt_leakage` has no data yet. Until it does, H2 is supported only if **all 4 tasks with data pass**. The "4 of 5" rule applies once `prompt_leakage` has data frozen in the manifest.
- **PII policy narrowed to what the labels measure.** ai4privacy tags dates without saying whether they are birth dates, and has no credential spans, so "date of birth" and "credentials" were dropped from the `pii` policy. The policy now lists the same identifiers as the label rule in `METHODOLOGY.md`.
- **Frozen set:** dataset commits are pinned (`data.REVISIONS`). The counted rows are listed in `manifests/full.jsonl` (id, task, label, hash of text and context). A run whose rows differ from the manifest stops with an error.

## Amendment 2 (2026-09-27, before any counted run)

A Jev key is now available. This replaces the Kev-related rules in amendment 1. Nothing has been run for the record yet: the only results are still Kev-0.8B smoke runs.

- **The System One guard is hosted `jev`, pinned to `jev-1.13.0`.** H1–H4 are judged on it, as in the original scope. Kev is dropped from the hypotheses. It remains usable for smoke tests, since it serves the same API.
- **Still in force from amendment 1:** the decomposed arm (now `jev-decomposed`), the System One framing, recording the model version, the H2 task count, the narrowed PII policy, and the frozen set.
- **Confidence:** a Noul answer has no separate confidence value; the probability is the confidence ([TypeSafe docs](https://docs.typesafe.ai/primitives/noul)). Besides the threshold metrics, the report gives **coverage**, the share of rows with a probability outside (0.2, 0.8), and **F1 on those rows**. This is the same band the cascade escalates. Both are reported, not tested.
- **H1 concurrency is 1 and 4, not 1 and 16.** Jev allows 1,200 requests per minute ([models](https://docs.typesafe.ai/models)). Sixteen requests in flight at about 0.4 s each would exceed that, so the 16-in-flight latency would measure rate limiting, not Jev. Four in flight stays under the limit. The judges use the same levels.

## Amendment 3 (2026-10-02, before any counted run)

Nothing has been run for the record yet.

- **The Gemini judge is `gemini-3.8-flash`, not Gemini 3.5 Flash.** 3.8 Flash is the newest Flash model and costs half as much ($0.75 input, $3.75 output per million tokens until 2026-12-31, [pricing](https://ai.google.dev/gemini-api/docs/pricing)). A newer judge is the harder baseline for H2. Minimal thinking and the JSON output are unchanged, and the guard keeps the name `gemini-flash`.
- The counted run is planned before 2027-01-01, when that price doubles. Cost is reported at the price in force on the run date.
