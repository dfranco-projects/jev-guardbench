# jev-guardbench

Can TypeSafe's hosted [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), a **System One model**, stand in for the **LLM-as-judge** in agent guardrail callbacks? Replacing the judge in `before_model` and `after_model` checks would only make sense if it were much faster with no loss in detection quality.

**Status:** counted run in progress (started 2026-10-02). Results and verdicts will be added here.

## What is compared

| Guard | What it is |
|---|---|
| `jev` | TypeSafe Jev, hosted, pinned to `jev-1.13.0`. Returns a probability. |
| `jev-decomposed` | Jev, each policy split into single-condition questions (secondary arm) |
| `gemini-flash` | `gemini-3.6-flash` judge on Vertex AI, minimal thinking, JSON ALLOW/BLOCK |
| `claude-haiku` | `claude-haiku-4-5` judge, JSON ALLOW/BLOCK |
| `jev-then-gemini` | Cascade: Jev first, Gemini only when Jev's probability is in (0.2, 0.8) |

Every guard gets the same policy text on 7,925 public rows across four checks: prompt injection, harmful requests, harmful responses and PII.

## How to read this repo

1. [HYPOTHESES.md](HYPOTHESES.md): what counts as "much faster" and "no loss in quality", fixed before any counted run. Start with the summary at the top; the amendments below it record every change and why.
2. [METHODOLOGY.md](METHODOLOGY.md): datasets, how labels map to the policies, and how latency and quality are measured.
3. The report (`guardbench report --config ...`): verdicts for each hypothesis first, then the full tables.

## Reproduce

```bash
uv sync
uv run pytest

cp .envrc.example .envrc && $EDITOR .envrc && direnv allow   # credentials

uv run python -m guardbench freeze configs/full.yaml          # downloads data, checks it against the manifest
uv run python -m guardbench run configs/full.yaml             # resumable; failed rows retry on the next run
uv run python examples/adk_latency.py configs/full.yaml \
  --guards jev,gemini-flash,claude-haiku --messages 100 --out results/full/h3.md
uv run python -m guardbench report --results results/full --config configs/full.yaml
```

`configs/smoke.yaml` tests the pipeline for free against a local [Kev](https://github.com/jaredpalmer/kev#quick-start) server, which serves the same API as Jev. Kev is not part of the benchmark. The paid guards in that file only run when named with `--guards`.

## Layout

```
src/guardbench/
  tasks.py        one policy per check, rendered for every guard
  guards/         systemone (Jev), llm_judge (Gemini, Claude), cascade
  data.py         public datasets -> rows, stable dev/test split
  runner.py       async, resumable JSONL results
  metrics.py      F1, FPR, AUROC, ECE, latency percentiles, bootstrap CIs
  hypotheses.py   the pre-registered decision rules
  report.py       markdown report
  adk.py          guards inside google-adk callbacks (H3)
examples/         adk_latency.py, the in-agent latency measurement
configs/          guard definitions and sample sizes
manifests/        the frozen row set: ids, labels and text hashes, no content
```
