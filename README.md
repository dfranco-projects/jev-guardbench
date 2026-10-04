# jev-guardbench

AI agents often check every message with a second LLM, a "judge", before and after the main model answers. That check is slow and costs money on every turn.

This repo tests whether TypeSafe's [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), a small model built for quick yes/no judgments, can do that job instead.

## The answer

**Jev is about 3× faster and 10–25× cheaper than the judges. It is as accurate on PII and prompt injection, but misses more harmful content.** So it is not a drop-in replacement, but it is a strong choice for some checks.

![Time per guardrail check: Jev 250 ms, Claude Haiku 672 ms, Gemini 1,189 ms](docs/figures/latency.png)

| | Jev | Gemini 3.6 Flash | Claude Haiku 4.5 |
|---|---|---|---|
| Time per check (median) | **250 ms** | 1,189 ms | 672 ms |
| Time added to an agent turn | **488 ms** | 2,282 ms | 1,554 ms |
| Cost per 1,000 checks | **$0.02–0.03** | $0.21–0.41 | $0.41–0.71 |
| PII (F1) | **0.93** | 0.90 | **0.93** |
| Prompt injection (F1) | **0.61** | **0.61** | 0.49 |
| Harmful request (F1) | 0.75 | **0.79** | 0.78 |
| Harmful response (F1) | 0.67 | 0.61 | **0.77** |

The full story, with the caveats, is in **[RESULTS.md](RESULTS.md)**.

## How it was tested

- **Same job for everyone:** each guard got the same policy text and 7,925 rows from public datasets, across four checks.
- **Rules first:** what counts as "much faster" and "as accurate" was written down before any run ([HYPOTHESES.md](HYPOTHESES.md)). Every later change is recorded there, with the reason.
- **Fair split:** thresholds were tuned on one part of the data and scored on another.
- **Real agent:** the guards also ran inside a google-adk agent, to measure the delay a user would feel.
- **Stress tests:** attacks on the guard itself, repeated runs, long inputs and 8 languages.

The guards:

| Name | What it is |
|---|---|
| `jev` | TypeSafe Jev (`jev-1.13.0`). Returns a probability. |
| `jev-decomposed` | Jev, with each policy split into smaller yes/no questions |
| `gemini-flash` | Gemini 3.6 Flash judge on Vertex AI, minimal thinking, answers ALLOW or BLOCK |
| `claude-haiku` | Claude Haiku 4.5 judge, answers ALLOW or BLOCK |
| `jev-then-gemini` | Jev first; Gemini only when Jev is unsure |

## Where to look

1. [RESULTS.md](RESULTS.md): what we found, in plain words.
2. [HYPOTHESES.md](HYPOTHESES.md): the questions and pass/fail rules, fixed in advance.
3. [METHODOLOGY.md](METHODOLOGY.md): datasets, labels and how everything was measured.
4. [results/full/report.md](results/full/report.md): every table. The rows behind them are in `results/full/` (no dataset text).

## Reproduce

```bash
uv sync
uv run pytest

cp .envrc.example .envrc && $EDITOR .envrc && direnv allow   # credentials

uv run python -m guardbench freeze configs/full.yaml          # downloads data, checks it against the manifest
uv run python -m guardbench run configs/full.yaml             # resumable; failed rows retry on the next run
uv run python -m guardbench repeat configs/full.yaml          # consistency: 3 runs on 200 test rows
uv run python -m guardbench attack configs/full.yaml          # attacks on the guard: 200 violating rows
uv run python examples/adk_latency.py configs/full.yaml \
  --guards jev,gemini-flash,claude-haiku --messages 100 --out results/full/h3.md
uv run python -m guardbench report --results results/full --config configs/full.yaml
uv run --with matplotlib python scripts/figures.py           # the charts
```

The counted run took about 10 hours of compute and cost about $16, mostly the two judges. `configs/smoke.yaml` tests the pipeline for free against a local [Kev](https://github.com/jaredpalmer/kev#quick-start) server, which serves the same API as Jev. Kev is not part of the benchmark.

## Layout

```
src/guardbench/
  tasks.py        one policy per check, rendered for every guard
  guards/         systemone (Jev), llm_judge (Gemini, Claude), cascade
  data.py         public datasets -> rows, stable dev/test split
  runner.py       async, resumable JSONL results
  metrics.py      F1, FPR, AUROC, ECE, latency percentiles, bootstrap CIs
  hypotheses.py   the pre-registered decision rules
  attacks.py      the attack texts, fixed before the attack run
  report.py       markdown report
  adk.py          guards inside google-adk callbacks
examples/         adk_latency.py, the in-agent latency measurement
scripts/          figures.py, the charts in docs/figures
configs/          guard definitions and sample sizes
manifests/        the frozen row set: ids, labels and text hashes, no content
results/full/     the counted run: every scored row, and the report
```
