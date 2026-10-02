# jev-guardbench

Can TypeSafe's hosted [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), a **System One model**, stand in for the **LLM-as-judge** in agent guardrail callbacks? Replacing the judge in `before_model` and `after_model` checks would only make sense if it were much faster with no loss in detection quality.

The counted run uses hosted **Jev**, pinned to `jev-1.13.0` ([amendment 2](HYPOTHESES.md#amendment-2-2026-09-27-before-any-counted-run)).

- Hypotheses and decision rules, fixed before any runs: [HYPOTHESES.md](HYPOTHESES.md)
- Data, label mapping and measurement: [METHODOLOGY.md](METHODOLOGY.md)

## Quick start

```bash
uv sync
uv run pytest

# guard credentials (only what you use), loaded by direnv
cp .envrc.example .envrc && $EDITOR .envrc && direnv allow

uv run python -m guardbench freeze configs/full.yaml   # checks rows against manifests/full.jsonl
uv run python -m guardbench run configs/smoke.yaml --guards jev,gemini-flash
uv run python -m guardbench report --results results/smoke --baseline gemini-flash
```

`configs/smoke.yaml` tests the pipeline without spending Jev credit by pointing a `systemone` guard at a local [Kev](https://github.com/jaredpalmer/kev#quick-start) server, which serves the same API. Kev is not part of the benchmark.

## Layout

```
src/guardbench/
  tasks.py        one policy per check, rendered for every guard
  guards/         systemone (Jev), llm_judge (Gemini, Claude), cascade
  data.py         public datasets -> rows, stable dev/test split
  runner.py       async, resumable JSONL results
  metrics.py      F1, FPR, AUROC, ECE, latency percentiles, bootstrap CIs
  report.py       markdown report
configs/          guard definitions and sample sizes
```
