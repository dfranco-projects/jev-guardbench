# jev-guardbench

Can a **System One model** stand in for the **LLM-as-judge** in agent guardrail callbacks? Candidates are TypeSafe's hosted [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) and the open-source [Kev](https://github.com/jaredpalmer/kev), which uses the same API. Replacing the judge in `before_model` and `after_model` checks would only make sense if it were much faster with no loss in detection quality.

The counted run uses hosted **Jev**, pinned to `jev-1.13.0` ([amendment 2](HYPOTHESES.md#amendment-2-2026-09-27-before-any-counted-run)).

- Hypotheses and decision rules, fixed before any runs: [HYPOTHESES.md](HYPOTHESES.md)
- Data, label mapping and measurement: [METHODOLOGY.md](METHODOLOGY.md)

## Quick start

```bash
uv sync
uv run pytest

# guard credentials (only what you use)
export TYPESAFE_API_KEY=...        # hosted Jev
export ANTHROPIC_API_KEY=...       # Claude judge
export GOOGLE_API_KEY=...          # Gemini judge (or GOOGLE_GENAI_USE_VERTEXAI=true + ADC)

uv run python -m guardbench freeze configs/full.yaml   # checks rows against manifests/full.jsonl
uv run python -m guardbench run configs/smoke.yaml --guards jev,gemini-flash
uv run python -m guardbench report --results results/smoke --baseline gemini-flash
```

To run Kev locally, follow the [Kev quick start](https://github.com/jaredpalmer/kev#quick-start) and point a `systemone` guard at it with `base_url: http://127.0.0.1:8009`. `configs/smoke.yaml` does this with Kev-0.8B to test the pipeline without spending Jev credit.

## Layout

```
src/guardbench/
  tasks.py        one policy per check, rendered for every guard
  guards/         systemone (Jev/Kev), llm_judge (Gemini, Claude), cascade
  data.py         public datasets -> rows, stable dev/test split
  runner.py       async, resumable JSONL results
  metrics.py      F1, FPR, AUROC, ECE, latency percentiles, bootstrap CIs
  report.py       markdown report
configs/          guard definitions and sample sizes
```
