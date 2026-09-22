# Methodology

## Tasks and data

| Task | Callback side | Sources (test/dev by stable hash, 70/30) |
|---|---|---|
| `prompt_injection` | before model | [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) (Apache-2.0), [in-the-wild jailbreak prompts](https://huggingface.co/datasets/TrustAIRLab/in-the-wild-jailbreak-prompts) (MIT) |
| `harmful_request` | before model | [Aegis 2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) prompts (CC-BY-4.0), [XSTest](https://huggingface.co/datasets/Paul/XSTest) (CC-BY-4.0) |
| `harmful_response` | after model | Aegis 2.0 responses, [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) 30k test (CC-BY-NC-4.0) |
| `pii` | both | [ai4privacy open-pii-masking-500k](https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy), validation split (see dataset license) |
| `prompt_leakage` | after model | synthetic (planned) |

This repository contains no dataset content. Rows are downloaded from Hugging Face at run time.

### Label mapping decisions

- **deepset:** the corpus was built for a news publisher's reader assistant, and requests off that purpose ("Generate SQL code…") are labelled as injections. Every row therefore carries the assistant's purpose as context, and all guards see it.
- **ai4privacy:** a row is a `pii` violation only if it contains a strong identifier: email, phone, ID card, passport, driver's licence, card, social security or tax number, or a street address. Rows whose only spans are weak (a first name, city, date, age or title) are negatives. Those spans don't identify a person on their own, which matches the policy text. They are hard negatives because they come from the same distribution as the positives.
- **Response tasks** carry the user's message as context.

## Guards

Every guard receives the same policy text (`src/guardbench/tasks.py`):

- **System One (Jev/Kev):** the policy is sent as one `noul` question (`instructions` = question, `criteria.true` / `criteria.false` = violation / allowed). Structured state is sent as `{...context, "text": ...}`.
- **LLM judges:** the policy goes in the system prompt, with content in `<content>` tags and an instruction to treat it as data. The output is constrained to JSON `{"decision": "ALLOW" | "BLOCK"}` through structured output.
- **Provider refusals:** when a provider's own safety filter stops the judge, the row counts as BLOCK (`note=provider_block`), which is what a production callback would do.

## Measurement

- **Latency:** client-side wall clock for one attempt. SDK retries are disabled. Failed rows are recorded with `error` and retried on the next run, and only the latest record per row counts.
- **Errors:** excluded from quality metrics and reported as an error rate.
- **Threshold tuning:** thresholds are tuned on dev (best F1) and applied unchanged to test.
- **Intervals:** 95% bootstrap intervals with 1,000 resamples. Guard-vs-guard differences use paired resamples of the same rows.
