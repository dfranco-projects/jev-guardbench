# Pre-registered hypotheses

Rules from HYPOTHESES.md, test split only. Choices the pre-registration leaves open: the best judge is chosen per task by F1 at concurrency 1; the XSTest over-blocking check compares against that task's best judge; latency is pooled over all tasks; H3 (in-agent latency) comes from `examples/adk_latency.py`.

## H1 latency

H1 (latency, ≥5× faster on p50 and p95 at every level): **Inconclusive**

| conc | System One p50 / p95 ms | faster judge | judge p50 / p95 ms | ratio p50 | ratio p95 |
|---|---|---|---|---|---|
| 1 | 250 / 342 | claude-haiku | 672 / 938 | 2.7× | 2.7× |
| 4 | 245 / 323 | claude-haiku | 676 / 921 | 2.8× | 2.8× |

## H2 quality (non-inferiority, margin 0.02)

H2, `jev` (primary), threshold 0.5: **Not supported** (1/4 tasks pass, 4 needed)

| task | best judge | thr | ΔF1 | 95% CI | XSTest FPR guard / judge | passes |
|---|---|---|---|---|---|---|
| harmful_request | gemini-flash | 0.5 | -0.038 | [-0.059, -0.018] | 0.006 / 0.000 | no |
| harmful_response | claude-haiku | 0.5 | -0.096 | [-0.121, -0.074] | 0.006 / 0.006 | no |
| pii | claude-haiku | 0.5 | 0.003 | [-0.012, 0.020] | 0.006 / 0.006 | yes |
| prompt_injection | gemini-flash | 0.5 | 0.001 | [-0.024, 0.024] | 0.006 / 0.000 | no |

H2, `jev` (primary), dev-tuned threshold: **Not supported** (0/4 tasks pass, 4 needed)

| task | best judge | thr | ΔF1 | 95% CI | XSTest FPR guard / judge | passes |
|---|---|---|---|---|---|---|
| harmful_request | gemini-flash | 0.05 | 0.023 | [-0.000, 0.048] | 0.270 / 0.000 | no |
| harmful_response | claude-haiku | 0.06 | 0.065 | [0.042, 0.086] | 0.270 / 0.006 | no |
| pii | claude-haiku | 0.20 | 0.025 | [0.010, 0.040] | 0.270 / 0.006 | no |
| prompt_injection | gemini-flash | 0.71 | -0.026 | [-0.052, -0.000] | 0.270 / 0.000 | no |

H2, `jev-decomposed` (decomposed, secondary arm), threshold 0.5: **Not supported** (0/4 tasks pass, 4 needed)

| task | best judge | thr | ΔF1 | 95% CI | XSTest FPR guard / judge | passes |
|---|---|---|---|---|---|---|
| harmful_request | gemini-flash | 0.5 | -0.056 | [-0.079, -0.033] | 0.056 / 0.000 | no |
| harmful_response | claude-haiku | 0.5 | -0.122 | [-0.150, -0.098] | 0.056 / 0.006 | no |
| pii | claude-haiku | 0.5 | 0.037 | [0.023, 0.052] | 0.056 / 0.006 | no |
| prompt_injection | gemini-flash | 0.5 | -0.077 | [-0.106, -0.048] | 0.056 / 0.000 | no |

H2, `jev-decomposed` (decomposed, secondary arm), dev-tuned threshold: **Not supported** (0/4 tasks pass, 4 needed)

| task | best judge | thr | ΔF1 | 95% CI | XSTest FPR guard / judge | passes |
|---|---|---|---|---|---|---|
| harmful_request | gemini-flash | 0.06 | -0.001 | [-0.026, 0.027] | 0.438 / 0.000 | no |
| harmful_response | claude-haiku | 0.06 | 0.050 | [0.026, 0.074] | 0.438 / 0.006 | no |
| pii | claude-haiku | 0.30 | 0.038 | [0.023, 0.054] | 0.438 / 0.006 | no |
| prompt_injection | gemini-flash | 0.58 | -0.059 | [-0.090, -0.032] | 0.438 / 0.000 | no |

## H3 in-agent latency

Agent model `scripted`, 100 messages, unguarded turn p50 1 ms.

| guard | turn p50 ms | added p50 ms | checks | flagged | errors |
|---|---|---|---|---|---|
| jev | 489 | 488 | 416 | 42 | 0 |
| gemini-flash | 2288 | 2282 | 416 | 46 | 5 |
| claude-haiku | 1555 | 1554 | 422 | 63 | 0 |

- H3, `gemini-flash` → `jev`: added latency cut by 79%: **Supported**
- H3, `claude-haiku` → `jev`: added latency cut by 69%: **Supported**

## H4 cascade

H4 (cascade non-inferior to the best judge, p50 within 1.5× of System One): **Not supported**

| task | best judge | thr | ΔF1 | 95% CI | XSTest FPR guard / judge | passes |
|---|---|---|---|---|---|---|
| harmful_request | gemini-flash | 0.5 | -0.031 | [-0.048, -0.015] | 0.000 / 0.000 | no |
| harmful_response | claude-haiku | 0.5 | -0.152 | [-0.180, -0.127] | 0.000 / 0.006 | no |
| pii | claude-haiku | 0.5 | -0.018 | [-0.036, 0.001] | 0.000 / 0.006 | no |
| prompt_injection | gemini-flash | 0.5 | 0.005 | [-0.014, 0.022] | 0.000 / 0.000 | yes |

- concurrency 1: cascade p50 is 1.06× the System One p50
- concurrency 4: cascade p50 is 1.02× the System One p50

## Measured risks

### By input length (all tasks, concurrency 1)

| guard | length | n | F1 | p50 ms |
|---|---|---|---|---|
| jev | ≤2k | 5256 | 0.760 | 250 |
| jev | 2k–8k | 303 | 0.438 | 248 |
| jev | >8k | 32 | 0.308 | 267 |
| gemini-flash | ≤2k | 5256 | 0.743 | 1189 |
| gemini-flash | 2k–8k | 303 | 0.446 | 1173 |
| gemini-flash | >8k | 32 | 0.364 | 1230 |
| claude-haiku | ≤2k | 5256 | 0.768 | 667 |
| claude-haiku | 2k–8k | 303 | 0.391 | 719 |
| claude-haiku | >8k | 32 | 0.242 | 816 |

### PII by language (concurrency 1)

| guard | lang | n | F1 |
|---|---|---|---|
| jev | de | 140 | 0.950 |
| jev | en | 267 | 0.892 |
| jev | es | 159 | 0.974 |
| jev | fr | 220 | 0.927 |
| jev | hi | 75 | 0.935 |
| jev | it | 101 | 0.941 |
| jev | nl | 53 | 0.936 |
| jev | te | 42 | 0.947 |
| gemini-flash | de | 140 | 0.912 |
| gemini-flash | en | 267 | 0.866 |
| gemini-flash | es | 159 | 0.947 |
| gemini-flash | fr | 220 | 0.894 |
| gemini-flash | hi | 75 | 0.889 |
| gemini-flash | it | 101 | 0.953 |
| gemini-flash | nl | 53 | 0.837 |
| gemini-flash | te | 42 | 0.857 |
| claude-haiku | de | 140 | 0.944 |
| claude-haiku | en | 267 | 0.898 |
| claude-haiku | es | 159 | 0.930 |
| claude-haiku | fr | 220 | 0.942 |
| claude-haiku | hi | 75 | 0.950 |
| claude-haiku | it | 101 | 0.955 |
| claude-haiku | nl | 53 | 0.898 |
| claude-haiku | te | 42 | 0.923 |

### Attacks on the guard

Violating test rows, scored unchanged and with each attack text appended (`src/guardbench/attacks.py`).

| guard | rows | clean recall | direct recall (Δ) | spoof recall (Δ) |
|---|---|---|---|---|
| claude-haiku | 200 | 0.775 | 0.845 (+0.070) | 0.735 (-0.040) |
| gemini-flash | 200 | 0.650 | 0.670 (+0.020) | 0.665 (+0.015) |
| jev | 200 | 0.665 | 0.675 (+0.010) | 0.655 (-0.010) |
| jev-decomposed | 200 | 0.655 | 0.660 (+0.005) | 0.645 (-0.010) |

### Determinism

3 runs on the same test rows, concurrency 1.

| guard | rows | flip rate | mean prob range | max prob range |
|---|---|---|---|---|
| claude-haiku | 200 | 0.035 | – | – |
| gemini-flash | 200 | 0.035 | – | – |
| jev | 200 | 0.010 | 0.011 | 0.090 |
| jev-decomposed | 200 | 0.005 | 0.013 | 0.090 |

## Provenance

| guard | model versions | rows | errors | first / last row (UTC) |
|---|---|---|---|---|
| claude-haiku | claude-haiku-4-5-20251001 | 15850 | 0 | 2026-10-02T23:55 / 2026-10-03T01:56 |
| gemini-flash | gemini-3.6-flash | 15850 | 5 | 2026-10-02T20:20 / 2026-10-03T05:50 |
| jev | jev-1.13.0 | 15850 | 0 | 2026-10-02T18:46 / 2026-10-02T19:36 |
| jev-decomposed | jev-1.13.0 | 15850 | 0 | 2026-10-02T19:36 / 2026-10-02T20:20 |
| jev-then-gemini | gemini-3.6-flash, jev-1.13.0 | 15850 | 0 | 2026-10-03T01:56 / 2026-10-03T05:50 |


# guardbench report

## harmful_request

| guard | conc | n | err | prec | rec | F1 | F1 95% CI | F1 tuned | FPR | AUROC | ECE | cover. | F1 sure | p50 ms | p95 ms | $/1k | escal. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-haiku | 1 | 1401 | 0.000 | 0.929 | 0.674 | 0.781 | [0.753, 0.805] | – | 0.055 | – | – | – | – | 647 | 884 | 0.421 | 0.000 |
| gemini-flash | 1 | 1401 | 0.000 | 0.948 | 0.676 | 0.789 | [0.763, 0.813] | – | 0.040 | – | – | – | – | 1207 | 1912 | 0.220 | 0.000 |
| jev | 1 | 1401 | 0.000 | 0.944 | 0.624 | 0.751 | [0.723, 0.776] | 0.812 | 0.040 | 0.889 | 0.174 | 0.811 | 0.766 | 253 | 347 | 0.017 | 0.000 |
| jev-decomposed | 1 | 1401 | 0.000 | 0.896 | 0.621 | 0.734 | [0.704, 0.761] | 0.789 | 0.077 | 0.876 | 0.144 | 0.806 | 0.787 | 245 | 335 | 0.020 | 0.000 |
| jev-then-gemini | 1 | 1401 | 0.000 | 0.956 | 0.628 | 0.758 | [0.729, 0.783] | – | 0.031 | – | – | – | – | 274 | 1749 | 0.058 | 0.186 |
| claude-haiku | 4 | 1401 | 0.000 | 0.929 | 0.674 | 0.781 | [0.755, 0.806] | – | 0.055 | – | – | – | – | 672 | 905 | 0.421 | 0.000 |
| gemini-flash | 4 | 1400 | 0.001 | 0.947 | 0.687 | 0.796 | [0.771, 0.820] | – | 0.041 | – | – | – | – | 1233 | 2062 | 0.219 | 0.000 |
| jev | 4 | 1401 | 0.000 | 0.945 | 0.621 | 0.750 | [0.722, 0.773] | 0.816 | 0.038 | 0.889 | 0.173 | 0.812 | 0.768 | 250 | 400 | 0.017 | 0.000 |
| jev-decomposed | 4 | 1401 | 0.000 | 0.893 | 0.625 | 0.736 | [0.707, 0.763] | 0.787 | 0.080 | 0.875 | 0.145 | 0.804 | 0.787 | 249 | 299 | 0.020 | 0.000 |
| jev-then-gemini | 4 | 1401 | 0.000 | 0.950 | 0.633 | 0.760 | [0.733, 0.784] | – | 0.035 | – | – | – | – | 248 | 1594 | 0.058 | 0.185 |

By source (concurrency 1):

| source | guard | F1 | FPR |
|---|---|---|---|
| aegis2 | claude-haiku | 0.764 | 0.072 |
| aegis2 | gemini-flash | 0.754 | 0.054 |
| aegis2 | jev | 0.713 | 0.052 |
| aegis2 | jev-decomposed | 0.693 | 0.084 |
| aegis2 | jev-then-gemini | 0.718 | 0.042 |
| xstest | claude-haiku | 0.852 | 0.006 |
| xstest | gemini-flash | 0.922 | 0.000 |
| xstest | jev | 0.893 | 0.006 |
| xstest | jev-decomposed | 0.880 | 0.056 |
| xstest | jev-then-gemini | 0.906 | 0.000 |

## harmful_response

| guard | conc | n | err | prec | rec | F1 | F1 95% CI | F1 tuned | FPR | AUROC | ECE | cover. | F1 sure | p50 ms | p95 ms | $/1k | escal. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-haiku | 1 | 1618 | 0.000 | 0.937 | 0.651 | 0.768 | [0.744, 0.792] | – | 0.053 | – | – | – | – | 648 | 899 | 0.488 | 0.000 |
| gemini-flash | 1 | 1618 | 0.000 | 0.948 | 0.452 | 0.612 | [0.581, 0.643] | – | 0.030 | – | – | – | – | 1217 | 1788 | 0.267 | 0.000 |
| jev | 1 | 1618 | 0.000 | 0.949 | 0.520 | 0.672 | [0.642, 0.698] | 0.832 | 0.034 | 0.890 | 0.236 | 0.739 | 0.685 | 258 | 348 | 0.021 | 0.000 |
| jev-decomposed | 1 | 1618 | 0.000 | 0.928 | 0.495 | 0.646 | [0.616, 0.674] | 0.818 | 0.047 | 0.870 | 0.238 | 0.747 | 0.661 | 259 | 352 | 0.026 | 0.000 |
| jev-then-gemini | 1 | 1618 | 0.000 | 0.959 | 0.453 | 0.616 | [0.584, 0.645] | – | 0.023 | – | – | – | – | 275 | 1749 | 0.091 | 0.260 |
| claude-haiku | 4 | 1618 | 0.000 | 0.935 | 0.653 | 0.769 | [0.747, 0.793] | – | 0.055 | – | – | – | – | 658 | 904 | 0.488 | 0.000 |
| gemini-flash | 4 | 1617 | 0.001 | 0.950 | 0.450 | 0.611 | [0.580, 0.642] | – | 0.029 | – | – | – | – | 1219 | 1949 | 0.266 | 0.000 |
| jev | 4 | 1618 | 0.000 | 0.948 | 0.519 | 0.671 | [0.641, 0.697] | 0.831 | 0.034 | 0.889 | 0.236 | 0.739 | 0.683 | 242 | 314 | 0.021 | 0.000 |
| jev-decomposed | 4 | 1618 | 0.000 | 0.924 | 0.492 | 0.642 | [0.612, 0.671] | 0.817 | 0.049 | 0.870 | 0.238 | 0.751 | 0.664 | 260 | 316 | 0.026 | 0.000 |
| jev-then-gemini | 4 | 1618 | 0.000 | 0.958 | 0.468 | 0.629 | [0.600, 0.657] | – | 0.025 | – | – | – | – | 250 | 1630 | 0.090 | 0.257 |

By source (concurrency 1):

| source | guard | F1 | FPR |
|---|---|---|---|
| aegis2 | claude-haiku | 0.733 | 0.085 |
| aegis2 | gemini-flash | 0.511 | 0.018 |
| aegis2 | jev | 0.577 | 0.039 |
| aegis2 | jev-decomposed | 0.516 | 0.057 |
| aegis2 | jev-then-gemini | 0.481 | 0.018 |
| beavertails | claude-haiku | 0.785 | 0.033 |
| beavertails | gemini-flash | 0.656 | 0.038 |
| beavertails | jev | 0.714 | 0.031 |
| beavertails | jev-decomposed | 0.703 | 0.040 |
| beavertails | jev-then-gemini | 0.673 | 0.027 |

## pii

| guard | conc | n | err | prec | rec | F1 | F1 95% CI | F1 tuned | FPR | AUROC | ECE | cover. | F1 sure | p50 ms | p95 ms | $/1k | escal. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-haiku | 1 | 1057 | 0.000 | 0.957 | 0.902 | 0.929 | [0.912, 0.945] | – | 0.039 | – | – | – | – | 708 | 954 | 0.413 | 0.000 |
| gemini-flash | 1 | 1057 | 0.000 | 0.993 | 0.822 | 0.900 | [0.879, 0.920] | – | 0.006 | – | – | – | – | 1223 | 1497 | 0.208 | 0.000 |
| jev | 1 | 1057 | 0.000 | 0.987 | 0.883 | 0.932 | [0.915, 0.948] | 0.954 | 0.011 | 0.981 | 0.066 | 0.901 | 0.965 | 245 | 323 | 0.017 | 0.000 |
| jev-decomposed | 1 | 1057 | 0.000 | 0.986 | 0.946 | 0.966 | [0.954, 0.976] | 0.967 | 0.013 | 0.993 | 0.047 | 0.939 | 0.976 | 267 | 362 | 0.016 | 0.000 |
| jev-then-gemini | 1 | 1057 | 0.000 | 0.991 | 0.843 | 0.911 | [0.891, 0.929] | – | 0.007 | – | – | – | – | 244 | 1504 | 0.037 | 0.097 |
| claude-haiku | 4 | 1057 | 0.000 | 0.955 | 0.902 | 0.928 | [0.912, 0.944] | – | 0.041 | – | – | – | – | 677 | 924 | 0.413 | 0.000 |
| gemini-flash | 4 | 1057 | 0.000 | 0.995 | 0.830 | 0.905 | [0.885, 0.924] | – | 0.004 | – | – | – | – | 1230 | 1921 | 0.209 | 0.000 |
| jev | 4 | 1057 | 0.000 | 0.985 | 0.887 | 0.934 | [0.917, 0.948] | 0.949 | 0.013 | 0.982 | 0.065 | 0.903 | 0.966 | 245 | 326 | 0.017 | 0.000 |
| jev-decomposed | 4 | 1057 | 0.000 | 0.986 | 0.939 | 0.962 | [0.949, 0.973] | 0.969 | 0.013 | 0.993 | 0.047 | 0.939 | 0.980 | 242 | 297 | 0.016 | 0.000 |
| jev-then-gemini | 4 | 1057 | 0.000 | 0.993 | 0.845 | 0.913 | [0.894, 0.932] | – | 0.006 | – | – | – | – | 245 | 1480 | 0.038 | 0.101 |

## prompt_injection

| guard | conc | n | err | prec | rec | F1 | F1 95% CI | F1 tuned | FPR | AUROC | ECE | cover. | F1 sure | p50 ms | p95 ms | $/1k | escal. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-haiku | 1 | 1515 | 0.000 | 0.334 | 0.917 | 0.489 | [0.452, 0.524] | – | 0.431 | – | – | – | – | 716 | 988 | 0.710 | 0.000 |
| gemini-flash | 1 | 1515 | 0.000 | 0.489 | 0.799 | 0.607 | [0.567, 0.645] | – | 0.197 | – | – | – | – | 1127 | 1743 | 0.413 | 0.000 |
| jev | 1 | 1515 | 0.000 | 0.476 | 0.844 | 0.608 | [0.567, 0.647] | 0.581 | 0.219 | 0.862 | 0.198 | 0.746 | 0.641 | 248 | 333 | 0.029 | 0.000 |
| jev-decomposed | 1 | 1515 | 0.000 | 0.380 | 0.875 | 0.530 | [0.493, 0.565] | 0.548 | 0.337 | 0.821 | 0.303 | 0.465 | 0.586 | 248 | 327 | 0.031 | 0.000 |
| jev-then-gemini | 1 | 1515 | 0.000 | 0.487 | 0.824 | 0.612 | [0.572, 0.652] | – | 0.205 | – | – | – | – | 296 | 1797 | 0.143 | 0.257 |
| claude-haiku | 4 | 1515 | 0.000 | 0.333 | 0.913 | 0.488 | [0.450, 0.522] | – | 0.431 | – | – | – | – | 700 | 931 | 0.710 | 0.000 |
| gemini-flash | 4 | 1515 | 0.000 | 0.481 | 0.806 | 0.603 | [0.562, 0.640] | – | 0.205 | – | – | – | – | 1198 | 1783 | 0.412 | 0.000 |
| jev | 4 | 1515 | 0.000 | 0.474 | 0.841 | 0.606 | [0.563, 0.646] | 0.588 | 0.220 | 0.861 | 0.199 | 0.741 | 0.638 | 244 | 303 | 0.029 | 0.000 |
| jev-decomposed | 4 | 1515 | 0.000 | 0.384 | 0.882 | 0.535 | [0.496, 0.572] | 0.533 | 0.334 | 0.822 | 0.304 | 0.462 | 0.591 | 250 | 310 | 0.031 | 0.000 |
| jev-then-gemini | 4 | 1515 | 0.000 | 0.493 | 0.837 | 0.621 | [0.581, 0.661] | – | 0.203 | – | – | – | – | 255 | 1616 | 0.142 | 0.256 |

By source (concurrency 1):

| source | guard | F1 | FPR |
|---|---|---|---|
| deepset | claude-haiku | 0.733 | 0.405 |
| deepset | gemini-flash | 0.857 | 0.004 |
| deepset | jev | 0.901 | 0.000 |
| deepset | jev-decomposed | 0.701 | 0.423 |
| deepset | jev-then-gemini | 0.877 | 0.004 |
| in_the_wild | claude-haiku | 0.291 | 0.439 |
| in_the_wild | gemini-flash | 0.402 | 0.253 |
| in_the_wild | jev | 0.376 | 0.284 |
| in_the_wild | jev-decomposed | 0.355 | 0.312 |
| in_the_wild | jev-then-gemini | 0.396 | 0.264 |
