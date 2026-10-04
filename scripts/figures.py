"""Figures for the README and RESULTS.md, from results/full (test split, concurrency 1).

uv run --with matplotlib python scripts/figures.py
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from guardbench.report import load_all, summarise

OUT = Path("docs/figures")
GUARDS = {"jev": "#2a78d6", "gemini-flash": "#eb6834", "claude-haiku": "#1baf7a"}
NAMES = {"jev": "Jev", "gemini-flash": "Gemini 3.6 Flash", "claude-haiku": "Claude Haiku 4.5"}
TASKS = {
    "pii": "PII",
    "prompt_injection": "Prompt injection",
    "harmful_request": "Harmful request",
    "harmful_response": "Harmful response",
}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    }
)

recs = [r for r in load_all(Path("results/full")) if r["concurrency"] == 1]
by_guard = defaultdict(list)
by_task = defaultdict(list)
for r in recs:
    by_guard[r["guard"]].append(r)
    by_task[(r["task"], r["guard"])].append(r)
OUT.mkdir(parents=True, exist_ok=True)


def tidy(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


# 1. Latency: one bar per guard (p50), p95 as a thin tick.
fig, ax = plt.subplots(figsize=(8, 2.8))
order = list(GUARDS)[::-1]
for y, g in enumerate(order):
    s = summarise(by_guard[g])
    ax.barh(y, s["p50_ms"], height=0.5, color=GUARDS[g])
    ax.plot([s["p95_ms"]] * 2, [y - 0.25, y + 0.25], color=INK, linewidth=2)
    ax.text(
        s["p95_ms"] + 25,
        y,
        f"{s['p50_ms']:.0f} ms  (p95 {s['p95_ms']:.0f})",
        va="center",
        color=INK,
    )
ax.set_yticks(range(len(order)), [NAMES[g] for g in order])
ax.set_xlim(0, 2000)
ax.set_xlabel("Latency per check, ms (bar = median, tick = p95)")
ax.set_title("Time per guardrail check", loc="left", color=INK, fontsize=13, pad=12)
tidy(ax)
fig.tight_layout()
fig.savefig(OUT / "latency.png", dpi=200)

# 2. Quality: F1 per task, one dot per guard, at the default threshold.
fig, ax = plt.subplots(figsize=(8, 3.6))
offsets = dict(zip(GUARDS, (0.18, 0, -0.18), strict=True))
for y, task in enumerate(reversed(TASKS)):
    for g, color in GUARDS.items():
        f1 = summarise(by_task[(task, g)])["f1"]
        ax.scatter(f1, y + offsets[g], s=70, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(f1 + 0.008, y + offsets[g], f"{f1:.2f}", va="center", fontsize=9, color=MUTED)
ax.set_yticks(range(len(TASKS)), list(reversed(TASKS.values())))
ax.set_xlim(0.45, 1.0)
ax.set_xticks(np.arange(0.5, 1.01, 0.1))
ax.set_xlabel("F1 on the test split (higher is better), threshold 0.5")
ax.set_title("Detection quality by check", loc="left", color=INK, fontsize=13, pad=12)
handles = [
    plt.Line2D([], [], marker="o", linestyle="", color=c, markersize=8) for c in GUARDS.values()
]
ax.legend(handles, [NAMES[g] for g in GUARDS], loc="lower right", frameon=False, fontsize=9)
tidy(ax)
fig.tight_layout()
fig.savefig(OUT / "quality.png", dpi=200)
print("wrote", *sorted(p.name for p in OUT.iterdir()))

# 3. One image for sharing: speed on top, accuracy below, each on its own axis.
fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 7.2), gridspec_kw={"height_ratios": [2.6, 4]})
for y, g in enumerate(order):
    s = summarise(by_guard[g])
    top.barh(y, s["p50_ms"], height=0.5, color=GUARDS[g])
    top.text(s["p50_ms"] + 25, y, f"{s['p50_ms']:.0f} ms", va="center", color=INK)
top.set_yticks(range(len(order)), [NAMES[g] for g in order])
top.set_xlim(0, 1400)
top.set_xlabel("Median time per check, ms (lower is better)")
top.set_title("Speed", loc="left", color=INK, fontsize=13, pad=10)
tidy(top)
for y, task in enumerate(reversed(TASKS)):
    for g, color in GUARDS.items():
        f1 = summarise(by_task[(task, g)])["f1"]
        bottom.scatter(
            f1, y + offsets[g], s=70, color=color, edgecolor=SURFACE, linewidth=2, zorder=3
        )
        bottom.text(f1 + 0.008, y + offsets[g], f"{f1:.2f}", va="center", fontsize=9, color=MUTED)
bottom.set_yticks(range(len(TASKS)), list(reversed(TASKS.values())))
bottom.set_xlim(0.45, 1.0)
bottom.set_xticks(np.arange(0.5, 1.01, 0.1))
bottom.set_xlabel("F1 on 5,591 test cases (higher is better)")
bottom.set_title("Accuracy by check", loc="left", color=INK, fontsize=13, pad=10)
bottom.legend(handles, [NAMES[g] for g in GUARDS], loc="lower right", frameon=False, fontsize=9)
tidy(bottom)
fig.suptitle("Jev vs. LLM judges as agent guardrails", x=0.02, ha="left", color=INK, fontsize=15)
fig.tight_layout()
fig.savefig(OUT / "summary.png", dpi=200)
print("wrote summary.png")
