"""Turn results/*.jsonl into a markdown report. Headline numbers use the test split only;
the dev split is used just to tune a per-guard threshold."""

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from guardbench import metrics
from guardbench.runner import read_results

Rec = dict[str, Any]


def load_all(results_dir: Path) -> list[Rec]:
    return [rec for path in sorted(results_dir.glob("*.jsonl")) for rec in read_results(path)]


def summarise(recs: list[Rec]) -> dict[str, Any]:
    """Metrics for one guard x concurrency x task group (dev and test records mixed in)."""
    test = [r for r in recs if r["split"] == "test"]
    ok = [r for r in test if r["error"] is None]
    labels = [r["label"] for r in ok]
    preds = [r["flagged"] for r in ok]
    out: dict[str, Any] = {
        "n": len(test),
        "error_rate": 1 - len(ok) / len(test) if test else float("nan"),
        **metrics.confusion(labels, preds),
    }
    if ok:
        out["f1_ci"] = metrics.bootstrap_ci(
            lambda idx: metrics.f1([labels[i] for i in idx], [preds[i] for i in idx]), len(ok)
        )
        out |= metrics.latency_summary([r["latency_ms"] for r in ok])
        costs = [r["cost_usd"] for r in ok]
        out["usd_per_1k"] = 1000 * float(np.mean(costs)) if None not in costs else None
        out["escalation_rate"] = float(np.mean([r["escalated"] for r in ok]))
        out["provider_blocks"] = sum(r["note"] == "provider_block" for r in ok)
    if ok and all(r["prob"] is not None for r in ok):
        probs = [r["prob"] for r in ok]
        out["auroc"] = metrics.auroc(labels, probs)
        out["ece"] = metrics.ece(labels, probs)
        dev = [r for r in recs if r["split"] == "dev" and r["error"] is None]
        if dev:
            t = metrics.best_f1_threshold([r["label"] for r in dev], [r["prob"] for r in dev])
            out["tuned_threshold"] = t
            out["f1_tuned"] = metrics.f1(labels, [p >= t for p in probs])
    return out


def paired_f1_diff(a: list[Rec], b: list[Rec]) -> tuple[int, float, tuple[float, float]]:
    """F1(a) - F1(b) on test rows both guards answered, with a paired bootstrap interval."""
    b_by_id = {r["id"]: r for r in b if r["split"] == "test" and r["error"] is None}
    pairs = [(r, b_by_id[r["id"]]) for r in a if r["id"] in b_by_id and r["error"] is None]
    if not pairs:
        return 0, float("nan"), (float("nan"), float("nan"))
    y = [ra["label"] for ra, _ in pairs]
    pa = [ra["flagged"] for ra, _ in pairs]
    pb = [rb["flagged"] for _, rb in pairs]

    def diff(idx: Iterable[int]) -> float:
        idx = list(idx)
        ys = [y[i] for i in idx]
        return metrics.f1(ys, [pa[i] for i in idx]) - metrics.f1(ys, [pb[i] for i in idx])

    return len(pairs), diff(range(len(pairs))), metrics.bootstrap_ci(diff, len(pairs))


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return "–" if np.isnan(v) else f"{v:.{digits}f}"
    if isinstance(v, tuple):
        return f"[{_fmt(v[0])}, {_fmt(v[1])}]"
    return str(v)


def render(recs: list[Rec], baseline: str | None = None) -> str:
    groups: dict[tuple[str, int, str], list[Rec]] = defaultdict(list)
    for r in recs:
        groups[(r["task"], r["concurrency"], r["guard"])].append(r)

    lines = ["# guardbench report", ""]
    cols = [
        ("guard", None),
        ("conc", None),
        ("n", "n"),
        ("err", "error_rate"),
        ("prec", "precision"),
        ("rec", "recall"),
        ("F1", "f1"),
        ("F1 95% CI", "f1_ci"),
        ("F1 tuned", "f1_tuned"),
        ("FPR", "fpr"),
        ("AUROC", "auroc"),
        ("ECE", "ece"),
        ("p50 ms", "p50_ms"),
        ("p95 ms", "p95_ms"),
        ("$/1k", "usd_per_1k"),
        ("escal.", "escalation_rate"),
    ]
    for task in sorted({k[0] for k in groups}):
        lines += [f"## {task}", ""]
        lines += ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
        for (t, conc, guard), group in sorted(groups.items()):
            if t != task:
                continue
            s = summarise(group)
            cells = [guard, str(conc)] + [_fmt(s.get(key)) for _, key in cols[2:]]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

        by_source: dict[tuple[str, str], list[Rec]] = defaultdict(list)
        for (t, conc, guard), group in groups.items():
            if t == task and conc == 1:
                for r in group:
                    by_source[(r["source"], guard)].append(r)
        if len({s for s, _ in by_source}) > 1:
            lines += [
                "By source (concurrency 1):",
                "",
                "| source | guard | F1 | FPR |",
                "|---|---|---|---|",
            ]
            for (source, guard), group in sorted(by_source.items()):
                s = summarise(group)
                lines.append(f"| {source} | {guard} | {_fmt(s.get('f1'))} | {_fmt(s.get('fpr'))} |")
            lines.append("")

        if baseline and (task, 1, baseline) in groups:
            lines += [
                f"Paired F1 difference vs `{baseline}` (concurrency 1):",
                "",
                "| guard | pairs | ΔF1 | 95% CI |",
                "|---|---|---|---|",
            ]
            for (t, conc, guard), group in sorted(groups.items()):
                if t == task and conc == 1 and guard != baseline:
                    n, d, ci = paired_f1_diff(group, groups[(task, 1, baseline)])
                    lines.append(f"| {guard} | {n} | {_fmt(d)} | {_fmt(ci)} |")
            lines.append("")
    return "\n".join(lines)
