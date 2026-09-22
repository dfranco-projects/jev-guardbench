from collections.abc import Collection
from pathlib import Path

from guardbench.data import Row
from guardbench.guards.base import Verdict
from guardbench.report import paired_f1_diff, render, summarise
from guardbench.runner import read_results, run_guard
from guardbench.tasks import Task

ROWS = [Row(f"r{i}", "pii", "src", f"text {i}", label=i % 2 == 0) for i in range(20)]


class ScriptedGuard:
    """Flags even rows; fails the first time it sees any id in `flaky`."""

    def __init__(self, name: str, flaky: Collection[str] = (), prob: bool = True):
        self.name = name
        self.flaky = set(flaky)
        self.prob = prob

    async def check(self, text: str, task: Task, context=None) -> Verdict:
        row_id = f"r{text.split()[1]}"
        if row_id in self.flaky:
            self.flaky.discard(row_id)
            return Verdict(flagged=None, latency_ms=5, error="boom")
        even = int(text.split()[1]) % 2 == 0
        return Verdict(
            flagged=even, latency_ms=10, prob=(0.9 if even else 0.1) if self.prob else None
        )


async def test_rerun_retries_only_failed_rows(tmp_path: Path):
    out = tmp_path / "g.jsonl"
    guard = ScriptedGuard("g", flaky={"r1", "r2"})

    assert await run_guard(guard, ROWS, out, concurrency=4) == (20, 2)
    assert await run_guard(guard, ROWS, out, concurrency=4) == (2, 0)
    assert await run_guard(guard, ROWS, out, concurrency=4) == (0, 0)

    recs = read_results(out)
    assert len(recs) == 20
    assert all(r["error"] is None for r in recs)


async def test_report_scores_a_perfect_guard(tmp_path: Path):
    out = tmp_path / "g__c1.jsonl"
    await run_guard(ScriptedGuard("g"), ROWS, out)
    s = summarise(read_results(out))
    assert s["n"] > 0
    assert s["f1"] == 1.0
    assert s["auroc"] == 1.0
    assert s["error_rate"] == 0.0


async def test_paired_diff_is_zero_for_identical_guards(tmp_path: Path):
    await run_guard(ScriptedGuard("a"), ROWS, tmp_path / "a__c1.jsonl")
    await run_guard(ScriptedGuard("b", prob=False), ROWS, tmp_path / "b__c1.jsonl")
    a, b = read_results(tmp_path / "a__c1.jsonl"), read_results(tmp_path / "b__c1.jsonl")
    n, diff, _ = paired_f1_diff(a, b)
    assert n > 0
    assert diff == 0.0
    text = render(a + b, baseline="b")
    assert "## pii" in text
    assert "Paired F1 difference vs `b`" in text
