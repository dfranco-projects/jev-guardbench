"""Run guards over rows with bounded concurrency. Results append to JSONL and runs resume:
rows that already have a successful result are skipped, failed rows are retried."""

import asyncio
import json
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from guardbench.data import Row, split_of
from guardbench.guards.base import Guard
from guardbench.tasks import TASKS


def results_path(results_dir: Path, guard: str, concurrency: int) -> Path:
    return results_dir / f"{guard}__c{concurrency}.jsonl"


def read_results(path: Path) -> list[dict[str, Any]]:
    """Latest record per row id: a successful retry replaces an earlier failure."""
    if not path.exists():
        return []
    latest: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            latest[rec["id"]] = rec
    return list(latest.values())


async def run_guard(
    guard: Guard, rows: Iterable[Row], out: Path, concurrency: int = 1
) -> tuple[int, int]:
    """Returns (rows run now, rows that failed)."""
    done = {r["id"] for r in read_results(out) if r["error"] is None}
    todo = [r for r in rows if r.id not in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    failed = 0

    with out.open("a", encoding="utf-8") as f:

        async def one(row: Row) -> None:
            nonlocal failed
            async with sem:
                verdict = await guard.check(row.text, TASKS[row.task], row.context)
            failed += verdict.error is not None
            rec = {
                "id": row.id,
                "task": row.task,
                "source": row.source,
                "split": split_of(row.id),
                "label": row.label,
                "lang": row.lang,
                "chars": len(row.text),
                "guard": guard.name,
                "concurrency": concurrency,
                "ts": time.time(),
                **asdict(verdict),
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()

        await asyncio.gather(*(one(r) for r in todo))
    return len(todo), failed
