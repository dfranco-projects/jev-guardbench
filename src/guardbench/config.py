"""Benchmark config: which guards exist, which rows to score, and how hard to push."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from guardbench.data import SOURCES, Row, sample, split_of
from guardbench.guards.base import Guard
from guardbench.guards.cascade import CascadeGuard
from guardbench.guards.llm_judge import ClaudeJudge, GeminiJudge
from guardbench.guards.systemone import SystemOneGuard


@dataclass
class Config:
    guards: dict[str, dict[str, Any]]
    sources: dict[str, int | None]
    run: list[str]
    concurrency: list[int] = field(default_factory=lambda: [1])
    splits: list[str] = field(default_factory=lambda: ["dev", "test"])
    seed: int = 0
    results_dir: Path = Path("results")
    manifest: Path | None = None


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["results_dir"] = Path(raw.get("results_dir", "results"))
    if raw.get("manifest"):
        raw["manifest"] = Path(raw["manifest"])
    return Config(**raw)


def build_guard(name: str, specs: dict[str, dict[str, Any]]) -> Guard:
    spec = dict(specs[name])
    kind = spec.pop("type")
    if kind == "systemone":
        return SystemOneGuard(name, **spec)
    if kind == "gemini":
        return GeminiJudge(name, **spec)
    if kind == "claude":
        return ClaudeJudge(name, **spec)
    if kind == "cascade":
        fast, slow = build_guard(spec.pop("fast"), specs), build_guard(spec.pop("slow"), specs)
        return CascadeGuard(name, fast, slow, **spec)
    raise ValueError(f"unknown guard type {kind!r} for {name!r}")


def load_rows(cfg: Config) -> list[Row]:
    rows: list[Row] = []
    for source, n in cfg.sources.items():
        picked = [r for r in SOURCES[source]() if split_of(r.id) in cfg.splits]
        rows += sample(picked, n, cfg.seed)
    return rows


def manifest_entry(row: Row) -> dict[str, object]:
    """What identifies a scored row: its id, label and a hash of the exact text and context."""
    content = json.dumps([row.text, dict(row.context or {})], sort_keys=True)
    return {
        "id": row.id,
        "task": row.task,
        "label": row.label,
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
    }


def freeze_or_verify(path: Path, rows: list[Row]) -> bool:
    """Write the manifest on first use; afterwards fail if the rows differ from it.
    Returns True when the manifest was written."""
    entries = [manifest_entry(r) for r in rows]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
        return True
    frozen = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if frozen != entries:
        changed = len({json.dumps(e) for e in entries} ^ {json.dumps(e) for e in frozen})
        raise ValueError(
            f"rows differ from {path} ({len(entries)} loaded, {len(frozen)} frozen, "
            f"{changed} entries differ); the benchmark set must not change after freezing"
        )
    return False
