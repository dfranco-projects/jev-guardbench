"""Benchmark config: which guards exist, which rows to score, and how hard to push."""

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


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text())
    raw["results_dir"] = Path(raw.get("results_dir", "results"))
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
