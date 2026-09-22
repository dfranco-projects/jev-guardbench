from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from guardbench.tasks import Task

Context = Mapping[str, str] | None


@dataclass(frozen=True)
class Verdict:
    """One guard decision. `flagged` is None when the call failed. `model` is the version the
    provider reports, since aliases such as `jev-latest` move between releases."""

    flagged: bool | None
    latency_ms: float
    prob: float | None = None
    cost_usd: float | None = None
    error: str | None = None
    note: str | None = None
    escalated: bool = False
    model: str | None = None


class Guard(Protocol):
    name: str

    async def check(self, text: str, task: Task, context: Context = None) -> Verdict: ...


def token_cost(
    input_tokens: int | None,
    output_tokens: int | None,
    input_per_mtok: float | None,
    output_per_mtok: float | None,
) -> float | None:
    if input_per_mtok is None or output_per_mtok is None:
        return None
    return ((input_tokens or 0) * input_per_mtok + (output_tokens or 0) * output_per_mtok) / 1e6


def elapsed_ms(start: float) -> float:
    return (perf_counter() - start) * 1000
