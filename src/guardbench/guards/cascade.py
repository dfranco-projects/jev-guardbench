"""Fast guard first; escalate to a slow guard only when the fast one is unsure or fails."""

from dataclasses import replace

from guardbench.guards.base import Context, Guard, Verdict
from guardbench.tasks import Task

# A System One probability inside this band is treated as unsure: the cascade escalates it, and
# the report counts it as not decided with confidence.
UNSURE_BAND = (0.2, 0.8)


class CascadeGuard:
    def __init__(
        self,
        name: str,
        fast: Guard,
        slow: Guard,
        *,
        low: float = UNSURE_BAND[0],
        high: float = UNSURE_BAND[1],
    ):
        self.name = name
        self.fast = fast
        self.slow = slow
        self.low = low
        self.high = high

    async def check(self, text: str, task: Task, context: Context = None) -> Verdict:
        first = await self.fast.check(text, task, context)
        if first.prob is not None and not (self.low < first.prob < self.high):
            return first
        second = await self.slow.check(text, task, context)
        costs = [c for c in (first.cost_usd, second.cost_usd) if c is not None]
        return replace(
            second,
            prob=None,
            latency_ms=first.latency_ms + second.latency_ms,
            cost_usd=sum(costs) if costs else None,
            escalated=True,
        )
