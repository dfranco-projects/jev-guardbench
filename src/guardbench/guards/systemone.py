"""TypeSafe System One guard. Serves both hosted Jev and a local Kev server (same API)."""

from time import perf_counter

import httpx2
from typesafe_sdk import AsyncTypeSafeClient, Noul, NoulAnswer, RetryPolicy, TypeSafeError

from guardbench.guards.base import Context, Verdict, elapsed_ms, token_cost
from guardbench.tasks import Task

QUESTION_KEY = "violation"


def build_state(text: str, context: Context) -> str | dict[str, str]:
    return {**context, "text": text} if context else text


def build_question(task: Task) -> Noul:
    return Noul(
        instructions=task.question,
        criteria={"true": task.violation, "false": task.allowed},
    )


class SystemOneGuard:
    def __init__(
        self,
        name: str,
        *,
        model: str = "jev-latest",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 10.0,
        threshold: float = 0.5,
        input_per_mtok: float | None = None,
        output_per_mtok: float | None = None,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = name
        self.threshold = threshold
        self._prices = (input_per_mtok, output_per_mtok)
        # Retries are off so latency is one attempt; re-running the benchmark fills failed rows.
        self._client = AsyncTypeSafeClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            retry=RetryPolicy(max_retries=0),
            transport=transport,
        )

    async def check(self, text: str, task: Task, context: Context = None) -> Verdict:
        start = perf_counter()
        try:
            resp = await self._client.system_one(
                build_state(text, context), {QUESTION_KEY: build_question(task)}
            )
        except TypeSafeError as e:
            return Verdict(
                flagged=None, latency_ms=elapsed_ms(start), error=f"{type(e).__name__}: {e}"
            )
        latency = elapsed_ms(start)
        answer = resp.answers[QUESTION_KEY]
        assert isinstance(answer, NoulAnswer)
        cost = token_cost(resp.usage.input_tokens, resp.usage.output_tokens, *self._prices)
        return Verdict(
            flagged=answer.noul >= self.threshold,
            latency_ms=latency,
            prob=answer.noul,
            cost_usd=cost,
        )
