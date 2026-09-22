"""Run any guard inside google-adk callbacks and record what each check cost.

- before_agent: the user's message, once per invocation, before any model call.
- before_model: tool results about to be read by the model (indirect injection).
- after_model: the model's final text, before the user sees it.
"""

import asyncio
from dataclasses import dataclass, field

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from guardbench.guards.base import Guard, Verdict
from guardbench.tasks import TASKS

BLOCKED_INPUT = "Sorry, I can't help with that request."
BLOCKED_OUTPUT = "Sorry, I can't share that response."


@dataclass(frozen=True)
class Check:
    stage: str
    task: str
    verdict: Verdict


@dataclass
class GuardCallbacks:
    guard: Guard
    input_tasks: tuple[str, ...] = ("prompt_injection", "harmful_request", "pii")
    tool_tasks: tuple[str, ...] = ("prompt_injection",)
    output_tasks: tuple[str, ...] = ("harmful_response", "pii")
    fail_closed: bool = False
    checks: list[Check] = field(default_factory=list)

    async def _screen(self, stage: str, text: str, tasks: tuple[str, ...]) -> bool:
        """Run the stage's checks, batched when the guard supports it; True means block."""
        check_many = getattr(self.guard, "check_many", None)
        if check_many:
            verdicts = await check_many(text, [TASKS[t] for t in tasks])
        else:
            verdicts = await asyncio.gather(*(self.guard.check(text, TASKS[t]) for t in tasks))
        self.checks += [Check(stage, t, v) for t, v in zip(tasks, verdicts, strict=True)]
        return any(v.flagged or (v.flagged is None and self.fail_closed) for v in verdicts)

    async def before_agent(self, callback_context: CallbackContext) -> types.Content | None:
        text = _text(callback_context.user_content)
        if text and await self._screen("before_agent", text, self.input_tasks):
            return types.Content(role="model", parts=[types.Part(text=BLOCKED_INPUT)])
        return None

    async def before_model(
        self, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        latest = llm_request.contents[-1] if llm_request.contents else None
        tool_output = _tool_output(latest)
        if tool_output and await self._screen("before_model", tool_output, self.tool_tasks):
            return LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text=BLOCKED_INPUT)])
            )
        return None

    async def after_model(
        self, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        if llm_response.partial:
            return None
        text = _text(llm_response.content)
        if text and await self._screen("after_model", text, self.output_tasks):
            return LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text=BLOCKED_OUTPUT)])
            )
        return None


def _text(content: types.Content | None) -> str:
    if not content or not content.parts:
        return ""
    return "\n".join(p.text for p in content.parts if p.text and not p.thought)


def _tool_output(content: types.Content | None) -> str:
    if not content or not content.parts:
        return ""
    return "\n".join(
        str(p.function_response.response)
        for p in content.parts
        if p.function_response and p.function_response.response
    )
