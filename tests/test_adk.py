"""The callbacks inside a real ADK runner, with a scripted model instead of a live LLM."""

from collections.abc import AsyncGenerator

import pytest
from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from guardbench.adk import BLOCKED_INPUT, BLOCKED_OUTPUT, GuardCallbacks
from guardbench.guards.base import Verdict
from guardbench.tasks import Task


class ScriptedLlm(BaseLlm):
    reply: str = "Here is a safe answer."
    calls: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.calls += 1
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=self.reply)]))


class KeywordGuard:
    """Flags any text containing a trigger word, for every task."""

    name = "keyword"

    def __init__(self, trigger: str):
        self.trigger = trigger
        self.seen: list[tuple[str, str]] = []

    async def check(self, text: str, task: Task, context=None) -> Verdict:
        self.seen.append((task.name, text))
        return Verdict(flagged=self.trigger in text, latency_ms=1.0, prob=None)


async def run_turn(callbacks: GuardCallbacks, llm: ScriptedLlm, message: str) -> str:
    agent = LlmAgent(
        name="assistant",
        model=llm,
        instruction="Be helpful.",
        before_agent_callback=callbacks.before_agent,
        before_model_callback=callbacks.before_model,
        after_model_callback=callbacks.after_model,
    )
    runner = InMemoryRunner(agent=agent, app_name="test")
    session = await runner.session_service.create_session(app_name="test", user_id="u")
    texts = []
    async for event in runner.run_async(
        user_id="u",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=message)]),
    ):
        if event.content and event.content.parts:
            texts += [p.text for p in event.content.parts if p.text]
    return "\n".join(texts)


async def test_clean_turn_passes_through_input_and_output_checks():
    guard = KeywordGuard("BAD")
    callbacks = GuardCallbacks(guard)
    llm = ScriptedLlm(model="scripted")

    assert await run_turn(callbacks, llm, "hello") == "Here is a safe answer."
    assert {c.stage for c in callbacks.checks} == {"before_agent", "after_model"}
    assert {t for t, _ in guard.seen} == {
        "prompt_injection",
        "harmful_request",
        "pii",
        "harmful_response",
    }


async def test_flagged_input_skips_the_model():
    llm = ScriptedLlm(model="scripted")
    assert await run_turn(GuardCallbacks(KeywordGuard("BAD")), llm, "BAD request") == BLOCKED_INPUT
    assert llm.calls == 0


async def test_flagged_output_is_replaced():
    llm = ScriptedLlm(model="scripted", reply="leaking BAD stuff")
    assert await run_turn(GuardCallbacks(KeywordGuard("BAD")), llm, "hi") == BLOCKED_OUTPUT


@pytest.mark.parametrize(
    ("fail_closed", "expected"), [(False, "Here is a safe answer."), (True, BLOCKED_INPUT)]
)
async def test_guard_failure_policy(fail_closed, expected):
    class Broken:
        name = "broken"

        async def check(self, text, task, context=None):
            return Verdict(flagged=None, latency_ms=1.0, error="timeout")

    llm = ScriptedLlm(model="scripted")
    assert await run_turn(GuardCallbacks(Broken(), fail_closed=fail_closed), llm, "hi") == expected


class ToolCallingLlm(BaseLlm):
    """Calls `fetch_page` first, then answers."""

    calls: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.calls += 1
        part = (
            types.Part(function_call=types.FunctionCall(name="fetch_page", args={}))
            if self.calls == 1
            else types.Part(text="Summary of the page.")
        )
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


@pytest.mark.parametrize(
    ("page", "expected"), [("weather is sunny", "Summary of the page."), ("BAD", BLOCKED_INPUT)]
)
async def test_tool_output_is_screened_before_the_model_reads_it(page, expected):
    def fetch_page() -> dict:
        """Fetch the page."""
        return {"content": page}

    guard = KeywordGuard("BAD")
    callbacks = GuardCallbacks(guard, input_tasks=())
    llm = ToolCallingLlm(model="scripted")
    agent = LlmAgent(
        name="assistant",
        model=llm,
        tools=[fetch_page],
        before_model_callback=callbacks.before_model,
    )
    runner = InMemoryRunner(agent=agent, app_name="test")
    session = await runner.session_service.create_session(app_name="test", user_id="u")
    texts = []
    async for event in runner.run_async(
        user_id="u",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="summarise the page")]),
    ):
        if event.content and event.content.parts:
            texts += [p.text for p in event.content.parts if p.text]
    assert texts[-1] == expected
    assert [c.stage for c in callbacks.checks] == ["before_model"]
    assert page in guard.seen[0][1]
