"""H3: how much latency does each guard add to a real ADK agent turn?

Runs the same user messages through an agent with no guard and with each named guard in its
callbacks, then reports the median added time per turn. With `--agent-model scripted` the
model answers instantly, so the added time is the guard alone.

    uv run python examples/adk_latency.py configs/smoke.yaml --guards kev-0.8b --messages 20
"""

import argparse
import asyncio
import logging
import statistics
import time
from collections.abc import AsyncGenerator
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from guardbench.adk import GuardCallbacks
from guardbench.config import build_guard, load_config
from guardbench.data import SOURCES, sample, split_of


class ScriptedLlm(BaseLlm):
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        text = "Thanks for your message. Here is a short, helpful answer."
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


async def time_turns(agent: LlmAgent, messages: list[str]) -> list[float]:
    runner = InMemoryRunner(agent=agent, app_name="bench")
    times = []
    for message in messages:
        session = await runner.session_service.create_session(app_name="bench", user_id="u")
        start = time.perf_counter()
        async for _ in runner.run_async(
            user_id="u",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=message)]),
        ):
            pass
        times.append((time.perf_counter() - start) * 1000)
    return times


def make_agent(model: str | BaseLlm, callbacks: GuardCallbacks | None) -> LlmAgent:
    hooks = (
        {
            "before_agent_callback": callbacks.before_agent,
            "before_model_callback": callbacks.before_model,
            "after_model_callback": callbacks.after_model,
        }
        if callbacks
        else {}
    )
    return LlmAgent(name="assistant", model=model, instruction="Be brief and helpful.", **hooks)


async def main(args: argparse.Namespace) -> None:
    logging.getLogger("google_adk").setLevel(logging.ERROR)
    cfg = load_config(args.config)
    rows = [r for r in SOURCES["aegis2_prompts"]() if split_of(r.id) == "test"]
    messages = [r.text for r in sample(rows, args.messages, seed=cfg.seed)]
    model: str | BaseLlm = (
        ScriptedLlm(model="scripted") if args.agent_model == "scripted" else args.agent_model
    )

    base = await time_turns(make_agent(model, None), messages)
    print(f"model={args.agent_model} messages={len(messages)}")
    print(f"{'no guard':18} turn p50 {statistics.median(base):8.0f} ms")
    for name in args.guards.split(","):
        callbacks = GuardCallbacks(build_guard(name, cfg.guards))
        guarded = await time_turns(make_agent(model, callbacks), messages)
        added = [g - b for g, b in zip(guarded, base, strict=True)]
        blocked = sum(c.verdict.flagged is True for c in callbacks.checks)
        errors = sum(c.verdict.error is not None for c in callbacks.checks)
        print(
            f"{name:18} turn p50 {statistics.median(guarded):8.0f} ms   "
            f"added p50 {statistics.median(added):8.0f} ms   "
            f"checks {len(callbacks.checks)} flagged {blocked} errors {errors}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--guards", required=True)
    parser.add_argument("--messages", type=int, default=20)
    parser.add_argument("--agent-model", default="scripted")
    asyncio.run(main(parser.parse_args()))
