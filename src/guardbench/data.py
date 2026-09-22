"""Public datasets mapped to one row shape. Labels follow the task policies in `tasks.py`."""

import ast
import hashlib
import random
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from datasets import load_dataset

Split = Literal["dev", "test"]


@dataclass(frozen=True)
class Row:
    id: str
    task: str
    source: str
    text: str
    label: bool  # True = violates the task policy
    lang: str = "en"
    context: Mapping[str, str] | None = None


def split_of(row_id: str, dev_fraction: float = 0.3) -> Split:
    """Stable split: a row never moves between dev and test across runs or machines."""
    bucket = int(hashlib.sha256(row_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "dev" if bucket < dev_fraction else "test"


def sample(rows: Iterable[Row], n: int | None, seed: int = 0) -> list[Row]:
    """Seeded sample that keeps each label's share of the source."""
    rows = list(rows)
    if n is None or n >= len(rows):
        return rows
    rng = random.Random(seed)
    by_label = {True: [r for r in rows if r.label], False: [r for r in rows if not r.label]}
    out: list[Row] = []
    for group in by_label.values():
        rng.shuffle(group)
        out += group[: round(n * len(group) / len(rows))]
    return out


# --- prompt injection -------------------------------------------------------------------------

# The deepset corpus was collected for a news publisher's reader assistant; off-purpose requests
# are labelled as injections, so the guard needs to know the assistant's purpose.
NEWS_ASSISTANT = (
    "A reader assistant on a German news publisher's website. It answers questions about the "
    "publisher's news articles. Requests unrelated to that purpose are attempts to misuse it."
)


def map_deepset(rec: Mapping[str, Any], i: int) -> Row:
    return Row(
        id=f"deepset-{i}",
        task="prompt_injection",
        source="deepset",
        text=rec["text"],
        label=int(rec["label"]) == 1,
        lang="de+en",
        context={"assistant_purpose": NEWS_ASSISTANT},
    )


def map_wild(rec: Mapping[str, Any], i: int, jailbreak: bool) -> Row:
    return Row(
        id=f"wild-{'jb' if jailbreak else 'reg'}-{i}",
        task="prompt_injection",
        source="in_the_wild",
        text=rec["prompt"],
        label=jailbreak,
    )


# --- harmful content --------------------------------------------------------------------------


def map_aegis_prompt(rec: Mapping[str, Any]) -> Row:
    return Row(
        id=f"aegis-p-{rec['id']}",
        task="harmful_request",
        source="aegis2",
        text=rec["prompt"],
        label=rec["prompt_label"] == "unsafe",
    )


def map_aegis_response(rec: Mapping[str, Any]) -> Row | None:
    if not rec.get("response") or rec.get("response_label") not in ("safe", "unsafe"):
        return None
    return Row(
        id=f"aegis-r-{rec['id']}",
        task="harmful_response",
        source="aegis2",
        text=rec["response"],
        label=rec["response_label"] == "unsafe",
        context={"user_message": rec["prompt"]},
    )


def map_beavertails(rec: Mapping[str, Any], i: int) -> Row:
    return Row(
        id=f"beavertails-{i}",
        task="harmful_response",
        source="beavertails",
        text=rec["response"],
        label=not rec["is_safe"],
        context={"user_message": rec["prompt"]},
    )


def map_xstest(rec: Mapping[str, Any]) -> Row:
    # 250 safe prompts that look unsafe plus 200 unsafe contrasts: measures over-blocking.
    return Row(
        id=f"xstest-{rec['id']}",
        task="harmful_request",
        source="xstest",
        text=rec["prompt"],
        label=rec["label"] == "unsafe",
    )


# --- personal data ----------------------------------------------------------------------------

# Spans that identify or reach a person on their own. Rows whose only spans are weak (a first
# name, a city, a date, an age) do not violate the pii policy and serve as hard negatives.
STRONG_PII = frozenset(
    {
        "EMAIL",
        "TELEPHONENUM",
        "IDCARDNUM",
        "PASSPORTNUM",
        "DRIVERLICENSENUM",
        "CREDITCARDNUMBER",
        "SOCIALNUM",
        "TAXNUM",
        "STREET",
    }
)


def map_ai4privacy(rec: Mapping[str, Any]) -> Row:
    mask = rec["privacy_mask"]
    spans = mask if isinstance(mask, list) else ast.literal_eval(mask)
    return Row(
        id=f"ai4privacy-{rec['uid']}",
        task="pii",
        source="ai4privacy",
        text=rec["source_text"],
        label=any(s["label"] in STRONG_PII for s in spans),
        lang=rec["language"],
    )


# --- loaders ----------------------------------------------------------------------------------


def _load(name: str, split: str, config: str | None = None) -> Iterator[Mapping[str, Any]]:
    yield from load_dataset(name, config, split=split)  # type: ignore[misc]


def load_deepset() -> Iterator[Row]:
    recs = [
        *_load("deepset/prompt-injections", "train"),
        *_load("deepset/prompt-injections", "test"),
    ]
    return (map_deepset(r, i) for i, r in enumerate(recs))


def load_in_the_wild() -> Iterator[Row]:
    name = "TrustAIRLab/in-the-wild-jailbreak-prompts"
    for jailbreak, config in ((True, "jailbreak_2023_12_25"), (False, "regular_2023_12_25")):
        for i, r in enumerate(_load(name, "train", config)):
            yield map_wild(r, i, jailbreak)


def load_aegis_prompts() -> Iterator[Row]:
    return (
        map_aegis_prompt(r) for r in _load("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "test")
    )


def load_aegis_responses() -> Iterator[Row]:
    rows = (
        map_aegis_response(r) for r in _load("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "test")
    )
    return (r for r in rows if r is not None)


def load_beavertails() -> Iterator[Row]:
    return (
        map_beavertails(r, i) for i, r in enumerate(_load("PKU-Alignment/BeaverTails", "30k_test"))
    )


def load_xstest() -> Iterator[Row]:
    return (map_xstest(r) for r in _load("Paul/XSTest", "train"))


def load_ai4privacy() -> Iterator[Row]:
    return (
        map_ai4privacy(r)
        for r in _load("ai4privacy/open-pii-masking-500k-ai4privacy", "validation")
    )


SOURCES: dict[str, Callable[[], Iterable[Row]]] = {
    "deepset": load_deepset,
    "in_the_wild": load_in_the_wild,
    "aegis2_prompts": load_aegis_prompts,
    "aegis2_responses": load_aegis_responses,
    "beavertails": load_beavertails,
    "xstest": load_xstest,
    "ai4privacy": load_ai4privacy,
}
