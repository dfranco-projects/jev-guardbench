from guardbench.hypotheses import Roles, determinism, h1, h2, render_hypotheses, tuned_threshold

ROLES = Roles(system_one="s1", decomposed=None, judges=["j"], cascade=None)


def rec(guard, i, *, label, flagged, conc=1, latency=10.0, prob=None, split="test", **kw):
    return {
        "id": f"r{i}",
        "task": kw.get("task", "harmful_request"),
        "source": kw.get("source", "aegis2"),
        "split": split,
        "label": label,
        "lang": "en",
        "chars": 100,
        "guard": guard,
        "concurrency": conc,
        "ts": 1_790_000_000.0,
        "flagged": flagged,
        "latency_ms": latency,
        "prob": prob,
        "cost_usd": 0.0,
        "error": None,
        "note": None,
        "escalated": False,
        "model": guard,
    }


def latency_recs(s1_ms, judge_ms):
    return [
        rec(g, i, label=True, flagged=True, conc=c, latency=ms)
        for c in (1, 4)
        for g, ms in (("s1", s1_ms), ("j", judge_ms))
        for i in range(20)
    ]


def test_h1_verdict_follows_the_ratio_thresholds():
    assert "**Supported**" in h1(latency_recs(10, 60), ROLES)[0]
    assert "**Inconclusive**" in h1(latency_recs(10, 30), ROLES)[0]
    assert "**Rejected**" in h1(latency_recs(10, 15), ROLES)[0]


def test_h1_needs_both_concurrency_levels():
    recs = [r for r in latency_recs(10, 60) if r["concurrency"] == 1]
    assert "Not decidable" in h1(recs, ROLES)[0]


def quality_recs(s1_correct):
    """Judge is always right; the System One guard is right on the first `s1_correct` rows."""
    recs = []
    for i in range(200):
        label = i % 2 == 0
        recs.append(rec("j", i, label=label, flagged=label))
        s1 = label if i < s1_correct else not label
        recs.append(rec("s1", i, label=label, flagged=s1, prob=0.9 if s1 else 0.1))
    for i in range(300, 310):  # safe XSTest prompts both guards allow
        recs.append(rec("j", i, label=False, flagged=False, source="xstest"))
        recs.append(rec("s1", i, label=False, flagged=False, prob=0.1, source="xstest"))
    return recs


def groups(recs):
    out = {}
    for r in recs:
        out.setdefault((r["task"], r["concurrency"], r["guard"]), []).append(r)
    return out


def test_h2_passes_an_identical_guard_and_fails_a_worse_one():
    assert "**Supported**" in h2(groups(quality_recs(200)), ROLES)[0]
    assert "**Not supported**" in h2(groups(quality_recs(120)), ROLES)[0]


def test_h2_over_blocking_on_xstest_fails_the_task():
    recs = quality_recs(200)
    for r in recs:  # System One now blocks the safe XSTest prompts the judge allows
        if r["source"] == "xstest" and r["guard"] == "s1":
            r.update(flagged=True, prob=0.9)
    assert "**Not supported**" in h2(groups(recs), ROLES)[0]


def test_tuned_threshold_comes_from_dev_rows_only():
    dev = [
        rec("s1", i, label=i < 5, flagged=False, prob=0.3 if i < 5 else 0.1, split="dev")
        for i in range(10)
    ]
    test = [rec("s1", 100 + i, label=True, flagged=True, prob=0.99) for i in range(10)]
    assert tuned_threshold(dev + test) == 0.3


def test_roles_are_read_from_the_config():
    guards = {
        "jev": {"type": "systemone"},
        "jev-d": {"type": "systemone", "decompose": True},
        "g": {"type": "gemini"},
        "c": {"type": "claude"},
        "cas": {"type": "cascade", "fast": "jev", "slow": "g"},
    }
    roles = Roles.from_guards(guards, list(guards))
    assert roles == Roles(system_one="jev", decomposed="jev-d", judges=["g", "c"], cascade="cas")


def test_render_runs_on_a_full_set():
    text = render_hypotheses(latency_recs(10, 60) + quality_recs(200), ROLES)
    assert "## H1 latency" in text and "## Provenance" in text


def test_determinism_counts_flips_and_probability_drift():
    runs = [
        [rec("s1", 0, label=True, flagged=True, prob=0.9), rec("j", 0, label=True, flagged=True)],
        [rec("s1", 0, label=True, flagged=True, prob=0.7), rec("j", 0, label=True, flagged=False)],
    ]
    lines = determinism(runs)
    assert "| j | 1 | 1.000 | – | – |" in lines
    assert "| s1 | 1 | 0.000 | 0.200 | 0.200 |" in lines
