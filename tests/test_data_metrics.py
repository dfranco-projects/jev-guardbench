import math

import pytest

from guardbench import data, metrics


def test_split_is_stable_and_roughly_proportional():
    ids = [f"row-{i}" for i in range(5000)]
    splits = [data.split_of(i) for i in ids]
    assert splits == [data.split_of(i) for i in ids]
    assert 0.27 < splits.count("dev") / len(ids) < 0.33


def test_sample_keeps_label_share_and_is_seeded():
    rows = [data.Row(str(i), "t", "s", "x", label=i < 20) for i in range(100)]
    picked = data.sample(rows, 50, seed=1)
    assert len(picked) == 50
    assert sum(r.label for r in picked) == 10
    assert picked == data.sample(rows, 50, seed=1)


def test_deepset_rows_carry_the_assistant_purpose():
    row = data.map_deepset({"text": "Generate SQL code", "label": 1}, 3)
    assert row.label is True
    assert row.context == {"assistant_purpose": data.NEWS_ASSISTANT}


def test_aegis_response_skips_missing_responses_and_keeps_the_prompt():
    rec = {"id": "a", "prompt": "q", "response": "r", "response_label": "unsafe"}
    row = data.map_aegis_response(rec)
    assert row is not None and row.label is True
    assert row.context == {"user_message": "q"}
    assert data.map_aegis_response({**rec, "response": None}) is None


@pytest.mark.parametrize(
    ("labels", "expected"),
    [(["EMAIL"], True), (["GIVENNAME", "STREET"], True), (["GIVENNAME", "CITY", "DATE"], False)],
)
def test_ai4privacy_only_strong_identifiers_count(labels, expected):
    mask = str([{"label": label, "start": 0, "end": 1, "value": "v"} for label in labels])
    rec = {"uid": "1", "source_text": "t", "privacy_mask": mask, "language": "de"}
    row = data.map_ai4privacy(rec)
    assert row.label is expected
    assert row.lang == "de"


def test_confusion():
    m = metrics.confusion([True, True, False, False], [True, False, True, False])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == m["recall"] == m["f1"] == m["fpr"] == 0.5


def test_auroc():
    assert metrics.auroc([True, False], [0.9, 0.1]) == 1.0
    assert metrics.auroc([True, False], [0.1, 0.9]) == 0.0
    assert metrics.auroc([True, False], [0.5, 0.5]) == 0.5
    assert math.isnan(metrics.auroc([True, True], [0.5, 0.6]))


def test_ece_is_zero_when_calibrated_and_large_when_not():
    assert metrics.ece([True, False], [1.0, 0.0]) == 0.0
    assert metrics.ece([False, True], [1.0, 0.0]) == 1.0


def test_thresholds():
    labels = [False, False, True, True]
    probs = [0.1, 0.4, 0.35, 0.9]
    assert metrics.best_f1_threshold(labels, probs) == 0.35
    assert metrics.threshold_at_fpr(labels, probs, 0.0) == 0.9
    assert metrics.threshold_at_fpr(labels, probs, 0.5) == 0.35


def test_bootstrap_ci_brackets_the_statistic():
    values = [0.0] * 50 + [1.0] * 50
    lo, hi = metrics.bootstrap_ci(lambda idx: sum(values[i] for i in idx) / len(idx), 100)
    assert lo < 0.5 < hi
