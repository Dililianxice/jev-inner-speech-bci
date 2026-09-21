import numpy as np

from inner_speech_benchmark.benchmark import (
    choose_threshold,
    forward_split,
    parse_cue,
    prefix,
    score_events,
    trigger,
)


def test_subject_specific_cues_preserve_imagined_vs_passive():
    assert parse_cue("ban (imaginedlistening)") == ("imagined", "ban")
    assert parse_cue("ban (passivelistening)") == ("listening", "ban")
    assert parse_cue("ban,(mimed)") == ("attempted", "ban")
    assert parse_cue("ban$0") == ("listening", "ban")
    assert parse_cue("(DONOTHING),") == ("idle", "")


def test_250ms_prefix_never_rounds_into_future():
    data = {
        "x": np.arange(100.0).reshape(50, 2),
        "dt": 0.02,
        "start": np.array([5]),
        "end": np.array([40]),
    }
    expected, effective = prefix(data, 0.25)
    assert effective == 0.24
    data["x"][17:] = 99999
    np.testing.assert_array_equal(prefix(data, 0.25)[0], expected)


def test_late_first_miss_duplicates_and_exposure_are_separate():
    data = {
        "block": np.array([1, 1, 1]),
        "b": np.ones(100, dtype=int),
        "split": {"test": [1]},
        "dt": 0.1,
        "start": np.array([0, 40, 80]),
        "end": np.array([30, 70, 100]),
        "behavior": np.array(["imagined", "imagined", "idle"]),
        "word": np.array(["ban", "day", ""]),
    }
    events = [
        {"block": 1, "time": 1.5, "word": "ban"},
        {"block": 1, "time": 2.0, "word": "ban"},
        {"block": 1, "time": 8.0, "word": "day"},
        {"block": 1, "time": 10.0, "word": "day"},
    ]
    scores, _ = score_events(data, events, "test")
    assert (
        scores["imagined_trials"],
        scores["fast_correct"],
        scores["late_first"],
        scores["miss"],
    ) == (2, 0, 1, 1)
    assert scores["duplicates"] == 1
    assert scores["idle_events"] == 1
    assert scores["outside_scored_events"] == 1
    assert scores["idle_seconds"] == 2.0
    assert scores["continuous_nontarget_events"] == 2
    assert scores["continuous_nontarget_seconds"] == 4.0


def test_validation_rejects_unsafe_high_recall_threshold():
    rows = [
        dict(threshold=0.5, imagined_trials=10, fast_correct=9,
             continuous_nontarget_events=2, continuous_nontarget_seconds=60.0, duplicates=0),
        dict(threshold=0.9, imagined_trials=10, fast_correct=4,
             continuous_nontarget_events=1, continuous_nontarget_seconds=60.0, duplicates=0),
        dict(threshold=1.01, imagined_trials=10, fast_correct=0,
             continuous_nontarget_events=0, continuous_nontarget_seconds=60.0, duplicates=0),
    ]
    assert choose_threshold(rows) == 0.9


def test_outside_epoch_events_make_threshold_ineligible():
    rows = [
        dict(threshold=0.9, imagined_trials=10, fast_correct=9,
             continuous_nontarget_events=8, continuous_nontarget_seconds=120.0,
             duplicates=0),
        dict(threshold=1.01, imagined_trials=10, fast_correct=0,
             continuous_nontarget_events=0, continuous_nontarget_seconds=120.0,
             duplicates=0),
    ]
    assert choose_threshold(rows) == 1.01


def test_forward_split_never_mixes_recording_blocks():
    blocks = np.repeat([5, 6, 8, 9, 12], [2, 3, 1, 4, 2])
    split = forward_split(blocks)
    assert split == {"train": [5, 6, 8], "validation": [9], "test": [12]}
    assert not (set(split["train"]) & set(split["validation"]))
    assert not (set(split["train"]) & set(split["test"]))


def test_future_stream_changes_do_not_change_past_events():
    times = np.arange(20) * 0.1
    classes = np.array(["idle", "imagined"])
    probabilities = np.tile([0.01, 0.99], (20, 1))
    words = np.array(["ban"] * 20)
    expected = trigger(times[:10], probabilities[:10], classes, words[:10], 0.9)
    probabilities[10:] = [0.99, 0.01]
    words[10:] = "day"
    observed = [
        event
        for event in trigger(times, probabilities, classes, words, 0.9)
        if event["time"] < times[10]
    ]
    assert observed == expected
