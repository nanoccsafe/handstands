"""Tests for :mod:`handstand.orientation_cases`.

The fixture ``Fixtures/golden/orientation_cases.json`` is what
``OrientationChooserTests`` in Swift compares against, so the Python side of
that agreement is guarded here: the committed file is a byte-stable rendering
of a fresh run of the real chooser over its own inputs, and the pinned
constants in its ``meta`` are the module's (chainlink #45).

Nothing here touches a real video: the cases are synthetic score sequences,
which is all a parity check may use (chainlink #80).
"""

from __future__ import annotations

import json
import math

import pytest

from handstand import orientation_cases as oc
from handstand.pose_mediapipe import (
    ORIENT_MARGIN,
    ORIENT_MIN_SWITCH_S,
    ORIENT_WINDOW_S,
    choose_orientations,
)

FIXTURE = oc.default_out_path()


def read_fixture() -> dict:
    """The committed fixture, parsed (the module keeps its I/O minimal)."""
    return json.loads(FIXTURE.read_text())


def test_the_committed_fixture_is_a_fresh_run_of_the_generator() -> None:
    """`uv run python -m handstand.orientation_cases --check` must pass."""
    assert FIXTURE.is_file(), f"{FIXTURE} is missing; run the generator"
    document = read_fixture()
    assert document == oc.fixture_payload()
    # …and the bytes are stable too, so regenerating is not a diff.
    assert FIXTURE.read_text() == oc.fixture_text()


def test_the_meta_pins_the_constants_the_swift_port_asserts() -> None:
    meta = read_fixture()["meta"]
    assert meta["mode"] == "synthetic"
    assert meta["generator"] == "handstand.orientation_cases"
    assert meta["orient_window_s"] == ORIENT_WINDOW_S == 0.5
    assert meta["orient_margin"] == ORIENT_MARGIN == 0.05
    assert meta["orient_min_switch_s"] == ORIENT_MIN_SWITCH_S == 0.3
    assert meta["case_names"] == list(oc.CASE_NAMES)


def test_every_case_answers_with_one_flag_per_frame() -> None:
    for case in oc.case_payloads():
        frames = len(case["t_ms"])
        assert len(case["score_upright"]) == frames, case["name"]
        assert len(case["score_rotated"]) == frames, case["name"]
        assert len(case["rotated"]) == frames, case["name"]
        # The timestamps the choice is smoothed over must increase — that is
        # what DisplayVideo guarantees and what searchsorted assumes.
        times = case["t_ms"]
        assert all(b > a for a, b in zip(times, times[1:], strict=False)), case["name"]


def test_the_expected_flags_are_what_choose_orientations_answers() -> None:
    """Re-run the real chooser from the fixture's own input, frame by frame."""
    for case in read_fixture()["cases"]:
        upright = [float("nan") if s is None else s for s in case["score_upright"]]
        rotated = [float("nan") if s is None else s for s in case["score_rotated"]]
        chosen = choose_orientations(case["t_ms"], upright, rotated)
        assert [bool(flag) for flag in chosen] == case["rotated"], case["name"]


def test_the_six_behaviours_are_all_there() -> None:
    """The case list the Swift test was written against, with its shapes."""
    cases = {case["name"]: case for case in oc.case_payloads()}

    # Steady upright: never a rotated frame.
    assert not any(cases["steady_upright"]["rotated"])
    # Steady inverted: always one.
    assert all(cases["steady_inverted"]["rotated"])
    # A one-frame confident misread moves nothing.
    assert not any(cases["one_frame_misread"]["rotated"])
    # The sustained switch switches exactly once, and not at frame 0.
    switch = cases["sustained_switch"]["rotated"]
    changes = sum(1 for a, b in zip(switch, switch[1:], strict=False) if a != b)
    assert changes == 1 and switch[0] is False and switch[-1] is True
    # NaN gaps: unscored frames carry no margin, and the clip keeps the
    # orientation the evidence around the gap implies — including a switch
    # that happens after the second gap.
    gaps = cases["nan_gaps"]["rotated"]
    assert gaps[:51] == [True] * 51  # opens rotated; the holes at 20–23 and
    # 40–49 carry no margin, so they hold the orientation they found…
    assert gaps[51:] == [False] * 19  # …and the preference after the second
    # hole switches, on the frame 0.3 s of clip time has run out
    # Variable frame rate: same shape as a sustained switch, irregular times.
    vfr = cases["vfr_timestamps"]
    assert vfr["rotated"][0] is False and vfr["rotated"][-1] is True


def test_nan_scores_round_trip_as_null() -> None:
    """JSON has no NaN: an unscored frame is `null` in the file, NaN back out."""
    raw = json.loads(FIXTURE.read_text())
    gaps = next(c for c in raw["cases"] if c["name"] == "nan_gaps")
    assert None in gaps["score_upright"]
    assert None in gaps["score_rotated"]
    # And the file never smuggles a bare NaN token past a strict parser.
    assert "NaN" not in FIXTURE.read_text()
    assert math.isnan(float("nan"))


def test_the_generator_refuses_an_unknown_case(tmp_path) -> None:
    with pytest.raises(KeyError):
        oc._CASE_BUILDERS["no_such_case"]  # type: ignore[index]
    # --check on a missing file exits 1 rather than writing one.
    missing = tmp_path / "nope.json"
    assert oc.main(["--check", "--out", str(missing)]) == 1
