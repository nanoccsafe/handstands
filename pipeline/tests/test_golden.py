"""Tests for :mod:`handstand.golden`.

Nothing here touches the real handstand videos. The fixtures under ``swift/``
are read back and re-run through the real stages, and the rest are generated on
the fly into a temporary directory — the two paths have to agree, because that
agreement *is* the guarantee a Swift port relies on: what the committed file
says is what the pipeline answers.

The three properties chainlink #25 asks for each get their own test: the
generator is deterministic (``test_the_generator_is_deterministic``), every
case round-trips (``test_every_case_round_trips``), and ``hand_step`` really
yields two holds while ``trainer_contact`` frames really are invalid. The fourth
guard, ``--real`` refusing a repository path, is what keeps keypoints of the
real videos out of a public tree.
"""

from __future__ import annotations

import json
import pathlib
import statistics
from collections.abc import Mapping

import numpy as np
import pytest

from handstand import features, golden, phases, postprocess

#: Where the committed fixtures live — the folder this test guards.
FIXTURES = golden.default_out_dir()


def fixture(case: str) -> dict:
    """One committed fixture, parsed."""
    return golden.read_fixture(FIXTURES / f"{case}.json")


def joints_of(document: Mapping) -> list[str]:
    """The joint names a fixture declares, i.e. the order of every array in it."""
    return [str(name) for name in document["meta"]["joint_names"]]


def clip_of(document: Mapping):
    """The fixture's ``input`` rebuilt into a clip, the way a reader would."""
    return golden.clip_from_frames(document["input"], joints_of(document))


def tolerance_for(path: str, tolerances: Mapping[str, float]) -> float:
    """The tolerance that applies to ``path``: the longest key that prefixes it.

    List indices are dropped first, so ``expected.features.12.com_u`` matches the
    key ``expected.features`` — a tolerance is a statement about a *field*, not
    about which frame of it.
    """
    parts = tuple(part for part in path.split(".") if not part.isdigit())
    best: tuple[str, ...] = ()
    value = float(tolerances.get("default", 1e-6))
    for key, tolerance in tolerances.items():
        key_parts = tuple(key.split("."))
        if parts[: len(key_parts)] == key_parts and len(key_parts) >= len(best):
            best, value = key_parts, float(tolerance)
    return value


def compare(stored: object, recomputed: object, tolerances: Mapping, path: str = "") -> list[str]:
    """Every place ``recomputed`` differs from ``stored`` past the field's tolerance."""
    problems: list[str] = []
    if isinstance(stored, Mapping):
        if not isinstance(recomputed, Mapping):
            return [f"{path or '<root>'}: expected an object, got {type(recomputed).__name__}"]
        if set(stored) != set(recomputed):
            missing = sorted(set(stored) - set(recomputed))
            extra = sorted(set(recomputed) - set(stored))
            problems.append(f"{path}: keys differ (missing={missing}, extra={extra})")
        for key in stored:
            if key in recomputed:
                problems += compare(stored[key], recomputed[key], tolerances, f"{path}.{key}")
    elif isinstance(stored, list):
        if not isinstance(recomputed, list):
            return [f"{path}: expected an array, got {type(recomputed).__name__}"]
        if len(stored) != len(recomputed):
            return [f"{path}: {len(stored)} entries stored, {len(recomputed)} recomputed"]
        for index, item in enumerate(stored):
            problems += compare(item, recomputed[index], tolerances, f"{path}.{index}")
    elif stored is None or recomputed is None:
        if stored is not recomputed:
            problems.append(f"{path}: {stored!r} stored, {recomputed!r} recomputed")
    elif isinstance(stored, bool) or isinstance(recomputed, bool):
        if stored is not recomputed:
            problems.append(f"{path}: {stored!r} stored, {recomputed!r} recomputed")
    elif isinstance(stored, str) or isinstance(recomputed, str):
        if stored != recomputed:
            problems.append(f"{path}: {stored!r} stored, {recomputed!r} recomputed")
    else:
        tolerance = tolerance_for(path, tolerances)
        if not np.isclose(float(stored), float(recomputed), rtol=0.0, atol=tolerance):
            problems.append(
                f"{path}: {stored!r} stored, {recomputed!r} recomputed (tolerance {tolerance:g})"
            )
    return problems


# --------------------------------------------------------------------------- #
# The committed fixtures
# --------------------------------------------------------------------------- #


def test_the_default_output_is_the_swift_fixtures_folder() -> None:
    assert FIXTURES == golden.repo_root() / "swift" / (
        "HandstandCore/Tests/HandstandCoreTests/Fixtures/golden"
    )
    assert FIXTURES.is_dir()


def test_the_readme_describing_the_format_is_committed() -> None:
    readme = FIXTURES / "README.md"
    assert readme.is_file()
    text = readme.read_text()
    for heading in ("meta", "input", "expected", "tolerances", "synthetic"):
        assert heading in text


@pytest.mark.parametrize("case", golden.CASES)
def test_every_case_is_committed_and_self_describing(case: str) -> None:
    document = fixture(case)
    meta = document["meta"]
    assert meta["case"] == case
    assert meta["mode"] == "synthetic"
    assert meta["generator"] == "handstand.golden"
    assert meta["generator_version"] == golden.GENERATOR_VERSION
    assert meta["seed"] == golden.DEFAULT_SEED
    assert meta["display"] == {"width": golden.DISPLAY_WIDTH, "height": golden.DISPLAY_HEIGHT}
    assert meta["L"] == 300.0
    assert isinstance(meta["body_length_px"], float) and meta["body_length_px"] > 100.0
    assert meta["joint_names"] == list(golden.JOINTS)
    assert meta["feature_columns"] == list(golden.FEATURE_COLUMNS)
    assert meta["tolerances"] == golden.TOLERANCES
    assert set(meta["packages"]) == {"python", "handstand", "numpy", "pandas"}

    frames = len(document["input"])
    assert frames > 0
    assert meta["frame_count"] == frames
    expected = document["expected"]
    assert len(expected["postprocess"]["frames"]) == frames
    assert len(expected["phases"]) == frames
    assert len(expected["features"]) == frames
    hold_ids = {row["hold_id"] for row in expected["phases"]}
    assert meta["hold_count"] == len(hold_ids - {-1})
    assert len(expected["hold_summary"]) == meta["hold_count"]


@pytest.mark.parametrize("case", golden.CASES)
def test_every_case_round_trips(case: str) -> None:
    """JSON → recomputed Python outputs → equal to the stored ``expected``."""
    document = fixture(case)
    problems = compare(
        document["expected"],
        golden.recompute(document),
        document["meta"]["tolerances"],
        "expected",
    )
    assert not problems, f"{case}:\n" + "\n".join(problems[:25])


@pytest.mark.parametrize("case", golden.CASES)
def test_the_generator_is_deterministic(case: str, tmp_path) -> None:
    """Same seed, identical JSON bytes — twice, written as files."""
    first = golden.write_fixture(tmp_path / "a", golden.build_fixture(case))
    second = golden.write_fixture(tmp_path / "b", golden.build_fixture(case))
    assert first.read_bytes() == second.read_bytes()


def test_a_different_seed_changes_the_noisy_case() -> None:
    """The seed is a real input, not decoration: the jitter comes out of it."""
    first = golden.fixture_text(golden.build_fixture("gaps_and_noise", seed=1))
    second = golden.fixture_text(golden.build_fixture("gaps_and_noise", seed=2))
    assert first != second


@pytest.mark.parametrize("case", golden.CASES)
def test_the_written_text_parses_back_to_the_same_document(case: str) -> None:
    """The hand-rolled writer is JSON, and the fixture survives a load."""
    document = golden.build_fixture(case)
    assert json.loads(golden.fixture_text(document)) == document


@pytest.mark.parametrize("case", golden.CASES)
def test_the_input_is_a_plausible_clip(case: str) -> None:
    """Variable frame rate, increasing time, pixels inside the 576x1024 frame."""
    document = fixture(case)
    times = [row["t_ms"] for row in document["input"]]
    assert times[0] == 0
    gaps = np.diff(times)
    assert gaps.min() > 0
    assert len(set(gaps.tolist())) > 1, "the fixtures are variable frame rate on purpose"
    assert 25 <= gaps.min() and gaps.max() <= 45
    assert [row["frame_idx"] for row in document["input"]] == list(range(len(times)))
    for row in document["input"]:
        assert set(row) == {"frame_idx", "t_ms", "detected", "trainer_contact", "joints"}
        assert set(row["joints"]) == set(golden.JOINTS)
        for values in row["joints"].values():
            assert len(values) == 3
            x, y, visibility = values
            if x is None:
                assert y is None and visibility is None
                assert not row["detected"], "only an undetected frame has no coordinates"
                continue
            assert 0.0 <= x <= golden.DISPLAY_WIDTH - 1
            assert 0.0 <= y <= golden.DISPLAY_HEIGHT - 1
            assert 0.0 <= visibility <= 1.0


def test_the_generated_clip_is_in_the_keypoint_schema(tmp_path) -> None:
    """The synthetic input is a real athlete parquet, not something shaped like one.

    It goes through the real loader — the one every stage reads with — so the
    fixture cannot be a private dialect the pipeline would never see.
    """
    clip = golden.generate_case("line_hold")
    path = tmp_path / "line_hold.parquet"
    clip.table.to_parquet(path, index=False)
    loaded = postprocess.read_clip(path)
    assert loaded.joints == clip.joints
    assert loaded.frames == clip.frames
    assert np.array_equal(loaded.frame_idx, clip.frame_idx)
    assert np.array_equal(loaded.t_ms, clip.t_ms)
    assert np.allclose(loaded.x, clip.x, equal_nan=True)
    assert np.allclose(loaded.y, clip.y, equal_nan=True)
    assert np.allclose(loaded.visibility, clip.visibility, equal_nan=True)
    assert np.array_equal(loaded.detected, clip.detected)
    assert np.array_equal(loaded.trainer_contact, clip.trainer_contact)


def test_the_committed_fixtures_fit_the_size_budget() -> None:
    total = sum(path.stat().st_size for path in FIXTURES.glob("*.json"))
    assert total <= golden.MAX_FIXTURE_BYTES, f"{total} bytes of committed fixtures"
    assert len(list(FIXTURES.glob("*.json"))) == len(golden.CASES)


def test_no_real_fixture_is_committed_under_swift() -> None:
    """The repo is public: whatever is under ``swift/`` is synthetic by construction."""
    for path in FIXTURES.glob("*.json"):
        text = path.read_text()
        assert golden.read_fixture(path)["meta"]["mode"] == "synthetic"
        assert "golden_real" not in text
        assert '"seed": null' not in text


# --------------------------------------------------------------------------- #
# What each case is for
# --------------------------------------------------------------------------- #


def test_line_hold_has_every_phase_and_a_four_second_hold() -> None:
    document = fixture("line_hold")
    rows = document["expected"]["phases"]
    assert {row["phase"] for row in rows} == {"pre", "kickup", "hold", "exit", "post"}
    assert document["meta"]["hold_count"] == 1

    held = [index for index, row in enumerate(rows) if row["phase"] == "hold"]
    times = [row["t_ms"] for row in document["input"]]
    duration_s = (times[held[-1]] - times[held[0]]) / 1000.0
    assert 3.8 <= duration_s <= 4.4, f"the hold is {duration_s:.2f} s, not about four"

    # The sway it is held with moves the centre of mass, and the body stays straight.
    measured = document["expected"]["features"]
    held_features = [
        row for row, phase in zip(measured, rows, strict=True) if phase["phase"] == "hold"
    ]
    sway = [row["com_forward"] for row in held_features if row["com_forward"] is not None]
    assert max(abs(value) for value in sway) > 0.005
    straight = sorted(row["line_deviation"] for row in held_features)
    assert straight[len(straight) // 2] < 0.1, "the control case is a line, not a shape fault"


def test_banana_hold_is_arched_through_the_hold() -> None:
    document = fixture("banana_hold")
    assert document["meta"]["hold_count"] == 1
    rows = document["expected"]["phases"]
    measured = document["expected"]["features"]
    hold = [row for row, phase in zip(measured, rows, strict=True) if phase["phase"] == "hold"]
    banana = statistics.median(row["banana"] for row in hold if row["banana"] is not None)
    deviation = statistics.median(
        row["line_deviation"] for row in hold if row["line_deviation"] is not None
    )
    assert banana > features.BANANA_TOLERANCE_L, f"median banana {banana:.4f} L"
    assert deviation > features.LINE_TOLERANCE_L, f"median line deviation {deviation:.4f} L"
    # The hips are ahead of the shoulder-ankle line, so the arch is on the belly side.
    assert statistics.median(
        row["off_hip"] for row in hold if row["off_hip"] is not None
    ) > 0.0


def test_hand_step_really_yields_two_holds() -> None:
    document = fixture("hand_step")
    assert document["meta"]["hold_count"] == 2

    frames: dict[int, list[int]] = {}
    for index, row in enumerate(document["expected"]["phases"]):
        if row["hold_id"] >= 0:
            frames.setdefault(row["hold_id"], []).append(index)
    assert set(frames) == {0, 1}
    assert len(document["expected"]["hold_summary"]) == 2

    clip = clip_of(document)
    processed = postprocess.process_clip(clip)
    signals = phases.frame_signals(
        clip.t_ms,
        processed.x,
        processed.y,
        processed.valid,
        clip.trainer_contact,
        clip.joints,
        processed.body_length.total,
    )
    stepped = np.flatnonzero(signals.hand_step)
    assert stepped.size > 0, "the wrist moved and the hand-step rule never saw it"
    assert float(np.nanmax(signals.wrist_step_l)) > phases.HAND_STEP_L

    # The first hold ends on the hand step itself — run end is the event frame
    # minus one — rather than expiring on the hold-break timeout.
    first_end, second_start = max(frames[0]), min(frames[1])
    event = next(index for index in stepped if index > min(frames[0]))
    assert event == first_end + 1, (
        f"hold 0 ends at frame {first_end}, the hand step is at frame {event}: "
        "the run should be ended by the step, not by the break timeout"
    )
    assert first_end < second_start
    # And the step really was one body length's worth of movement, 0.15 L.
    assert golden.STEP_L > phases.HAND_STEP_L


def test_gaps_and_noise_fills_the_short_gap_and_leaves_the_long_one() -> None:
    document = fixture("gaps_and_noise")
    frames = document["input"]
    post = document["expected"]["postprocess"]["frames"]
    times = [row["t_ms"] for row in frames]

    low_visibility = [
        index
        for index, row in enumerate(frames)
        if row["detected"] and row["joints"]["nose"][2] < postprocess.MIN_VISIBILITY
    ]
    undetected = [index for index, row in enumerate(frames) if not row["detected"]]
    assert len(low_visibility) == 4 and len(undetected) == 7

    # One stretch under the gap threshold and one over it, timed from the frames.
    short = (times[low_visibility[-1] + 1] - times[low_visibility[0] - 1]) / 1000.0
    long = (times[undetected[-1] + 1] - times[undetected[0] - 1]) / 1000.0
    assert short < postprocess.MAX_GAP_S < long, f"short={short} long={long}"

    # The short one is bridged, the long one is left as no position at all.
    filled = {index for index, row in enumerate(post) if any(row["filled"])}
    assert set(low_visibility) <= filled
    for index in undetected:
        assert not any(post[index]["valid"])
        assert not any(post[index]["filled"])
    # The short gap plus exactly one more frame: the joint that teleported.
    assert len(filled) == len(low_visibility) + 1
    outlier = next(iter(filled - set(low_visibility)))
    jumped = frames[outlier]["joints"]["left_ankle"][0]
    assert jumped > frames[outlier - 1]["joints"]["left_ankle"][0] + 100

    # The hold survives both of them: an occlusion is not a fall.
    assert document["meta"]["hold_count"] == 1


def test_trainer_contact_frames_are_invalid() -> None:
    document = fixture("trainer_contact")
    frames = document["input"]
    post = document["expected"]["postprocess"]["frames"]
    contact = [index for index, row in enumerate(frames) if row["trainer_contact"]]
    assert len(contact) == 7, "the spotter is in frame for a stretch, not one frame"

    times = [row["t_ms"] for row in frames]
    stretch = (times[contact[-1] + 1] - times[contact[0] - 1]) / 1000.0
    assert stretch > postprocess.MAX_GAP_S, "a bridged contact would be a valid frame again"

    for index in contact:
        assert not any(post[index]["valid"]), f"frame {index} is trainer contact but valid"
        assert all(frames[index]["joints"][name][0] is not None for name in golden.JOINTS)

    run = golden.run_pipeline(clip_of(document), "trainer_contact")
    assert not bool(run.features.valid[contact].any())
    assert not bool(run.phases.signals.known[contact].any())
    # ...and the hold the spotter walked through carries on over them.
    assert document["meta"]["hold_count"] == 1
    assert {document["expected"]["phases"][index]["hold_id"] for index in contact} == {0}


# --------------------------------------------------------------------------- #
# The real-clip mode must stay out of the repository
# --------------------------------------------------------------------------- #


def test_require_outside_repo_rejects_the_checkout() -> None:
    root = golden.repo_root()
    with pytest.raises(ValueError, match="public"):
        golden.require_outside_repo(root)
    with pytest.raises(ValueError, match="public"):
        golden.require_outside_repo(root / "swift" / "HandstandCore" / "Tests")
    with pytest.raises(ValueError, match="public"):
        golden.require_outside_repo(root / "data" / golden.REAL_SUBDIR)
    # A path outside the checkout is accepted, resolved.
    outside = golden.require_outside_repo(pathlib.Path("/tmp") / golden.REAL_SUBDIR)
    assert outside == pathlib.Path("/tmp") / golden.REAL_SUBDIR


def test_real_mode_refuses_an_output_path_inside_the_repo(tmp_path) -> None:
    target = golden.repo_root() / "swift" / golden.REAL_SUBDIR
    assert not target.exists()
    assert golden.main(["--real", "not_a_clip", "--real-out", str(target)]) == 2
    assert not target.exists()


def test_real_mode_refuses_a_data_dir_inside_the_repo() -> None:
    """Even the default ``--real-out`` is refused when it lands in the checkout."""
    data = golden.repo_root() / "data"
    target = data / golden.REAL_SUBDIR
    assert golden.main(["--real", "not_a_clip", "--data", str(data)]) == 2
    assert not target.exists()


def test_real_mode_reports_a_missing_clip_rather_than_writing(tmp_path) -> None:
    """A clip that is not there is a failure of that clip, not of the guard."""
    assert golden.main(["--real", "000000000000", "--real-out", str(tmp_path)]) == 1
    assert list(tmp_path.glob("*.json")) == []
