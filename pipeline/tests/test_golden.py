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
guard, ``--real`` refusing every path git would commit (and every source tree
outright), is what keeps keypoints of the real videos out of a public tree — it
is tested against a temporary repository laid out like the main checkout, whose
``data/`` sits *inside* the repo and is git-ignored.
"""

from __future__ import annotations

import json
import pathlib
import statistics
import subprocess
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
# The real-clip mode must never write where git would commit
# --------------------------------------------------------------------------- #


def checkout_like_the_main_one(tmp_path, ignore: tuple[str, ...] = ("data/",)) -> pathlib.Path:
    """A miniature checkout laid out like the main one: ``data/`` *inside* it.

    The main checkout keeps the shared data tree within the repository —
    git-ignored, as its ``.gitignore`` says — so a rule of the shape "real
    fixtures stay outside the repository" breaks ``--real`` there. ``git init``
    plus a ``.gitignore`` models that layout faithfully, because
    ``require_outside_repo`` asks git, not the file tree, what may be written.
    """
    root = tmp_path / "checkout"
    (root / "data").mkdir(parents=True)
    for tree in golden.NEVER_REAL_TREES:
        (root / tree).mkdir(exist_ok=True)
    (root / "pipeline" / "pyproject.toml").write_text("[project]\nname = 'handstand'\n")
    (root / ".gitignore").write_text("".join(f"{pattern}\n" for pattern in ignore))
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
    return root


def test_the_default_real_out_is_accepted_inside_an_ignored_data_tree(tmp_path) -> None:
    """The main-checkout case: ``data/`` is in the repo, git ignores it → allowed."""
    root = checkout_like_the_main_one(tmp_path)
    target = root / "data" / golden.REAL_SUBDIR
    assert golden.require_outside_repo(target, root=root) == target.resolve()


def test_main_accepts_the_default_real_out_when_data_sits_inside_the_repo(
    tmp_path, monkeypatch
) -> None:
    """The CLI under that layout: the guard passes, only the clip itself fails.

    Exit code 1 is "the clip is missing"; 2 is the guard refusing, which is
    what this did before the rule became "where git would commit it".
    """
    root = checkout_like_the_main_one(tmp_path)
    monkeypatch.setattr(golden, "repo_root", lambda: root)
    target = root / "data" / golden.REAL_SUBDIR
    assert golden.main(["--real", "000000000000", "--data", str(root / "data")]) == 1
    assert not target.exists()


def test_a_pre_existing_real_fixture_is_left_untouched(tmp_path, monkeypatch) -> None:
    """An ignored ``data/golden_real`` that already holds files: run adds nothing.

    The main checkout's data directory legitimately contains fixtures from an
    earlier ``--real`` run, so a missing clip must leave those files exactly as
    they were — and must not fail because the folder happens to be there.
    """
    root = checkout_like_the_main_one(tmp_path)
    target = root / "data" / golden.REAL_SUBDIR
    target.mkdir(parents=True)
    existing = target / "existing.json"
    existing.write_text('{"mode": "synthetic"}\n')
    monkeypatch.setattr(golden, "repo_root", lambda: root)

    assert golden.main(["--real", "not_a_clip", "--data", str(root / "data")]) == 1

    assert existing.read_text() == '{"mode": "synthetic"}\n'
    assert sorted(entry.name for entry in target.iterdir()) == ["existing.json"]


def test_a_tracked_path_inside_the_repo_is_refused(tmp_path) -> None:
    """git would commit it, so it is refused — even force-added under ``data/``.

    ``git check-ignore`` consults the index like git does, which is what makes
    a file someone staged despite the ignore rule (or committed long ago) as
    dangerous as any other tracked file.
    """
    root = checkout_like_the_main_one(tmp_path)
    forced = root / "data" / "forced.json"
    forced.write_text("{}\n")
    subprocess.run(["git", "add", "-f", "data/forced.json"], cwd=root, check=True)
    with pytest.raises(ValueError, match="public"):
        golden.require_outside_repo(forced, root=root)

    plain = root / "README.md"
    plain.write_text("# checkout\n")
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    with pytest.raises(ValueError, match="public"):
        golden.require_outside_repo(plain, root=root)


def test_an_unignored_path_inside_the_repo_is_refused(tmp_path) -> None:
    """Nothing covers it, so git would commit it: refused — the root included."""
    root = checkout_like_the_main_one(tmp_path)
    with pytest.raises(ValueError, match="public"):
        golden.require_outside_repo(root / golden.REAL_SUBDIR, root=root)
    with pytest.raises(ValueError, match="public"):
        golden.require_outside_repo(root, root=root)


@pytest.mark.parametrize("tree", golden.NEVER_REAL_TREES)
def test_a_source_tree_is_refused_even_when_git_ignores_it(tmp_path, tree: str) -> None:
    """swift/, ios/, pipeline/ and docs/ hold no real data, ignore rules or not."""
    root = checkout_like_the_main_one(
        tmp_path, ignore=("data/", *(f"{name}/" for name in golden.NEVER_REAL_TREES))
    )
    target = root / tree / golden.REAL_SUBDIR
    covered = subprocess.run(["git", "check-ignore", "-q", str(target)], cwd=root)
    assert covered.returncode == 0, "the fixture: the ignore rule must cover it"
    with pytest.raises(ValueError, match="public"):
        golden.require_outside_repo(target, root=root)


def test_a_path_outside_the_repository_is_accepted(tmp_path) -> None:
    root = checkout_like_the_main_one(tmp_path)
    outside = tmp_path / "elsewhere" / golden.REAL_SUBDIR
    assert golden.require_outside_repo(outside, root=root) == outside.resolve()


def test_require_outside_repo_in_this_checkout(tmp_path) -> None:
    """The live layout: source trees out, this checkout's own ``data/`` in."""
    root = golden.repo_root()
    with pytest.raises(ValueError, match="public"):
        golden.require_outside_repo(root)
    with pytest.raises(ValueError, match="public"):
        golden.require_outside_repo(root / "swift" / "HandstandCore" / "Tests")
    with pytest.raises(ValueError, match="public"):
        golden.require_outside_repo(root / "docs" / "keypoint_schema.md")
    # ``data/`` is inside the checkout but git ignores it, so the default stands.
    target = root / "data" / golden.REAL_SUBDIR
    assert golden.require_outside_repo(target) == target.resolve()
    # A path outside the checkout is accepted, resolved.
    outside = tmp_path / golden.REAL_SUBDIR
    assert golden.require_outside_repo(outside) == outside.resolve()


def test_real_mode_refuses_an_output_path_inside_the_repo() -> None:
    """Under ``swift/`` even: exit 2 from the guard, nothing written.

    The assertions are about *this run* — the clip the guard rejected must not
    leave a fixture behind — not about whether the destination folder already
    exists, which no test may assume of a path it did not create.
    """
    target = golden.repo_root() / "swift" / golden.REAL_SUBDIR
    marker = target / "not_a_clip.json"
    assert not marker.exists()
    assert golden.main(["--real", "not_a_clip", "--real-out", str(target)]) == 2
    assert not marker.exists()


def test_real_mode_accepts_the_default_data_dir_that_git_ignores() -> None:
    """The default ``--real-out`` of this checkout: in the repo, ignored, allowed.

    The live data directory may already hold fixtures from an earlier ``--real``
    run — in the main checkout it does — so nothing here asserts whether the
    folder exists, only that this missing clip writes no file of its own.
    """
    data = golden.repo_root() / "data"
    marker = data / golden.REAL_SUBDIR / "not_a_clip.json"
    existed_before = marker.exists()
    assert golden.main(["--real", "not_a_clip", "--data", str(data)]) == 1
    assert not existed_before, "this clip never ran; its fixture must not be there"
    assert not marker.exists(), "and the run must not have written it"


def test_real_mode_reports_a_missing_clip_rather_than_writing(tmp_path) -> None:
    """A clip that is not there is a failure of that clip, not of the guard."""
    assert golden.main(["--real", "000000000000", "--real-out", str(tmp_path)]) == 1
    assert list(tmp_path.glob("*.json")) == []
