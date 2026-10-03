"""``--rotate best``'s orientation chooser, and the synthetic parity cases.

Chainlink #45 ports :func:`handstand.pose_mediapipe.choose_orientations` to
the app (`swift/HandstandCore/Sources/HandstandCore/OrientationChooser.swift`)
so the phone picks each frame's orientation the same way the pipeline did —
the thresholds, the reference and the scorer were all built on keypoints from
that choice. This module is the fixture generator that makes the port
checkable: it runs the **real** Python chooser over hand-built **synthetic**
score sequences and writes input + expected output to one JSON file::

    swift/HandstandCore/Tests/HandstandCoreTests/Fixtures/golden/orientation_cases.json

Only score sequences live here — no keypoints, no frames, nothing derived
from the real videos (chainlink #80: parity checks stay synthetic). The cases
mirror what the pipeline tests already pin down one assertion at a time
(``pipeline/tests/test_pose_mediapipe.py``): a steady upright clip, a steady
inverted clip, a one-frame confident misread, a sustained switch, NaN gaps and
variable frame-rate timestamps.

CLI::

    cd pipeline
    uv run python -m handstand.orientation_cases            # regenerate
    uv run python -m handstand.orientation_cases --check    # fail if stale

The file is stable text: numbers are rounded to 1e-6, ``NaN`` is written as
``null``, and the keys are sorted, so the same generator produces the same
bytes. ``tests/test_orientation_cases.py`` reads the committed file back and
re-runs the chooser from its ``t_ms``/``score_upright``/``score_rotated``.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections.abc import Sequence
from typing import Any

from handstand.pose_mediapipe import (
    ORIENT_MARGIN,
    ORIENT_MIN_SWITCH_S,
    ORIENT_WINDOW_S,
    choose_orientations,
)

__all__ = [
    "CASE_NAMES",
    "DEFAULT_OUT",
    "FIXTURE_NAME",
    "GENERATOR_VERSION",
    "case_payloads",
    "default_out_path",
    "fixture_payload",
    "fixture_text",
    "main",
    "repo_root",
    "write_fixture",
]

#: The fixture's file name, beside the golden fixtures (same directory, so
#: one `Bundle.module` copy carries both).
FIXTURE_NAME = "orientation_cases.json"

#: Bumped when the cases themselves change shape — Swift's decoder reads it.
GENERATOR_VERSION = "1"

#: Where the fixture lives, relative to the repository root — the directory
#: ``handstand.golden.DEFAULT_OUT`` writes the golden fixtures into.
DEFAULT_OUT = pathlib.Path(
    "swift/HandstandCore/Tests/HandstandCoreTests/Fixtures/golden"
)

#: The cases, in file order (the Swift test walks them in this order).
CASE_NAMES: tuple[str, ...] = (
    "steady_upright",
    "steady_inverted",
    "one_frame_misread",
    "sustained_switch",
    "nan_gaps",
    "vfr_timestamps",
)


def repo_root() -> pathlib.Path:
    """The repository root, found from this file rather than from the cwd."""
    return pathlib.Path(__file__).resolve().parents[2]


def default_out_path() -> pathlib.Path:
    """The absolute path of the committed fixture."""
    return repo_root() / DEFAULT_OUT / FIXTURE_NAME


# --------------------------------------------------------------------------- #
# The cases: synthetic score sequences and the shapes they are meant to test
# --------------------------------------------------------------------------- #


def _regular_times(frames: int, step_ms: float = 100.0) -> list[float]:
    """`frames` timestamps `step_ms` apart — a fixed-frame-rate clip."""
    return [step_ms * index for index in range(frames)]


def _case_steady_upright() -> tuple[list[float], list[float], list[float]]:
    """Two seconds of the upright pass winning every frame: nothing moves."""
    return _regular_times(40), [0.9] * 40, [0.6] * 40


def _case_steady_inverted() -> tuple[list[float], list[float], list[float]]:
    """The same clip with the rotated pass winning: opens and stays rotated."""
    return _regular_times(40), [0.6] * 40, [0.9] * 40


def _case_one_frame_misread() -> tuple[list[float], list[float], list[float]]:
    """One frame the model is confidently wrong about — orientation holds.

    Frame 12 alone reports the rotated pass far surer (the pattern
    ``test_choose_orientations_reads_the_whole_clip_not_one_frame_at_a_time``
    uses with four frames; here one): smoothed over the half-second window it
    is a blip, not a preference, and the clip stays upright.
    """
    times = _regular_times(40)
    rotated = [0.6] * 40
    rotated[12] = 0.99
    return times, [0.9] * 40, rotated


def _case_sustained_switch() -> tuple[list[float], list[float], list[float]]:
    """The rotated preference holds from frame 10 on: the switch, timed.

    The exact sequence of ``test_choose_orientations_starts_from_the_first_
    half_second`` — upright for the first second, then confidently rotated —
    whose Python answer is 13 upright frames then 27 rotated ones.
    """
    return _regular_times(40), [0.9] * 40, [0.6] * 10 + [0.99] * 30


def _case_nan_gaps() -> tuple[list[float], list[float], list[float]]:
    """A stretch neither pass could score, then the preference resumes.

    Frames 20–23 are NaN on both sides (nobody found), and the clip is
    otherwise steadily inverted: the hole is not evidence, so the whole clip
    — hole included — is read rotated. Then frames 40–49 are another double
    NaN gap with a switch after it, so the gap's interaction with the
    hysteresis is pinned as well.
    """
    times = _regular_times(70)
    upright = [0.6] * 70
    rotated = [0.9] * 70
    for index in (20, 21, 22, 23):
        upright[index] = math.nan
        rotated[index] = math.nan
    # Second half: the upright pass turns confident from frame 50 on, after a
    # gap, so the switch has to hold across unscored frames' neighbours.
    for index in range(40, 70):
        upright[index] = math.nan if 40 <= index < 50 else 0.99
        rotated[index] = math.nan if 40 <= index < 50 else 0.6
    return times, upright, rotated


def _case_vfr_timestamps() -> tuple[list[float], list[float], list[float]]:
    """Variable frame rate: the window and the hysteresis are clip *time*.

    Timestamps step 60–160 ms irregularly (never a fixed rate); the rotated
    preference starts at frame 15 and the switch must land where 0.3 s of
    clip time has passed, not 0.3 s worth of frames.
    """
    steps = [
        80, 60, 140, 70, 160, 90, 65, 120, 75, 110,
        60, 150, 85, 70, 130, 65, 95, 140, 60, 105,
        75, 160, 68, 122, 88, 64, 145, 72, 118, 90,
        66, 135, 80, 100, 62, 155, 70, 92, 128, 78,
    ]
    times: list[float] = []
    clock = 0.0
    for step in steps:
        times.append(clock)
        clock += step
    upright = [0.9] * len(times)
    rotated = [0.6] * 15 + [0.99] * (len(times) - 15)
    return times, upright, rotated


_CASE_BUILDERS = {
    "steady_upright": _case_steady_upright,
    "steady_inverted": _case_steady_inverted,
    "one_frame_misread": _case_one_frame_misread,
    "sustained_switch": _case_sustained_switch,
    "nan_gaps": _case_nan_gaps,
    "vfr_timestamps": _case_vfr_timestamps,
}


def _round(value: float | None) -> float | None:
    """NaN (and inf, which never appears) out; everything else to 1e-6."""
    if value is None or not math.isfinite(value):
        return None
    return round(value, 6)


def case_payloads() -> list[dict[str, Any]]:
    """One case per entry: input sequences and the Python chooser's answer."""
    payloads: list[dict[str, Any]] = []
    for name in CASE_NAMES:
        times, upright, rotated = _CASE_BUILDERS[name]()
        chosen = choose_orientations(times, upright, rotated)
        assert len(chosen) == len(times), name
        payloads.append(
            {
                "name": name,
                "t_ms": [round(float(t), 6) for t in times],
                "score_upright": [_round(s) for s in upright],
                "score_rotated": [_round(s) for s in rotated],
                "rotated": [bool(flag) for flag in chosen],
            }
        )
    return payloads


def fixture_payload() -> dict[str, Any]:
    """The whole fixture: `meta` (with the pinned constants) + `cases`."""
    return {
        "meta": {
            "case": FIXTURE_NAME.removesuffix(".json"),
            "mode": "synthetic",
            "generator": "handstand.orientation_cases",
            "generator_version": GENERATOR_VERSION,
            # The constants OrientationChooser must reproduce exactly; the
            # Swift test asserts them against these values.
            "orient_window_s": ORIENT_WINDOW_S,
            "orient_margin": ORIENT_MARGIN,
            "orient_min_switch_s": ORIENT_MIN_SWITCH_S,
            "case_names": list(CASE_NAMES),
        },
        "cases": case_payloads(),
    }


def fixture_text(payload: dict[str, Any] | None = None) -> str:
    """The fixture as stable JSON: sorted keys, 1e-6 numbers, NaN as null."""
    payload = fixture_payload() if payload is None else payload
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_fixture(path: pathlib.Path | None = None) -> pathlib.Path:
    """Write the fixture (atomically) and return its path."""
    path = default_out_path() if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(fixture_text())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the OrientationChooser parity fixture (synthetic only)."
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help=f"output file (default: {DEFAULT_OUT / FIXTURE_NAME} under the repository root)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 when the committed fixture is stale",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    path = args.out if args.out is not None else default_out_path()
    if args.check:
        current = path.read_text() if path.exists() else ""
        if current != fixture_text():
            print(
                f"{path} is stale; run: uv run python -m handstand.orientation_cases",
                file=sys.stderr,
            )
            return 1
        print(f"{path} is up to date")
        return 0
    written = write_fixture(path)
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
