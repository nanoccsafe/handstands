"""Render keypoint overlays: the same pose, drawn on the video it came from.

Two jobs, one module:

* :func:`draw_pose` and :func:`draw_caption` are pure drawing functions on a
  numpy BGR frame. Later stages draw feature values and scores on the very same
  frames, so the drawing is kept apart from the video plumbing and is usable on
  its own.
* :func:`render_overlay` decodes a clip in display orientation, looks up each
  frame's pose in one or more keypoint parquets by ``frame_idx``, composes the
  panels side by side and writes an MP4::

      <data_dir>/overlays/<clip_id>_<mode>.mp4          # one panel
      <data_dir>/overlays/<clip_id>_none-vs-auto.mp4     # two panels

Frames come from :class:`handstand.pose_mediapipe.DisplayVideo`, the same reader
the runner used, so every drawn keypoint lands in the coordinate system the
parquet already stores (upright display pixels) with no conversion. Bones are
MediaPipe's own skeleton, named through
:data:`handstand.pose_mediapipe.JOINT_NAMES`, and the keypoint layout is the one
in ``docs/keypoint_schema.md``.

``--source`` picks which keypoints are drawn:

``mediapipe`` (default)
    ``<data_dir>/keypoints/mediapipe/<mode>/``, the plain single-person runner.
``athlete``
    ``<data_dir>/keypoints/mediapipe_athlete/<mode>/``, the athlete selection
    from :mod:`handstand.athlete`. Same schema, plus a ``trainer_contact``
    column: a frame flagged there is drawn with a red border and the extra
    caption :data:`TRAINER_CONTACT_LABEL`, so the frames that must not be scored
    are obvious when watching the clip.

CLI::

    cd pipeline
    uv run python -m handstand.overlay --clip 6508f9b355bd --modes none auto
    uv run python -m handstand.overlay --clip 6508f9b355bd --source athlete
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import pathlib
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
from mediapipe.tasks.python import vision as mp_vision

from handstand.athlete import ATHLETE_OUTPUT_DIRNAME
from handstand.paths import data_dir, videos_dir
from handstand.pose_mediapipe import (
    JOINT_NAMES,
    ROTATE_MODES,
    SINGLE_OUTPUT_DIRNAME,
    DisplayVideo,
    load_catalogue,
    resolve_clip_id,
)

__all__ = [
    "BONES",
    "CAPTION_BACKGROUND_ALPHA",
    "CAPTION_FONT",
    "COLOR_ALERT",
    "COLOR_LOW",
    "COLOR_OK",
    "COLOR_GOOD",
    "COLOR_TEXT",
    "CONTACT_COLUMN",
    "DEFAULT_FPS",
    "DEFAULT_SOURCE",
    "FILE_MODE",
    "FRAME_COUNT_TOLERANCE",
    "MEDIA_FOURCC",
    "MIN_VISIBILITY",
    "NO_POSE_LABEL",
    "REFERENCE_HEIGHT",
    "SOURCES",
    "TRAINER_CONTACT_LABEL",
    "VISIBILITY_GOOD",
    "VISIBILITY_OK",
    "FrameKeypoints",
    "OverlayReport",
    "PanelKeypoints",
    "build_arg_parser",
    "caption_metrics",
    "draw_border",
    "draw_caption",
    "draw_metrics",
    "draw_pose",
    "keypoints_root",
    "load_panel",
    "main",
    "missing_parquet_message",
    "overlay_filename",
    "overlays_dir",
    "panel_parquet_path",
    "render_clip",
    "render_overlay",
    "require_parquet",
    "source_dirname",
    "video_for_clip",
    "visibility_colour",
]

#: MediaPipe's own skeleton, as joint *names*, so a renderer needs no index maths.
BONES: tuple[tuple[str, str], ...] = tuple(
    (JOINT_NAMES[connection.start], JOINT_NAMES[connection.end])
    for connection in mp_vision.PoseLandmarksConnections.POSE_LANDMARKS
)

#: MediaPipe visibility at or above this reads as confident.
VISIBILITY_GOOD = 0.8
#: MediaPipe visibility at or above this is still usable; below it it is not.
VISIBILITY_OK = 0.5
#: Default ``min_visibility`` of :func:`draw_pose`.
MIN_VISIBILITY = VISIBILITY_OK

#: BGR colours for confident / usable / barely-there keypoints.
COLOR_GOOD = (40, 200, 40)
COLOR_OK = (0, 165, 255)
COLOR_LOW = (0, 0, 220)

#: Caption text colour, and how much of the dark plate shows through (0..1).
COLOR_TEXT = (245, 245, 245)
CAPTION_BACKGROUND_ALPHA = 0.55

#: BGR colour of the "this frame must not be scored" border, and its thickness in
#: pixels at :data:`REFERENCE_HEIGHT`.
COLOR_ALERT = (0, 0, 220)
BORDER_THICKNESS = 6

#: Frame height the drawing constants below are calibrated for.
REFERENCE_HEIGHT = 720
#: Bone thickness and joint radius in pixels at :data:`REFERENCE_HEIGHT`.
BONE_THICKNESS = 2
JOINT_RADIUS = 5
#: Caption font scale and text thickness at :data:`REFERENCE_HEIGHT`.
CAPTION_FONT_SCALE = 0.6
CAPTION_THICKNESS = 2
#: Caption plate padding in pixels at :data:`REFERENCE_HEIGHT`.
CAPTION_PADDING = 4

#: Font used for every caption drawn by this module.
CAPTION_FONT = cv2.FONT_HERSHEY_SIMPLEX

#: Caption line shown for a frame where the model found no pose.
NO_POSE_LABEL = "no pose"

#: Extra caption line on a frame the athlete selection flagged: the trainer is
#: overlapping or touching the athlete there, so the frame is not scored.
TRAINER_CONTACT_LABEL = "trainer contact"
#: The column of the ``--source athlete`` parquets carrying that flag.
CONTACT_COLUMN = "trainer_contact"

#: ``--source`` choices: which ``keypoints/`` root a panel is read from.
SOURCES: tuple[str, ...] = ("mediapipe", "athlete")
#: The plain single-person runner, unchanged from before the flag existed.
DEFAULT_SOURCE = "mediapipe"
#: ``--source`` value -> the directory name under ``<data_dir>/keypoints/``.
SOURCE_DIRNAMES: dict[str, str] = {
    DEFAULT_SOURCE: SINGLE_OUTPUT_DIRNAME,
    "athlete": ATHLETE_OUTPUT_DIRNAME,
}

#: Frames a video and a keypoint parquet may disagree by before the render fails.
FRAME_COUNT_TOLERANCE = 1

#: FourCC of the written files. ``mp4v`` plays anywhere and needs no extra codec.
MEDIA_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

#: Frame rate used when the keypoints carry no usable duration (a single frame).
DEFAULT_FPS = 30.0

#: Overlays are shared, human-watched files, so they are not left private.
FILE_MODE = 0o644

#: Name of the output directory inside the data directory.
OVERLAYS_DIR_NAME = "overlays"

#: Columns a keypoint parquet must carry to be drawable.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "frame_idx",
    "t_ms",
    "joint",
    "x",
    "y",
    "visibility",
    "detected",
)


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #


def _scale(height: int) -> float:
    """Drawing scale for a frame of this height; 1.0 at :data:`REFERENCE_HEIGHT`."""
    return max(1, int(height)) / REFERENCE_HEIGHT


def draw_metrics(height: int) -> tuple[int, int]:
    """``(bone_thickness, joint_radius)`` in pixels for a frame of this height.

    Both scale linearly with the frame height, so an overlay stays legible on a
    4k clip and does not vanish on a thumbnail; both have a floor so that a very
    small panel still shows a skeleton rather than single pixels.
    """
    scale = _scale(height)
    return max(1, round(BONE_THICKNESS * scale)), max(2, round(JOINT_RADIUS * scale))


def draw_border(
    frame: np.ndarray,
    colour: tuple[int, int, int] = COLOR_ALERT,
    thickness: int | None = None,
) -> np.ndarray:
    """Draw a rectangle just inside the frame's edge and return a new frame.

    Used to make a frame that must not be scored (trainer contact) obvious at a
    glance. The line scales with the frame height like everything else here and
    is drawn *inside* the frame, so the composite's panel borders stay straight.
    The input array is never modified.
    """
    out = frame.copy()
    height, width = out.shape[0], out.shape[1]
    if height < 2 or width < 2:
        return out
    line = max(1, round(BORDER_THICKNESS * _scale(height))) if thickness is None else thickness
    inset = line // 2
    cv2.rectangle(out, (inset, inset), (width - 1 - inset, height - 1 - inset), colour, line)
    return out


def caption_metrics(height: int) -> tuple[float, int]:
    """``(font_scale, text_thickness)`` for a caption on a frame of this height."""
    scale = _scale(height)
    return max(0.4, round(CAPTION_FONT_SCALE * scale, 3)), max(1, round(CAPTION_THICKNESS * scale))


def visibility_colour(visibility: float) -> tuple[int, int, int]:
    """BGR colour for a MediaPipe visibility score.

    Green at :data:`VISIBILITY_GOOD` and above, amber down to
    :data:`VISIBILITY_OK`, red below. A NaN score falls through to red, but
    :func:`draw_pose` skips such joints anyway.
    """
    if visibility >= VISIBILITY_GOOD:
        return COLOR_GOOD
    if visibility >= VISIBILITY_OK:
        return COLOR_OK
    return COLOR_LOW


def _as_point(value: Sequence[float]) -> tuple[float | None, float | None]:
    """Coerce one ``(x, y)`` to floats; ``(None, None)`` when either is not finite."""
    x, y = float(value[0]), float(value[1])
    return (x, y) if (math.isfinite(x) and math.isfinite(y)) else (None, None)


def _drawable_points(
    joints_xy: Mapping[str, tuple[float, float]],
    visibility: Mapping[str, float],
    min_visibility: float,
) -> dict[str, tuple[int, int]]:
    """Integer pixel positions of the joints that may be drawn.

    A joint is dropped when its coordinates are missing (NaN — the parquet stores
    NaN for a joint the model never reported) or when its visibility is unknown
    or below ``min_visibility``.
    """
    points: dict[str, tuple[int, int]] = {}
    for name, xy in joints_xy.items():
        score = visibility.get(name, float("nan"))
        if not math.isfinite(score) or score < min_visibility:
            continue
        x, y = _as_point(xy)
        if x is None or y is None:
            continue
        points[name] = (int(round(x)), int(round(y)))
    return points


def draw_pose(
    frame: np.ndarray,
    joints_xy: Mapping[str, tuple[float, float]],
    visibility: Mapping[str, float],
    min_visibility: float = MIN_VISIBILITY,
) -> np.ndarray:
    """Draw one pose on a BGR frame and return a new frame.

    Bones are drawn first (MediaPipe's own connections, named through
    :data:`BONES`) and joints on top as filled circles, so a joint never hides
    the line that ends on it. Bones are coloured by the mean visibility of their
    two ends, joints by their own: green at or above
    :data:`VISIBILITY_GOOD`, amber down to :data:`VISIBILITY_OK`, red below. A
    bone needs *both* of its joints drawn, so lowering ``min_visibility`` to see
    the red joints does not produce half-drawn limbs.

    The input array is never modified; the returned frame is a new array.
    """
    out = frame.copy()
    thickness, radius = draw_metrics(out.shape[0])
    points = _drawable_points(joints_xy, visibility, min_visibility)

    for start, end in BONES:
        first, second = points.get(start), points.get(end)
        if first is None or second is None:
            continue
        mean = (visibility[start] + visibility[end]) / 2
        cv2.line(out, first, second, visibility_colour(mean), thickness)

    for name, (x, y) in points.items():
        cv2.circle(out, (x, y), radius, visibility_colour(visibility[name]), -1)

    return out


def draw_caption(frame: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    """Draw a small text block in the top-left corner and return a new frame.

    The block is a dark plate behind the text, so captions stay readable over a
    bright wall or a pale mat. Font size scales with the frame height the same
    way the skeleton does. An empty ``lines`` returns an unchanged copy; the
    input array is never modified.
    """
    out = frame.copy()
    text_lines = [str(line) for line in lines]
    if not text_lines:
        return out

    height, width = out.shape[0], out.shape[1]
    font_scale, thickness = caption_metrics(height)
    padding = max(2, round(CAPTION_PADDING * _scale(height)))
    sizes = [cv2.getTextSize(line, CAPTION_FONT, font_scale, thickness)[0] for line in text_lines]
    text_width = max(w for w, _ in sizes)
    text_height = max(h for _, h in sizes)
    step = max(text_height + 1, round(text_height * 1.4))

    right = min(width, padding + text_width + 2 * padding)
    bottom = min(height, padding + step * (len(text_lines) - 1) + text_height + 2 * padding)
    if right > padding and bottom > padding:
        plate = out[padding:bottom, padding:right]
        dark = np.zeros_like(plate)
        out[padding:bottom, padding:right] = cv2.addWeighted(
            dark, CAPTION_BACKGROUND_ALPHA, plate, 1.0 - CAPTION_BACKGROUND_ALPHA, 0.0
        )

    for index, line in enumerate(text_lines):
        baseline = min(height - 1, padding + text_height + step * index)
        cv2.putText(
            out,
            line,
            (padding, baseline),
            CAPTION_FONT,
            font_scale,
            COLOR_TEXT,
            thickness,
        )
    return out


# --------------------------------------------------------------------------- #
# Keypoints
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class FrameKeypoints:
    """One frame of a keypoint parquet, in display-frame pixels."""

    frame_idx: int
    t_ms: int
    detected: bool
    joints_xy: dict[str, tuple[float, float]]
    visibility: dict[str, float]
    #: ``--source athlete`` parquets flag the frames where the trainer overlaps
    #: or touches the athlete; plain runner output has no such column and is
    #: never in contact.
    trainer_contact: bool = False


@dataclasses.dataclass(frozen=True)
class PanelKeypoints:
    """Every frame of one keypoint parquet, addressed by ``frame_idx``.

    ``frames`` is sparse-friendly: a renderer looks frames up by index, and a
    missing index is exactly how a video/keypoint mismatch shows up.
    """

    label: str
    path: pathlib.Path
    frames: Mapping[int, FrameKeypoints]

    @property
    def frame_count(self) -> int:
        """How many frames this parquet covers."""
        return len(self.frames)

    @property
    def max_frame_idx(self) -> int:
        """Highest ``frame_idx`` present, or ``-1`` for an empty parquet."""
        return max(self.frames, default=-1)

    @property
    def detected_frames(self) -> int:
        """How many frames have a pose."""
        return sum(frame.detected for frame in self.frames.values())

    @property
    def duration_seconds(self) -> float:
        """``last t_ms`` in seconds — the clip length the runner measured."""
        last_t_ms = max((frame.t_ms for frame in self.frames.values()), default=0)
        return last_t_ms / 1000.0

    def frame(self, frame_idx: int) -> FrameKeypoints | None:
        """The keypoints of ``frame_idx``, or ``None`` when the parquet stops earlier."""
        return self.frames.get(frame_idx)

    def fps(self) -> float:
        """``frame_count / duration`` — the average frame rate of the clip.

        The clips are variable frame rate, so the frame count over the measured
        duration is the honest rate; it is taken from the keypoints rather than
        from a decoded frame count so the writer can be opened before the first
        frame is read.
        """
        duration = self.duration_seconds
        if duration <= 0 or not self.frame_count:
            return DEFAULT_FPS
        return self.frame_count / duration


def load_panel(label: str, parquet_path: str | pathlib.Path) -> PanelKeypoints:
    """Read one keypoint parquet into a :class:`PanelKeypoints`.

    ``label`` is what a caption calls this panel (the rotation mode, normally).
    A frame with no detection keeps its 33 NaN rows, so ``detected`` is the
    flag to filter on rather than the coordinates. A ``trainer_contact`` column
    — the ``--source athlete`` parquets — is picked up when it is there and
    treated as "not in contact" when it is not.
    """
    path = pathlib.Path(parquet_path)
    table = pd.read_parquet(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(
            f"{path}: keypoint parquet is missing column(s) {missing}; see docs/keypoint_schema.md"
        )
    has_contact = CONTACT_COLUMN in table.columns

    frames: dict[int, FrameKeypoints] = {}
    for frame_idx, group in table.groupby("frame_idx", sort=True):
        joints = [str(name) for name in group["joint"]]
        xs = [float(value) for value in group["x"]]
        ys = [float(value) for value in group["y"]]
        scores = [float(value) for value in group["visibility"]]
        frames[int(frame_idx)] = FrameKeypoints(
            frame_idx=int(frame_idx),
            t_ms=int(group["t_ms"].iloc[0]),
            detected=bool(group["detected"].any()),
            joints_xy=dict(zip(joints, zip(xs, ys, strict=True), strict=True)),
            visibility=dict(zip(joints, scores, strict=True)),
            trainer_contact=(bool(group[CONTACT_COLUMN].any()) if has_contact else False),
        )
    return PanelKeypoints(label=str(label), path=path, frames=frames)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class OverlayReport:
    """What a render produced."""

    video_path: pathlib.Path
    out_path: pathlib.Path
    labels: tuple[str, ...]
    frame_count: int
    width: int
    height: int
    fps: float
    detected_frames: tuple[int, ...]
    runtime_seconds: float
    #: Per panel, how many frames the athlete selection flagged as trainer
    #: contact. Empty tuple when the panel came from a parquet without the flag.
    contact_frames: tuple[int, ...] = ()

    @property
    def no_pose_frames(self) -> tuple[int, ...]:
        """Per panel, how many frames the model found no pose on."""
        return tuple(self.frame_count - detected for detected in self.detected_frames)


def _even_size(width: int, height: int) -> tuple[int, int]:
    """Round a frame size up to even numbers, which ``mp4v`` insists on."""
    return width + width % 2, height + height % 2


class _AtomicVideoWriter:
    """Writes an MP4 next to its destination and renames it into place at the end.

    ``cv2.VideoWriter`` picks its container from the file suffix, so the
    temporary file keeps the ``.mp4`` extension; it is still hidden and only
    becomes visible under its real name once the last frame has been written.
    """

    def __init__(self, out_path: str | pathlib.Path, size: tuple[int, int], fps: float) -> None:
        self.out_path = pathlib.Path(out_path)
        self.size = size
        self.fps = float(fps)
        self.tmp_path: pathlib.Path | None = None

    def __enter__(self) -> cv2.VideoWriter:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(
            dir=self.out_path.parent,
            prefix=f".{self.out_path.stem}.",
            suffix=self.out_path.suffix or ".mp4",
        )
        os.close(handle)
        self.tmp_path = pathlib.Path(name)
        writer = cv2.VideoWriter(str(self.tmp_path), MEDIA_FOURCC, self.fps, self.size)
        if not writer.isOpened():
            writer.release()
            self.tmp_path.unlink(missing_ok=True)
            self.tmp_path = None
            raise RuntimeError(
                f"could not open an mp4v writer for {self.out_path} at "
                f"{self.size[0]}x{self.size[1]} ({self.fps:.3f} fps)"
            )
        return writer

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.tmp_path is None:
            return
        if exc_type is not None:
            self.tmp_path.unlink(missing_ok=True)
            self.tmp_path = None
            return
        os.chmod(self.tmp_path, FILE_MODE)
        self.tmp_path.replace(self.out_path)
        self.tmp_path = None


def _missing_frame_error(video_name: str, panel: PanelKeypoints, frame_idx: int) -> RuntimeError:
    return RuntimeError(
        f"{video_name}: panel {panel.label!r} has no keypoints for frame {frame_idx} "
        f"({panel.path} covers frames 0-{panel.max_frame_idx}) but the video has more "
        "frames; re-run the keypoint extraction for this video"
    )


def _frame_count_error(video_name: str, panel: PanelKeypoints, written: int) -> RuntimeError:
    difference = written - panel.frame_count
    return RuntimeError(
        f"{video_name}: panel {panel.label!r} covers {panel.frame_count} frames but the "
        f"video decoded {written} (off by {abs(difference)} frames, tolerance "
        f"{FRAME_COUNT_TOLERANCE}); {panel.path}"
    )


def render_overlay(
    video_path: str | pathlib.Path,
    panels: Sequence[tuple[str, str | pathlib.Path]],
    out_path: str | pathlib.Path,
    max_frames: int | None = None,
) -> OverlayReport:
    """Draw one or more keypoint panels on a clip and write an MP4.

    ``panels`` is one ``(label, parquet_path)`` pair per panel, in the order
    they should appear; the panels are placed side by side horizontally, all at
    the clip's display height, so two rotation modes of the same clip line up
    pixel for pixel. Every panel carries a caption with its label, the
    ``frame_idx`` and the ``t_ms``; a frame with no pose shows the plain frame
    and the extra line :data:`NO_POSE_LABEL`, and a frame the athlete selection
    flagged (``trainer_contact``) additionally gets a red border and the line
    :data:`TRAINER_CONTACT_LABEL`. The border is drawn after the caption, so it
    stays a continuous rectangle.

    Rows are matched to frames by ``frame_idx``, never by position, and a
    parquet that disagrees with the video by more than
    :data:`FRAME_COUNT_TOLERANCE` frames raises instead of drawing a pose on the
    wrong frame. The output is written to a temporary file next to ``out_path``
    and renamed once the last frame is in, so a failed run never leaves a
    half-written MP4 behind.

    ``max_frames`` renders a prefix of the clip, keeping the clip's own frame
    rate so the preview still plays at the speed it was recorded.
    """
    video_path = pathlib.Path(video_path)
    out_path = pathlib.Path(out_path)
    if not panels:
        raise ValueError("render_overlay needs at least one panel")
    if max_frames is not None and max_frames < 0:
        raise ValueError(f"max_frames must be >= 0, got {max_frames}")

    keypoint_panels = [load_panel(label, path) for label, path in panels]
    labels = tuple(panel.label for panel in keypoint_panels)

    started = time.perf_counter()
    with DisplayVideo(video_path) as video:
        width, height = _even_size(video.display_width * len(keypoint_panels), video.display_height)
        detected = [0] * len(keypoint_panels)
        contact = [0] * len(keypoint_panels)
        written = 0
        with _AtomicVideoWriter(out_path, (width, height), keypoint_panels[0].fps()) as writer:
            try:
                for packet in video:
                    if max_frames is not None and written >= max_frames:
                        break
                    keypoints = []
                    for panel in keypoint_panels:
                        frame_keys = panel.frame(packet.frame_idx)
                        if frame_keys is None:
                            raise _missing_frame_error(video_path.name, panel, packet.frame_idx)
                        keypoints.append(frame_keys)

                    tiles = []
                    for index, (panel, frame_keys) in enumerate(
                        zip(keypoint_panels, keypoints, strict=True)
                    ):
                        tile = (
                            draw_pose(packet.frame, frame_keys.joints_xy, frame_keys.visibility)
                            if frame_keys.detected
                            else packet.frame
                        )
                        if frame_keys.detected:
                            detected[index] += 1
                        lines = [
                            panel.label,
                            f"frame_idx {frame_keys.frame_idx}",
                            f"t_ms {frame_keys.t_ms}",
                        ]
                        if not frame_keys.detected:
                            lines.append(NO_POSE_LABEL)
                        if frame_keys.trainer_contact:
                            contact[index] += 1
                            lines.append(TRAINER_CONTACT_LABEL)
                        tile = draw_caption(tile, lines)
                        if frame_keys.trainer_contact:
                            tile = draw_border(tile)
                        tiles.append(tile)

                    composed = np.hstack(tiles)
                    pad_h = height - composed.shape[0]
                    pad_w = width - composed.shape[1]
                    if pad_h or pad_w:
                        composed = cv2.copyMakeBorder(
                            composed, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT
                        )
                    writer.write(composed)
                    written += 1
            finally:
                writer.release()

            # Checked while the output is still a temporary file, so a mismatch
            # never leaves a rendered MP4 behind.
            if max_frames is None:
                for panel in keypoint_panels:
                    if abs(written - panel.frame_count) > FRAME_COUNT_TOLERANCE:
                        raise _frame_count_error(video_path.name, panel, written)

    if not written:
        raise RuntimeError(f"{video_path.name}: no frames decoded")

    return OverlayReport(
        video_path=video_path,
        out_path=out_path,
        labels=labels,
        frame_count=written,
        width=width,
        height=height,
        fps=keypoint_panels[0].fps(),
        detected_frames=tuple(detected),
        runtime_seconds=time.perf_counter() - started,
        contact_frames=tuple(contact),
    )


# --------------------------------------------------------------------------- #
# Locations
# --------------------------------------------------------------------------- #


def source_dirname(source: str = DEFAULT_SOURCE) -> str:
    """Which ``keypoints/`` sub-directory a ``--source`` value reads from."""
    try:
        return SOURCE_DIRNAMES[source]
    except KeyError:
        raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}") from None


def keypoints_root(
    data: str | pathlib.Path | None = None, source: str = DEFAULT_SOURCE
) -> pathlib.Path:
    """``<data_dir>/keypoints/<source>``, where the parquets of that source live."""
    root = pathlib.Path(data) if data is not None else data_dir()
    return root / "keypoints" / source_dirname(source)


def overlays_dir(data: str | pathlib.Path | None = None) -> pathlib.Path:
    """``<data_dir>/overlays``, where the rendered MP4s go."""
    root = pathlib.Path(data) if data is not None else data_dir()
    return root / OVERLAYS_DIR_NAME


def panel_parquet_path(
    clip_id: str,
    mode: str,
    data: str | pathlib.Path | None = None,
    source: str = DEFAULT_SOURCE,
) -> pathlib.Path:
    """``<data_dir>/keypoints/<source>/<mode>/<clip_id>.parquet``."""
    return keypoints_root(data, source) / mode / f"{clip_id}.parquet"


def missing_parquet_message(
    clip_id: str,
    mode: str,
    path: str | pathlib.Path,
    video_path: str | pathlib.Path | None = None,
    source: str = DEFAULT_SOURCE,
) -> str:
    """The error shown when a panel's keypoint parquet has not been generated yet."""
    if source == DEFAULT_SOURCE:
        steps = [f"cd pipeline && uv run python -m handstand.pose_mediapipe --rotate {mode}"]
        if video_path is not None:
            steps[0] += f' --clips "{video_path}"'
    else:
        # The athlete parquet is the second step: the multi-person keypoints it
        # reads have to exist first, and that run needs its own flags.
        steps = [
            "cd pipeline && uv run python -m handstand.pose_mediapipe "
            f"--rotate {mode} --num-poses 3 --running-mode image "
            "--min-detection 0.2 --min-presence 0.2",
            f"cd pipeline && uv run python -m handstand.athlete --rotate {mode} --clips {clip_id}",
        ]
    return (
        f"no keypoints for clip {clip_id} in mode {mode!r}: {path}\n"
        "generate them first with:\n" + "\n".join(f"  {step}" for step in steps)
    )


def require_parquet(
    clip_id: str,
    mode: str,
    data: str | pathlib.Path | None = None,
    video_path: str | pathlib.Path | None = None,
    source: str = DEFAULT_SOURCE,
) -> pathlib.Path:
    """The parquet of one mode, or a :class:`FileNotFoundError` naming the command."""
    path = panel_parquet_path(clip_id, mode, data, source)
    if not path.is_file():
        raise FileNotFoundError(missing_parquet_message(clip_id, mode, path, video_path, source))
    return path


def overlay_filename(clip_id: str, labels: Sequence[str], source: str = DEFAULT_SOURCE) -> str:
    """Output file name: ``<clip_id>_<mode>.mp4``, or ``<clip_id>_none-vs-auto.mp4``.

    A non-default ``--source`` is part of the name (``<clip_id>_athlete_auto.mp4``),
    so the two sources of the same rotation mode never overwrite each other. The
    default source keeps the name it has always had.
    """
    if not labels:
        raise ValueError("an overlay needs at least one label")
    suffix = labels[0] if len(labels) == 1 else "-vs-".join(labels)
    if source != DEFAULT_SOURCE:
        return f"{clip_id}_{source}_{suffix}.mp4"
    return f"{clip_id}_{suffix}.mp4"


def video_for_clip(
    clip_id: str,
    videos: str | pathlib.Path | None = None,
    data: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """The raw video of ``clip_id``: the catalogue's filename, else the SHA-1 match.

    The videos directory is only read, never written to.
    """
    videos_root = pathlib.Path(videos) if videos is not None else videos_dir()
    if not videos_root.is_dir():
        raise FileNotFoundError(f"videos dir does not exist: {videos_root}")

    data_root = pathlib.Path(data) if data is not None else data_dir()
    catalogue = load_catalogue(data_root / "catalogue.csv")
    if catalogue:
        for filename, mapped in catalogue.items():
            if mapped != clip_id:
                continue
            candidate = videos_root / pathlib.Path(filename).name
            if candidate.is_file():
                return candidate
            raise FileNotFoundError(
                f"clip {clip_id} is {candidate.name} in {videos_root}, but that file is missing"
            )

    for candidate in sorted(videos_root.glob("*.mp4")):
        if resolve_clip_id(candidate) == clip_id:
            return candidate
    raise FileNotFoundError(f"no video for clip {clip_id} in {videos_root}")


def render_clip(
    clip_id: str,
    modes: Sequence[str],
    *,
    max_frames: int | None = None,
    videos: str | pathlib.Path | None = None,
    data: str | pathlib.Path | None = None,
    source: str = DEFAULT_SOURCE,
) -> OverlayReport:
    """Render ``<clip_id>`` with one panel per rotation mode.

    The library-level entry point behind the CLI: it resolves the raw video, the
    ``<mode>`` parquets under ``<data_dir>/keypoints/<source>/`` and the output
    path ``<data_dir>/overlays/<clip_id>_<label>.mp4``, then calls
    :func:`render_overlay`.
    """
    if not modes:
        raise ValueError("render_clip needs at least one mode")
    video_path = video_for_clip(clip_id, videos, data)
    panels = [(mode, require_parquet(clip_id, mode, data, video_path, source)) for mode in modes]
    out_path = overlays_dir(data) / overlay_filename(clip_id, list(modes), source)
    return render_overlay(video_path, panels, out_path, max_frames=max_frames)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m handstand.overlay",
        description="Render MP4s with MediaPipe keypoints drawn on the frames.",
    )
    parser.add_argument(
        "--clip",
        required=True,
        metavar="CLIP_ID",
        help="clip_id to render, as named in <data_dir>/catalogue.csv",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=ROTATE_MODES,
        default=["auto"],
        metavar="MODE",
        help="rotation modes to show, one panel each, side by side (default: auto)",
    )
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default=DEFAULT_SOURCE,
        help=(
            "which keypoints to draw: mediapipe is the plain single-person "
            "runner (default), athlete is the selection from handstand.athlete, "
            "whose trainer-contact frames get a red border"
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="render at most N frames (the clip's own frame rate is kept)",
    )
    parser.add_argument(
        "--videos",
        type=pathlib.Path,
        default=None,
        help="videos directory (default: $HANDSTAND_VIDEOS, else the workspace videos dir)",
    )
    parser.add_argument(
        "--data",
        type=pathlib.Path,
        default=None,
        help="data directory holding keypoints/ and overlays/ (default: $HANDSTAND_DATA)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames < 0:
        parser.error("--max-frames must be >= 0")

    try:
        report = render_clip(
            args.clip,
            args.modes,
            max_frames=args.max_frames,
            videos=args.videos,
            data=args.data,
            source=args.source,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"overlay: {error}", file=sys.stderr)
        return 2

    print(
        f"clip  {report.video_path.name} id={args.clip} source={args.source} "
        f"panels={'|'.join(report.labels)} frames={report.frame_count} "
        f"poses={'|'.join(str(count) for count in report.detected_frames)} "
        f"contact={'|'.join(str(count) for count in report.contact_frames)} "
        f"size={report.width}x{report.height} fps={report.fps:.2f} "
        f"runtime={report.runtime_seconds:.1f}s -> {report.out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
