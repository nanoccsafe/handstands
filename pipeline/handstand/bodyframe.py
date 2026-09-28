"""The body frame: the coordinate system every later measurement is taken in.

Raw keypoints are in display pixels, so the same movement is a different number
of pixels in every clip, and a pixel is not a body part. Two things fix that at
once, and they are what this module is:

* the **origin** is the *wrist midpoint* — where a handstand's hands meet the
  floor, and the point the whole body is measured from, because that is the only
  part of an inverted athlete that stays put;
* the **unit** is the clip's body length ``L`` (:func:`handstand.postprocess.estimate_body_length`),
  so a 500 px tall athlete and a 350 px one are measured in the same numbers.

The axes are the display frame's, with one deliberate change: ``u`` runs to the
right, like the image's ``x``, and ``v`` runs **up**, which is the image's ``-y``.
A handstand is upside down, so an unflipped ``v`` would say the athlete's feet
are *below* their hands. Everything downstream — phases, features, centre of
mass — works in ``(u, v)`` with ``v`` up, so a positive ``v`` always means
"further from the floor".

Nothing here knows about frames, joints or parquet files: it is four small pure
functions over numbers, so the Swift port can mirror it exactly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = [
    "AXIS_U",
    "AXIS_V",
    "ORIGIN",
    "UNIT",
    "body_frame_points",
    "midpoint",
    "to_body_frame",
    "wrist_midpoint",
]

#: Where the origin is, in words, for the docstrings and the later reports.
ORIGIN = "the midpoint of the two wrists"
#: The horizontal axis, in words: rightwards, like the display frame's ``x``.
AXIS_U = "to the right, in display-frame x"
#: The vertical axis, in words: upwards, i.e. display-frame ``y`` with the sign
#: flipped, so a positive ``v`` is further from the floor.
AXIS_V = "up the image, i.e. display-frame y with the sign flipped"
#: The unit, in words: the clip's body length ``L``.
UNIT = "the clip's body length L"


def _checked_length(body_length: float) -> float:
    """``body_length`` as a usable scale: finite and above zero.

    A zero or negative ``L`` would make every ``(u, v)`` infinite, and a NaN one
    would make them NaN — both silently, deep inside a later stage. Refusing it
    here means the mistake is reported by the stage that made it.
    """
    length = float(body_length)
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError(f"body_length must be a positive, finite number, got {body_length!r}")
    return length


def _point(value: Sequence[float]) -> tuple[float, float]:
    """One ``(x, y)`` as a tuple of floats, or raise."""
    pair = np.asarray(value, dtype=np.float64)
    if pair.ndim != 1 or pair.shape[0] != 2:
        raise ValueError(f"a point is (x, y), got {value!r}")
    return float(pair[0]), float(pair[1])


def midpoint(first: Sequence[float], second: Sequence[float]) -> tuple[float, float]:
    """The mean of two ``(x, y)`` points, as display-frame pixels.

    Used for the shoulder and hip midpoints the body length is built from. A
    value that is not a pair, or holds something other than numbers, raises.
    """
    first_x, first_y = _point(first)
    second_x, second_y = _point(second)
    return (first_x + second_x) / 2.0, (first_y + second_y) / 2.0


def wrist_midpoint(
    left_wrist: Sequence[float], right_wrist: Sequence[float]
) -> tuple[float, float]:
    """The origin of the body frame: the mean of the two wrist positions.

    Both wrists are averaged whether they are level or not, so the origin does
    not swing left and right as the athlete shifts their hands — a handstand's
    support point is a place, not a side.
    """
    left_x, left_y = _point(left_wrist)
    right_x, right_y = _point(right_wrist)
    return (left_x + right_x) / 2.0, (left_y + right_y) / 2.0


def to_body_frame(
    x: float | np.ndarray,
    y: float | np.ndarray,
    wrist_mid_x: float,
    wrist_mid_y: float,
    body_length: float,
) -> tuple[float, float] | np.ndarray:
    """One point in the body frame: ``(u, v)``, from :data:`ORIGIN`, in units of ``L``.

    ``u`` is ``x - wrist_mid_x`` and ``v`` is ``wrist_mid_y - y``, both divided by
    ``body_length`` — see :data:`AXIS_U` and :data:`AXIS_V`. So the wrist
    midpoint itself is ``(0, 0)``, a point one body length above it is
    ``(0, 1)`` and a point one body length to the right of it is ``(1, 0)``.

    Scalars in give a ``(u, v)`` tuple of floats out. Arrays in give a
    ``(..., 2)`` array out, broadcast against each other, which is how a whole
    clip's points are converted at once. ``body_length`` must be positive and
    finite (:func:`_checked_length`).
    """
    scale = _checked_length(body_length)
    origin_x = float(wrist_mid_x)
    origin_y = float(wrist_mid_y)
    u = (np.asarray(x, dtype=np.float64) - origin_x) / scale
    v = (origin_y - np.asarray(y, dtype=np.float64)) / scale
    if u.ndim == 0 and v.ndim == 0:
        return (float(u), float(v))
    u, v = np.broadcast_arrays(u, v)
    return np.stack((u, v), axis=-1)


def body_frame_points(
    points: np.ndarray,
    wrist_mid: Sequence[float],
    body_length: float,
) -> np.ndarray:
    """``(..., 2)`` display-frame points as ``(..., 2)`` body-frame points.

    :func:`to_body_frame` one coordinate at a time, for the case where the
    points already sit in an array — one clip's joints, or a trajectory.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 0 or pts.shape[-1] != 2:
        raise ValueError(f"points must have shape (..., 2), got {pts.shape}")
    origin = _point(wrist_mid)
    return to_body_frame(pts[..., 0], pts[..., 1], origin[0], origin[1], body_length)  # type: ignore[return-value]
