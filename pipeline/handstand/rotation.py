"""Pure coordinate helpers for rotating keypoints between frame orientations.

A 180° rotation of the *frame* flips both axes, so a point at pixel ``(x, y)``
of a ``width x height`` frame lands on ``(width - 1 - x, height - 1 - y)`` of the
rotated frame — the formula MediaPipe keypoints have to be mapped back through
when the runner feeds rotated frames to the model. 90° and 270° are supported
too because they are cheap, exact and reused when comparing orientations later.

Conventions used throughout:

* Angles are degrees **counter-clockwise**, measured in the image coordinate
  system (``x`` right, ``y`` down), i.e. ``rotate_points(..., 90, ...)`` is what
  ``cv2.ROTATE_90_COUNTERCLOCKWISE`` does to the pixels.
* ``width``/``height`` always describe the frame the *input* points live in.
* Rotating by 90° or 270° swaps the frame size: the result lives in a
  ``(height, width)`` frame. Other angles keep the frame size.
* Results are ``float64`` arrays shaped ``(..., 2)``; integer inputs on the
  90° multiples come back bit-exact because those branches are integer maths.

Everything here is pure numpy — no OpenCV, no MediaPipe — so it is trivial to
test and to port to Swift later.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["inverse_rotate_points", "rotate_points", "rotated_frame_size"]

#: Angles handled by exact integer branches; anything else uses float maths.
_CARDINAL_ANGLES = (0.0, 90.0, 180.0, 270.0)


def rotated_frame_size(angle_deg: float, width: int, height: int) -> tuple[int, int]:
    """Return ``(width, height)`` of the frame produced by :func:`rotate_points`.

    90° and 270° rotations swap the axes; every other angle keeps them.
    """
    if width < 1 or height < 1:
        raise ValueError(f"frame size must be positive, got {width}x{height}")
    if float(angle_deg) % 360.0 in (90.0, 270.0):
        return int(height), int(width)
    return int(width), int(height)


def rotate_points(points: np.ndarray, angle_deg: float, width: int, height: int) -> np.ndarray:
    """Map points from a frame into the coordinate system of the rotated frame.

    Parameters
    ----------
    points:
        Array shaped ``(..., 2)`` of ``(x, y)`` points in the original frame.
    angle_deg:
        Rotation angle in degrees, counter-clockwise in image coordinates.
        90° multiples map pixel indices exactly; other angles rotate about the
        frame centre and keep the frame size.
    width, height:
        Size of the frame the points live in.

    Returns
    -------
    numpy.ndarray
        ``float64`` array shaped ``(..., 2)`` in the rotated frame, whose size
        is given by :func:`rotated_frame_size`.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 0 or pts.shape[-1] != 2:
        raise ValueError(f"points must have shape (..., 2), got {pts.shape}")
    if width < 1 or height < 1:
        raise ValueError(f"frame size must be positive, got {width}x{height}")

    x = pts[..., 0]
    y = pts[..., 1]
    angle = float(angle_deg) % 360.0

    if angle == 0.0:
        out_x, out_y = x, y
    elif angle == 90.0:
        out_x, out_y = y, float(width - 1) - x
    elif angle == 180.0:
        out_x, out_y = float(width - 1) - x, float(height - 1) - y
    elif angle == 270.0:
        out_x, out_y = float(height - 1) - y, x
    else:
        # Non-cardinal angles: rotate about the frame centre, same frame size.
        theta = math.radians(angle)
        cos, sin = math.cos(theta), math.sin(theta)
        centre_x = (width - 1) / 2.0
        centre_y = (height - 1) / 2.0
        delta_x = x - centre_x
        delta_y = y - centre_y
        out_x = centre_x + delta_x * cos + delta_y * sin
        out_y = centre_y - delta_x * sin + delta_y * cos

    return np.stack([out_x, out_y], axis=-1)


def inverse_rotate_points(
    points: np.ndarray, angle_deg: float, width: int, height: int
) -> np.ndarray:
    """Undo :func:`rotate_points`.

    ``points`` must be expressed in the rotated frame produced by
    ``rotate_points(..., angle_deg, width, height)``; the result is back in the
    original ``width x height`` frame. ``width``/``height`` always refer to the
    *original* frame, so ``inverse_rotate_points(rotate_points(p, a, w, h), a,
    w, h) == p`` for every supported angle.
    """
    angle = float(angle_deg) % 360.0
    rotated_width, rotated_height = rotated_frame_size(angle, width, height)
    return rotate_points(points, -angle, rotated_width, rotated_height)
