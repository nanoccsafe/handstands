"""Tests for :mod:`handstand.rotation`: pure coordinate rotation helpers.

No videos, no models — just known points and exact expectations.
"""

from __future__ import annotations

import numpy as np
import pytest

from handstand.rotation import inverse_rotate_points, rotate_points, rotated_frame_size

# A 4x3 frame: width=4 (x in 0..3), height=3 (y in 0..2).
WIDTH = 4
HEIGHT = 3
POINTS = np.array(
    [
        [0, 0],  # top-left
        [3, 0],  # top-right
        [0, 2],  # bottom-left
        [3, 2],  # bottom-right
        [1, 1],  # interior
    ],
    dtype=np.float64,
)


def test_180_is_the_index_flip_formula() -> None:
    """x = W - 1 - x_r, y = H - 1 - y_r on known points."""
    rotated = rotate_points(POINTS, 180, WIDTH, HEIGHT)
    expected = np.array(
        [
            [WIDTH - 1 - 0, HEIGHT - 1 - 0],
            [WIDTH - 1 - 3, HEIGHT - 1 - 0],
            [WIDTH - 1 - 0, HEIGHT - 1 - 2],
            [WIDTH - 1 - 3, HEIGHT - 1 - 2],
            [WIDTH - 1 - 1, HEIGHT - 1 - 1],
        ],
        dtype=np.float64,
    )
    assert np.array_equal(rotated, expected)
    assert np.array_equal(rotated, np.array([[3, 2], [0, 2], [3, 0], [0, 0], [2, 1]]))
    assert np.array_equal(inverse_rotate_points(rotated, 180, WIDTH, HEIGHT), POINTS)


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_cardinal_round_trips(angle: int) -> None:
    rng = np.random.default_rng(angle)
    points = rng.uniform(0, 1, size=(50, 2)) * np.array([[WIDTH - 1, HEIGHT - 1]])
    rotated_size = rotated_frame_size(angle, WIDTH, HEIGHT)

    rotated = rotate_points(points, angle, WIDTH, HEIGHT)
    assert rotated.shape == points.shape
    # The frame size swaps for 90/270; the point array itself stays (N, 2).
    assert rotated_frame_size(angle, WIDTH, HEIGHT) == (
        WIDTH if angle in (0, 180) else HEIGHT,
        HEIGHT if angle in (0, 180) else WIDTH,
    )

    back = inverse_rotate_points(rotated, angle, WIDTH, HEIGHT)
    assert np.allclose(back, points, atol=1e-9)

    # Rotating twice equals one 2*angle rotation (frame size swapped in between).
    twice = rotate_points(rotated, angle, *rotated_size)
    assert np.allclose(twice, rotate_points(points, (2 * angle) % 360, WIDTH, HEIGHT), atol=1e-9)


def test_90_counter_clockwise_maps_known_points() -> None:
    """90° CCW is what cv2.ROTATE_90_COUNTERCLOCKWISE does to pixels."""
    rotated = rotate_points(np.array([[0.0, 0.0], [3.0, 0.0], [1.0, 1.0]]), 90, WIDTH, HEIGHT)
    # (x, y) -> (y, W - 1 - x) in the (HEIGHT, WIDTH) rotated frame.
    assert np.array_equal(rotated, np.array([[0.0, 3.0], [0.0, 0.0], [1.0, 2.0]]))
    assert rotated_frame_size(90, WIDTH, HEIGHT) == (HEIGHT, WIDTH)


def test_270_counter_clockwise_maps_known_points() -> None:
    rotated = rotate_points(np.array([[0.0, 0.0], [0.0, 2.0], [1.0, 1.0]]), 270, WIDTH, HEIGHT)
    # (x, y) -> (H - 1 - y, x) in the (HEIGHT, WIDTH) rotated frame.
    assert np.array_equal(rotated, np.array([[2.0, 0.0], [0.0, 0.0], [1.0, 1.0]]))
    assert rotated_frame_size(270, WIDTH, HEIGHT) == (HEIGHT, WIDTH)


def test_90_and_270_undo_each_other() -> None:
    points = POINTS.copy()
    assert np.array_equal(
        rotate_points(rotate_points(points, 90, WIDTH, HEIGHT), 270, HEIGHT, WIDTH), points
    )
    assert np.array_equal(
        rotate_points(rotate_points(points, 270, WIDTH, HEIGHT), 90, HEIGHT, WIDTH), points
    )
    # The explicit inverse helper is exactly the opposite angle.
    assert np.array_equal(
        inverse_rotate_points(points, 90, WIDTH, HEIGHT),
        rotate_points(points, 270, HEIGHT, WIDTH),
    )
    assert np.array_equal(
        inverse_rotate_points(points, 270, WIDTH, HEIGHT),
        rotate_points(points, 90, HEIGHT, WIDTH),
    )


def test_zero_rotation_is_identity_and_keeps_size() -> None:
    assert np.array_equal(rotate_points(POINTS, 0, WIDTH, HEIGHT), POINTS)
    assert rotated_frame_size(0, WIDTH, HEIGHT) == (WIDTH, HEIGHT)
    assert rotated_frame_size(180, WIDTH, HEIGHT) == (WIDTH, HEIGHT)


def test_negative_and_large_angles_normalise() -> None:
    assert np.array_equal(
        rotate_points(POINTS, -180, WIDTH, HEIGHT), rotate_points(POINTS, 180, WIDTH, HEIGHT)
    )
    assert np.array_equal(
        rotate_points(POINTS, 450, WIDTH, HEIGHT), rotate_points(POINTS, 90, WIDTH, HEIGHT)
    )
    assert np.array_equal(
        inverse_rotate_points(POINTS, -90, WIDTH, HEIGHT),
        inverse_rotate_points(POINTS, 270, WIDTH, HEIGHT),
    )


def test_non_cardinal_angle_rotates_about_the_centre_and_round_trips() -> None:
    points = np.array([[0.0, 0.0], [3.0, 2.0], [1.5, 1.0]])
    angle = 45.0
    rotated = rotate_points(points, angle, WIDTH, HEIGHT)
    assert rotated_frame_size(angle, WIDTH, HEIGHT) == (WIDTH, HEIGHT)

    centre = np.array([[(WIDTH - 1) / 2, (HEIGHT - 1) / 2]])
    assert np.allclose(rotate_points(centre, angle, WIDTH, HEIGHT), centre, atol=1e-12)
    assert not np.allclose(rotated, points, atol=1e-6)
    assert np.allclose(inverse_rotate_points(rotated, angle, WIDTH, HEIGHT), points, atol=1e-9)

    # Four eighth-turns are a half turn, i.e. the exact 180° branch.
    four_times = rotate_points(rotated, angle, WIDTH, HEIGHT)
    for _ in range(2):
        four_times = rotate_points(four_times, angle, WIDTH, HEIGHT)
    assert np.allclose(four_times, rotate_points(points, 180, WIDTH, HEIGHT), atol=1e-9)


def test_accepts_python_lists_and_keeps_float64() -> None:
    out = rotate_points([[0, 0], [3, 2]], 180, WIDTH, HEIGHT)
    assert out.dtype == np.float64
    assert np.array_equal(out, [[3.0, 2.0], [0.0, 0.0]])
    single = rotate_points(np.array([1.0, 1.0]), 180, WIDTH, HEIGHT)
    assert single.shape == (2,)
    assert np.array_equal(single, [2.0, 1.0])


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="shape"):
        rotate_points(np.zeros((3, 3)), 180, WIDTH, HEIGHT)
    with pytest.raises(ValueError, match="shape"):
        rotate_points(np.zeros((3,)), 180, WIDTH, HEIGHT)
    with pytest.raises(ValueError, match="positive"):
        rotate_points(POINTS, 180, 0, HEIGHT)
    with pytest.raises(ValueError, match="positive"):
        rotated_frame_size(90, WIDTH, 0)
