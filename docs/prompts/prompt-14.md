# Task: chainlink issue #14 — MediaPipe pose runner with optional 180° rotation

Batch-run MediaPipe Pose Landmarker (Full) over handstand videos and save per-frame keypoints. Handstand bodies
are upside down, and pose models do worse on inverted bodies, so the runner can rotate frames 180° before
inference and must map keypoints back to the un-rotated frame.

## Where
- `pipeline/handstand/pose_mediapipe.py` (library + CLI: `uv run python -m handstand.pose_mediapipe`).
- `pipeline/handstand/rotation.py` for pure coordinate helpers (easy to test, reused later).
- `pipeline/scripts/download_models.py` downloads
  `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task`
  into `pipeline/models/` (already git-ignored). Skip if present.
- `docs/keypoint_schema.md` documents the output format (the Apple Vision runner will match it later).
- Tests in `pipeline/tests/test_rotation.py` and `pipeline/tests/test_pose_mediapipe.py`.

## Environment facts (verified)
- mediapipe 1.0.1 is installed; the API is `mediapipe.tasks.python.vision.PoseLandmarker`,
  `PoseLandmarkerOptions`, `RunningMode.VIDEO`. Check signatures in the installed package source
  (`.venv/lib/python3.12/site-packages/mediapipe/tasks/python/vision/pose_landmarker.py`) rather than
  guessing from older tutorials.
- Videos are H.264 1024x576 with rotation metadata -90 (portrait shown sideways) and a VARIABLE frame rate.
  Decode with OpenCV and make sure frames are in DISPLAY orientation (upright as a person would view it).
  Verified with the installed cv2 5.0.0: `CAP_PROP_ORIENTATION_AUTO` defaults to 1, so `cap.read()` already
  returns upright 1024x576 -> (1024, 576, 3) frames (H=1024, W=576) and CAP_PROP_FRAME_WIDTH/HEIGHT report
  576/1024. Rely on that, but assert the decoded frame shape matches the display dims so a future OpenCV change
  fails loudly.
- Timestamps: use `CAP_PROP_POS_MSEC` per frame (not frame_idx / fps). MediaPipe VIDEO mode requires strictly
  increasing integer ms; if two frames share a timestamp, bump by 1 ms.

## Rotation modes (`--rotate none|180|auto`)
- `none`: feed frames as displayed.
- `180`: rotate every frame 180° before inference.
- `auto`: rotate 180° when the PREVIOUS frame's result was inverted: mean y of both wrists greater
  (lower in the image) than mean y of both ankles, using the display-frame coordinates. First frame: not rotated.
  Because MediaPipe VIDEO mode tracks between frames, use TWO landmarker instances in auto mode, one that only
  ever sees upright frames and one that only sees rotated frames, so each tracker sees a consistent orientation.
- Mapping back from a 180° rotated frame of size (W, H): `x = W - 1 - x_r`, `y = H - 1 - y_r` in pixels
  (convert MediaPipe's normalized coords to pixels first). Put this in `rotation.py` with a general
  `rotate_points(points, angle_deg, width, height)` + inverse, and unit-test round trips for 0/90/180/270.

## Output
`<data_dir>/keypoints/mediapipe/<rotate>/<clip_id>.parquet`, long format, one row per (frame, joint):
`frame_idx:int, t_ms:int, joint:str, x:float, y:float, z:float, visibility:float, presence:float, rotated:bool,
detected:bool`
- `x, y` in DISPLAY-frame pixels (never rotated coords). Frames with no detection still get rows with
  NaN coords and `detected=false`, so every frame is represented.
- `joint` uses the 33 MediaPipe landmark names in snake_case (nose, left_shoulder, ..., right_foot_index).
- `clip_id`: use `data_dir()/catalogue.csv` if it exists (columns `clip_id, filename`); otherwise compute the
  first 12 hex chars of the SHA-1 of the file bytes (same definition as the catalogue).
- Also write `<clip_id>.json` next to it: model file name, rotate mode, video display size, frame count,
  mediapipe version, runtime seconds.
CLI: `--rotate`, `--clips FILE...` (default: all mp4 in videos_dir), `--limit N`, `--overwrite`.
Skip clips whose parquet exists unless `--overwrite`.

## Tests (must not use the real videos)
- rotation round trips and the 180° formula on known points.
- the auto-mode inversion rule on synthetic landmark arrays.
- schema test: generate a 20-frame synthetic video with ffmpeg testsrc into tmp_path, run the runner with a
  STUB landmarker (inject a fake detector object) and check parquet columns, dtypes, one row per frame x joint,
  NaN rows for no-detection frames. Do not require the real .task model in tests.

## Acceptance
- `cd pipeline && uv run pytest && uv run ruff check` pass.
- Download the model and run for real on 3 clips with each of `none` and `auto` (`--limit 3`). Report per clip:
  frames, % frames detected, % frames rotated (auto), runtime. Do not commit outputs or the model.

Follow AGENTS.md.
