"""Weekly ingest: inbox -> catalogue -> full pipeline -> hold-shape review queue.

The user exports about ten clips a week (WhatsApp) and drops them into
``videos/inbox/``. ONE command catalogues every new clip, runs the whole
pipeline over exactly those clips and queues every new HOLD for the shape
review in Label Studio (#86)::

    cd pipeline
    uv run python -m handstand.ingest --dry-run    # look first: writes NOTHING
    uv run python -m handstand.ingest              # do it

The rules that matter, in order:

* Only ``*.mp4`` files in the inbox are clips; anything else (a WhatsApp
  ``.jpeg``) is reported and left alone.
* Identity is the **content** hash (:func:`handstand.catalogue.file_clip_id`),
  never the file name: a ``(1)`` suffix does not mean "duplicate". A file whose
  clip id is already in ``catalogue.csv`` is reported and stays in the inbox.
* A WhatsApp ``(N)`` suffix is renamed as the file moves into ``videos/``:
  ``... PM(1).mp4`` becomes ``... PM take2.mp4`` (#86's binding rule, so nobody
  mistakes the two 12.47.45 clips for copies of each other). Nothing is ever
  overwritten, deleted or skipped because of a name: a clash with different
  content gets the smallest free `` (ingest N)`` suffix instead.
* The catalogue only ever gains appended rows, written atomically; every
  existing byte — rows, columns, hand-typed annotations — is kept as it is.
* The pipeline runs the six steps in a fixed order, new clips only. One clip
  failing a step is recorded and the other clips continue; every step skips the
  work a previous run already did, so a re-run picks up where it failed.
* ``hold_shapes.prelabel(clips=..., skip_existing=True)`` queues only NEW
  holds: a hold that already has an image or a reviewed row in
  ``hold_shapes.csv`` is never touched again.

CLI flags: ``--dry-run`` (print all of it, write nothing), ``--keep`` (copy the
files instead of moving them), ``--label-studio`` (import the new tasks into
the ``handstand-hold-shapes`` project through the Label Studio API instead of
only writing the import JSON) and ``--sample-frames N`` (additionally sample N
keypoint-labelling frames per new clip into a separate import file).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from handstand import (
    athlete,
    features,
    frame_sampler,
    hold_shapes,
    overlay,
    phases,
    pose_mediapipe,
    postprocess,
)
from handstand import catalogue as catalogue_module
from handstand.labels import load_manifest
from handstand.paths import data_dir, videos_dir
from handstand.prelabel import make_model as make_pose_model
from handstand.prelabel import prelabel as prelabel_keypoints
from handstand.trainer_report import DETECTOR_SETTINGS, NUM_POSES

__all__ = [
    "DEFAULT_LS_URL",
    "INBOX_DIRNAME",
    "PROJECT_TITLE",
    "REPORTS_DIRNAME",
    "STEPS",
    "HttpCall",
    "IngestReport",
    "LabelStudioClient",
    "LabelStudioError",
    "NewClip",
    "PipelineReport",
    "StepRunner",
    "build_arg_parser",
    "build_runners",
    "main",
    "pick_evenly",
    "plan_inbox",
    "read_ls_credentials",
    "run_ingest",
    "run_pipeline",
    "update_hold_summary",
    "urllib_http",
    "whatsapp_target",
]

#: Where the week's exports are dropped, under ``videos_dir()``. Created on the
#: first real run; a dry run never creates it.
INBOX_DIRNAME = "inbox"
#: A clip in the inbox: this suffix, compared case-insensitively.
CLIP_SUFFIX = ".mp4"

#: A WhatsApp duplicate suffix at the end of the stem: ``(1)``, `` (2)`` … —
#: it appears when several videos are exported quickly, and says nothing about
#: the content (#86's binding rule: duplicates are decided by content only).
WHATSAPP_SUFFIX_RE = re.compile(r" ?\((?P<n>\d+)\)$")

#: The pipeline steps, in the order every new clip runs through them.
STEPS: tuple[str, ...] = (
    "pose_mediapipe",
    "athlete",
    "postprocess",
    "phases",
    "features",
    "prelabel",
)

#: Rotation mode of the whole chain: the recommended one, the mode
#: ``data/keypoints/mediapipe_multi/best`` was made with.
ROTATE = athlete.DEFAULT_ROTATE
#: Which pose model's outputs the post-processing chain reads.
SOURCE = features.DEFAULT_SOURCE
#: Detector settings and people per frame of the multi-person run the athlete
#: input was made with (recorded in the sidecars under
#: ``data/keypoints/mediapipe_multi/best/``), shared with the trainer report.
#: ``rotate best`` plus these three people in IMAGE mode is what every existing
#: ``mediapipe_athlete`` parquet was produced with.
POSE_SETTINGS = DETECTOR_SETTINGS

#: The Label Studio project the new hold tasks are imported into (#86).
PROJECT_TITLE = "handstand-hold-shapes"
#: Where Label Studio is expected to run; overridable on the command line.
DEFAULT_LS_URL = "http://localhost:8080"
#: Credentials file for the API import: ``LS_USER=…`` / ``LS_PASSWORD=…``.
#: Read only when ``--label-studio`` is given, never printed, never committed.
LS_ENV_PATH = pathlib.Path.home() / ".config" / "handstand" / "label-studio.env"

#: Where this command writes its report and its import files, under ``data/``.
REPORTS_DIRNAME = "reports"
INGEST_DIRNAME = "ingest"
#: The catalogue under ``<data>/``.
CATALOGUE_NAME = frame_sampler.CATALOGUE_NAME

#: One step of the pipeline: takes the clip ids to process and returns
#: ``{clip_id: error message}`` for the ones that failed.
StepRunner = Callable[[Sequence[str]], Mapping[str, str]]
#: The HTTP call the Label Studio client is built on, so tests can mock it:
#: ``(method, url, body, headers) -> (status, response headers, payload)``.
HttpCall = Callable[
    [str, str, bytes | None, Mapping[str, str]], tuple[int, Sequence[tuple[str, str]], bytes]
]


# --------------------------------------------------------------------------- #
# Planning: what is in the inbox, what is a duplicate, what it will be called
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class NewClip:
    """One inbox file that is new by content and where it will end up."""

    #: The file in the inbox, before the move.
    source: pathlib.Path
    #: The content hash — the clip's identity, never inferred from the name.
    clip_id: str
    #: The name it gets in ``videos/`` after the ``(N)`` rename rule.
    name: str
    #: The full target path (``videos / name``), resolved at plan time.
    path: pathlib.Path

    @property
    def renamed(self) -> bool:
        """Did the move change the file's name (rename rule or name clash)?"""
        return self.name != self.source.name


@dataclasses.dataclass(frozen=True)
class Plan:
    """What the inbox holds before anything is touched."""

    inbox: pathlib.Path
    videos: pathlib.Path
    catalogue_path: pathlib.Path
    #: New clips, in inbox order (sorted by name).
    new: list[NewClip]
    #: ``(file, clip_id, reason)`` — content already known, left in the inbox.
    duplicates: list[tuple[pathlib.Path, str, str]]
    #: Everything in the inbox that is not an ``.mp4`` clip; left alone.
    others: list[pathlib.Path]
    #: ``clip_id ->`` the first inbox file with that content, for the reasons.
    seen: dict[str, pathlib.Path]


def whatsapp_target(name: str) -> str:
    """The name a file gets as it moves into ``videos/`` (#86's rename rule).

    A WhatsApp ``(N)`` suffix becomes `` take{N+1}``, with or without a space
    before the parenthesis: ``… 12.47.45 PM(1).mp4`` becomes
    ``… 12.47.45 PM take2.mp4`` and ``… PM (1).mp4`` becomes the same thing.
    A name without a suffix keeps itself — it is take 1 implicitly.
    """
    stem, suffix = os.path.splitext(name)
    match = WHATSAPP_SUFFIX_RE.search(stem)
    if match is None:
        return name
    take = int(match["n"]) + 1
    return f"{stem[: match.start()]} take{take}{suffix}"


def _free_name(directory: pathlib.Path, name: str) -> str:
    """The smallest `` (ingest N)`` suffix whose target does not exist yet.

    Used only when the plain name is taken by a DIFFERENT file; the unsuffixed
    name is tried first, then `` (ingest 1)``, `` (ingest 2)`` … Nothing is
    ever overwritten or skipped because of the name.
    """
    stem, suffix = os.path.splitext(name)
    candidate = name
    index = 1
    while (directory / candidate).exists():
        candidate = f"{stem} (ingest {index}){suffix}"
        index += 1
    return candidate


def scan_inbox(inbox: pathlib.Path) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    """The inbox split into ``(clips, other entries)``, both sorted by name.

    A clip is a **file** whose suffix is ``.mp4`` (case-insensitive). Everything
    else — the WhatsApp ``.jpeg`` stills, a stray sub-directory — is reported
    and left exactly where it is.
    """
    if not inbox.is_dir():
        return [], []
    clips: list[pathlib.Path] = []
    others: list[pathlib.Path] = []
    for path in sorted(inbox.iterdir()):
        if path.is_file() and path.suffix.lower() == CLIP_SUFFIX:
            clips.append(path)
        else:
            others.append(path)
    return clips, others


def plan_inbox(
    inbox: pathlib.Path,
    videos: pathlib.Path,
    catalogue_path: pathlib.Path,
) -> Plan:
    """Decide what every inbox file is, by content, without touching anything.

    A file whose clip id is already in the catalogue is a duplicate (reason:
    "already in the catalogue"); two identical files in this inbox are one clip
    and one duplicate (reason: "identical to … in this inbox"). The rename rule
    then runs, and a target name that exists with DIFFERENT content is moved
    aside to the smallest free `` (ingest N)``. The reason never mentions the
    file name: only the SHA-1 decides.
    """
    clips, others = scan_inbox(inbox)
    known = {
        row.get("clip_id") or ""
        for row in catalogue_module.read_rows(catalogue_path)
        if row.get("clip_id")
    }
    seen: dict[str, pathlib.Path] = {}
    new: list[NewClip] = []
    duplicates: list[tuple[pathlib.Path, str, str]] = []
    for path in clips:
        clip_id = catalogue_module.file_clip_id(path)
        if clip_id in known:
            earlier = seen.get(clip_id)
            if earlier is not None:
                reason = f"identical to {earlier.name} in this inbox"
            else:
                reason = "already in the catalogue"
            duplicates.append((path, clip_id, reason))
            continue
        name = whatsapp_target(path.name)
        destination = videos / name
        if destination.exists():
            if catalogue_module.file_clip_id(destination) == clip_id:
                duplicates.append((path, clip_id, f"identical to {name} in videos/"))
                continue
            name = _free_name(videos, name)
            destination = videos / name
        seen[clip_id] = path
        known.add(clip_id)
        new.append(NewClip(source=path, clip_id=clip_id, name=name, path=destination))
    return Plan(
        inbox=inbox,
        videos=videos,
        catalogue_path=catalogue_path,
        new=new,
        duplicates=duplicates,
        others=others,
        seen=seen,
    )


# --------------------------------------------------------------------------- #
# Placing the files and appending to the catalogue
# --------------------------------------------------------------------------- #


def place_clip(clip: NewClip, *, keep: bool = False) -> pathlib.Path:
    """Move (or, with ``keep``, copy) one inbox file into ``videos/``.

    The target is re-checked at move time: if it exists — a file appeared since
    the plan, or two inbox files wanted the same name — the smallest free
    `` (ingest N)`` is chosen then. Never overwrite, never delete, never skip:
    the returned path is where the file actually landed.
    """
    target = clip.path
    if target.exists():
        target = target.with_name(_free_name(target.parent, target.name))
    target.parent.mkdir(parents=True, exist_ok=True)
    if keep:
        shutil.copy2(clip.source, target)
    else:
        shutil.move(str(clip.source), str(target))
    return target


def _identity_row(path: pathlib.Path) -> catalogue_module.Row:
    """The row the filesystem alone can fill in, for an unreadable video."""
    stamp = catalogue_module.parse_filename(path.name)
    row: dict[str, str] = {
        "clip_id": catalogue_module.file_clip_id(path),
        "filename": path.name,
        "session_date": stamp.session_date,
        "recorded_at": stamp.recorded_at,
    }
    return catalogue_module.normalise_row(row)


def _serialise_rows(path: pathlib.Path, rows: Sequence[Mapping[str, str]]) -> str:
    """The text to append to ``path``: its header, then the new rows.

    The existing bytes are copied through untouched — the file's own header and
    line endings are what the rows are written against — so appending cannot
    reformat what a human already typed.
    """
    columns = list(catalogue_module.COLUMNS)
    prefix = ""
    if path.is_file() and path.stat().st_size > 0:
        # newline="" keeps the file's own line endings byte-for-byte.
        with path.open(encoding="utf-8", newline="") as handle:
            prefix = handle.read()
        header = next(csv.reader(io.StringIO(prefix)), None)
        if header:
            columns = header
        if not prefix.endswith("\n"):
            prefix += "\n"
    else:
        buffer = io.StringIO()
        csv.writer(buffer, lineterminator="\n").writerow(columns)
        prefix = buffer.getvalue()
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return prefix + buffer.getvalue()


def _atomic_write(path: pathlib.Path, text: str) -> None:
    """Write ``text`` to ``path`` via a temp file beside it, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temp_path = pathlib.Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_path, catalogue_module.FILE_MODE)
        temp_path.replace(path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def append_catalogue_rows(path: pathlib.Path, rows: Sequence[Mapping[str, str]]) -> int:
    """Append rows to the catalogue, keeping every existing byte byte-for-byte.

    The whole file is written atomically (temp file plus rename), so an
    interrupted run cannot damage it — and because only new text is added, the
    hand-typed annotations of the existing rows are untouched by construction.
    Returns how many rows were appended; nothing is written for an empty list.
    """
    if not rows:
        return 0
    normalised = [catalogue_module.normalise_row(row) for row in rows]
    _atomic_write(path, _serialise_rows(path, normalised))
    return len(normalised)


def catalogue_new_clips(
    clips: Mapping[str, pathlib.Path],
    catalogue_path: pathlib.Path,
    probe: catalogue_module.Probe | None = None,
) -> tuple[int, dict[str, str]]:
    """Build and append one catalogue row per placed clip.

    ``probe`` is ffprobe by default (tests inject a stub instead of shelling
    out). A clip ffprobe cannot read still gets its identity row — filename,
    hash and timestamp — and is reported as a catalogue failure, so a bad file
    can never vanish from the sheet or block the pipeline behind it.
    """
    rows: list[catalogue_module.Row] = []
    failures: dict[str, str] = {}
    for clip_id, path in clips.items():
        try:
            rows.append(catalogue_module.build_row(path, probe))
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            failures[clip_id] = f"{type(error).__name__}: {error}"
            rows.append(_identity_row(path))
    added = append_catalogue_rows(catalogue_path, rows)
    return added, failures


def update_hold_summary(path: pathlib.Path, table: Any) -> pathlib.Path:
    """Add this run's holds to ``hold_summary.csv`` without dropping the rest.

    ``table`` is :func:`handstand.features.hold_summary_table`'s output for the
    clips this run measured: the rows of *those* clips are replaced and every
    other clip's rows survive, which is what makes the weekly run safe next to
    the 180 clips already in the file.
    """
    return features.write_hold_summary(path, table)


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class PipelineReport:
    """What one pipeline run produced: which clips made it, which failed."""

    order: tuple[str, ...]
    #: ``step -> {clip_id: error}`` — one clip failing a step is recorded here
    #: and that clip simply stops; the other clips continue through every step.
    failures: dict[str, dict[str, str]]
    #: Clips that completed every step.
    done: list[str]
    #: Clips the pipeline was started with (the placed, catalogued new clips).
    attempted: list[str]

    @property
    def failed(self) -> dict[str, str]:
        """``clip_id ->`` the step that failed it (the first one)."""
        found: dict[str, str] = {}
        for step, per_clip in self.failures.items():
            for clip_id in per_clip:
                found.setdefault(clip_id, step)
        return found


def _per_clip(call: Callable[[str], Any]) -> StepRunner:
    """Wrap one clip-level call into a step runner that never raises.

    A clip that raises is recorded with its error and the loop moves on: one
    undecodable video must not cost the other 22 clips their run (#86).
    """

    def run(clips: Sequence[str]) -> Mapping[str, str]:
        failed: dict[str, str] = {}
        for clip_id in clips:
            try:
                call(clip_id)
            except Exception as error:  # one clip must not end the batch
                failed[clip_id] = f"{type(error).__name__}: {error}"
        return failed

    return run


@dataclasses.dataclass
class StepSet:
    """The runners of one ingest run, plus what the pre-label step produced."""

    runners: dict[str, StepRunner]
    #: Filled by the real ``prelabel`` runner; ``None`` when it was stubbed.
    prelabel_report: hold_shapes.PrelabelReport | None = None
    #: The features reports this run measured (for the hold-summary update).
    feature_reports: list[Any] = dataclasses.field(default_factory=list)


def build_runners(
    root: pathlib.Path,
    videos: pathlib.Path,
    *,
    source: str = SOURCE,
    rotate: str = ROTATE,
    model_path: str | pathlib.Path = pose_mediapipe.DEFAULT_MODEL_PATH,
) -> StepSet:
    """The six steps of :data:`STEPS`, wired to the modules' library functions.

    Mirrors the six CLIs exactly — ``pose_mediapipe`` with the settings the
    existing athlete input was made with (:data:`NUM_POSES`,
    :data:`DETECTOR_SETTINGS`, ``rotate best``), then ``athlete`` with its
    trainer/contact flagging, ``postprocess``, ``phases``, ``features`` (which
    also updates ``hold_summary.csv`` for these clips only) and finally
    ``hold_shapes.prelabel`` for just these clips, skipping the holds that are
    already queued or reviewed. Each per-clip step skips work already done, so
    a re-run after a failure picks up where it stopped.
    """
    multi_root = root / "keypoints" / pose_mediapipe.output_dirname(NUM_POSES)
    athlete_root = root / "keypoints" / athlete.output_dirname(source)
    processed_root = postprocess.output_dir(root, source)
    phases_root = phases.output_dir(root, source)
    features_root = features.output_dir(root, source)
    steps = StepSet(runners={})

    def pose(clip_id: str) -> None:
        video_path = overlay.video_for_clip(clip_id, videos, root)
        pose_mediapipe.run_clip(
            video_path,
            clip_id=clip_id,
            rotate_mode=rotate,
            out_root=multi_root,
            landmarker_factory=pose_mediapipe.make_landmarker_factory(
                model_path, NUM_POSES, POSE_SETTINGS
            ),
            num_poses=NUM_POSES,
            settings=POSE_SETTINGS,
        )

    def athlete_step(clip_id: str) -> None:
        athlete.run_clip(
            clip_id,
            rotate_mode=rotate,
            in_root=root / "keypoints" / athlete.input_dirname(source),
            out_root=athlete_root,
            source=source,
        )

    def postprocess_step(clip_id: str) -> None:
        postprocess.run_clip(
            clip_id,
            rotate_mode=rotate,
            in_root=athlete_root,
            out_root=processed_root,
            source=source,
        )

    def phases_step(clip_id: str) -> None:
        phases.run_clip(
            clip_id,
            in_root=processed_root,
            out_root=phases_root,
            source=source,
        )

    def features_step(clips: Sequence[str]) -> Mapping[str, str]:
        """Measure the clips, then update ``hold_summary.csv`` for all of them.

        The summary write is part of the step: a clip whose features exist but
        whose hold rows never reached the summary would be invisible to the
        pre-label, so if the write fails every measured clip fails with it.
        """
        failed: dict[str, str] = {}
        for clip_id in clips:
            try:
                steps.feature_reports.append(
                    features.run_clip(
                        clip_id,
                        in_root=processed_root,
                        labels_root=phases_root,
                        out_root=features_root,
                        source=source,
                    )
                )
            except Exception as error:  # one clip must not end the batch
                failed[clip_id] = f"{type(error).__name__}: {error}"
        if steps.feature_reports:
            try:
                update_hold_summary(
                    features_root / features.HOLD_SUMMARY_NAME,
                    features.hold_summary_table(
                        [report.features for report in steps.feature_reports]
                    ),
                )
            except Exception as error:  # reported, not raised
                message = f"{type(error).__name__}: {error}"
                for clip_id in clips:
                    failed.setdefault(clip_id, f"hold summary: {message}")
        return failed

    def prelabel_step(clips: Sequence[str]) -> Mapping[str, str]:
        if not clips:
            return {}
        try:
            steps.prelabel_report = hold_shapes.prelabel(
                data=root,
                videos=videos,
                clips=list(clips),
                skip_existing=True,
            )
        except Exception as error:  # one broken run is a report, not a crash
            message = f"{type(error).__name__}: {error}"
            return {clip_id: message for clip_id in clips}
        return {}

    steps.runners = {
        "pose_mediapipe": _per_clip(pose),
        "athlete": _per_clip(athlete_step),
        "postprocess": _per_clip(postprocess_step),
        "phases": _per_clip(phases_step),
        "features": features_step,
        "prelabel": prelabel_step,
    }
    return steps


def run_pipeline(
    clips: Sequence[str],
    runners: Mapping[str, StepRunner],
    *,
    order: Sequence[str] = STEPS,
    progress: Callable[[str], None] | None = None,
) -> PipelineReport:
    """Run every step over the clips, in order, isolating per-clip failures.

    Step-major: all clips through ``pose_mediapipe``, then all through
    ``athlete``, and so on. A clip that fails a step is recorded under that
    step and drops out of the later ones; the other clips keep going. Steps are
    idempotent per clip (each module skips outputs that already exist), so the
    same command run again continues where this one stopped.
    """
    say = progress or (lambda message: None)
    alive = list(clips)
    failures: dict[str, dict[str, str]] = {}
    for step in order:
        runner = runners.get(step)
        if runner is None:
            continue
        say(f"{step}: {len(alive)} clip(s)")
        failed = dict(runner(alive))
        if failed:
            failures[step] = failed
            alive = [clip_id for clip_id in alive if clip_id not in failed]
    return PipelineReport(
        order=tuple(order),
        failures=failures,
        done=alive,
        attempted=list(clips),
    )


# --------------------------------------------------------------------------- #
# Per-clip facts the summary reports
# --------------------------------------------------------------------------- #


def holds_per_clip(root: pathlib.Path, source: str, clips: Iterable[str]) -> dict[str, int]:
    """``clip_id ->`` how many holds ``hold_summary.csv`` has for these clips.

    Empty when the summary does not exist yet (the features step never
    finished): the caller reports that rather than inventing zeros.
    """
    wanted = {str(clip_id) for clip_id in clips}
    path = features.output_dir(root, source) / features.HOLD_SUMMARY_NAME
    try:
        rows = hold_shapes.read_hold_summary(path)
    except FileNotFoundError:
        return {}
    counts = Counter(str(row["clip_id"]) for row in rows)
    return {clip_id: counts[clip_id] for clip_id in sorted(wanted) if counts.get(clip_id)}


def trainer_present_clips(
    root: pathlib.Path, source: str, rotate: str, clips: Iterable[str]
) -> list[str]:
    """The clips whose athlete sidecar says a second person was in the shot.

    "Trainer present" is a frame with two or more people, or any frame flagged
    as trainer contact — read from ``keypoints/<athlete dir>/<rotate>/*.json``,
    which the athlete step writes for every clip (and which a re-run keeps).
    """
    base = root / "keypoints" / athlete.output_dirname(source) / rotate
    found: list[str] = []
    for clip_id in sorted(clips):
        sidecar = base / f"{clip_id}.json"
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        people = data.get("frames_by_people") or {}
        reported = [int(key) for key in people if str(key).lstrip("-").isdigit()]
        contact = int(data.get("contact_frames") or 0) > 0
        if any(count >= 2 for count in reported) or contact:
            found.append(clip_id)
    return found


def pick_evenly(
    candidates: Sequence[frame_sampler.FrameCandidate], n: int
) -> list[frame_sampler.FrameCandidate]:
    """At most ``n`` of a clip's frame candidates, evenly spaced and stable.

    First and last are always in when there is more than one pick, and the
    answer depends only on the sequence's order — the same input always yields
    the same frames, which is what makes ``--sample-frames`` repeatable.
    """
    if n <= 0 or not candidates:
        return []
    ordered = sorted(candidates, key=lambda frame: frame.frame_idx)
    if len(ordered) <= n:
        return list(ordered)
    if n == 1:
        return [ordered[len(ordered) // 2]]
    step = (len(ordered) - 1) / (n - 1)
    indices = sorted({round(index * step) for index in range(n)})
    return [ordered[index] for index in indices]


def sample_clip_frames(
    root: pathlib.Path,
    videos: pathlib.Path,
    clips: Sequence[str],
    n: int,
    *,
    rotate: str = ROTATE,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """``n`` keypoint-label frames per clip, written for ``handstand.prelabel``.

    Composes the sampler's own pieces — ``read_catalogue``,
    ``load_clip_frames``, ``write_frames``, ``_manifest_row``,
    ``write_manifest`` — because the pool builder samples the whole dataset by
    skill and these clips are the ones whose skill column is still empty. The
    rows are merged into ``data/label_frames/manifest.csv`` (frames this run
    wrote replace their old rows, every other row survives), and returns the
    new rows plus per-clip failures.
    """
    info = {clip.clip_id: clip for clip in frame_sampler.read_catalogue(root / CATALOGUE_NAME)}
    trainer = frame_sampler.read_trainer_present(
        root / REPORTS_DIRNAME / frame_sampler.TRAINER_REPORT_NAME
    )
    destination = frame_sampler.label_frames_dir(root)
    rows: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    for clip_id in clips:
        clip = info.get(clip_id)
        if clip is None:
            failures[clip_id] = "not in the catalogue"
            continue
        parquet = (
            overlay.keypoints_root(root, frame_sampler.DEFAULT_SOURCE)
            / rotate
            / f"{clip_id}.parquet"
        )
        if not parquet.is_file():
            failures[clip_id] = f"no keypoints at {parquet}"
            continue
        candidates, _has_contact = frame_sampler.load_clip_frames(
            parquet, clip, trainer.get(clip_id, False)
        )
        picked = pick_evenly(candidates, n)
        if not picked:
            failures[clip_id] = "no candidate frames (nobody was detected?)"
            continue
        try:
            video_path = overlay.video_for_clip(clip_id, videos, root)
            destination.mkdir(parents=True, exist_ok=True)
            sizes = frame_sampler.write_frames(
                video_path, {frame.frame_idx: frame for frame in picked}, destination
            )
        except (OSError, RuntimeError, ValueError) as error:
            failures[clip_id] = f"{type(error).__name__}: {error}"
            continue
        for frame in picked:
            # _manifest_row is this package's own one-row schema for the sampler
            # manifest; building it here keeps the ingest's rows byte-identical
            # to what `handstand.frame_sampler` would have written.
            rows.append(
                frame_sampler._manifest_row(
                    frame,
                    sizes[frame.frame_idx],
                    skill=frame.skill or frame_sampler.UNLABELLED_SKILL,
                )
            )
    if rows:
        manifest = frame_sampler.manifest_path(root)
        fresh = {str(row["image"]) for row in rows}
        existing = [
            row for row in load_manifest(manifest) if str(row.get("image", "")) not in fresh
        ]
        frame_sampler.write_manifest([*existing, *rows], manifest)
    return rows, failures


# --------------------------------------------------------------------------- #
# Label Studio over its API
# --------------------------------------------------------------------------- #


class LabelStudioError(RuntimeError):
    """A Label Studio API failure. The message never carries credentials."""


def read_ls_credentials(path: str | pathlib.Path | None = None) -> tuple[str, str]:
    """``(user, password)`` from ``~/.config/handstand/label-studio.env``.

    The file holds ``LS_USER=…`` and ``LS_PASSWORD=…`` lines (``#`` comments
    allowed). Only the names of the variables ever appear in an error — the
    values are returned and never printed, logged or committed.
    """
    env_path = pathlib.Path(path) if path is not None else LS_ENV_PATH
    if not env_path.is_file():
        raise LabelStudioError(
            f"no Label Studio credentials at {env_path}; create it with the lines\n"
            "  LS_USER=<login>\n"
            "  LS_PASSWORD=<password>\n"
            "and chmod 600 (it is read only for --label-studio and never committed)"
        )
    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip().strip("\"'")
    if not values.get("LS_USER") or not values.get("LS_PASSWORD"):
        raise LabelStudioError(f"{env_path} must define both LS_USER and LS_PASSWORD")
    return values["LS_USER"], values["LS_PASSWORD"]


def urllib_http(
    method: str,
    url: str,
    body: bytes | None,
    headers: Mapping[str, str],
) -> tuple[int, Sequence[tuple[str, str]], bytes]:
    """The default :data:`HttpCall`: one stdlib request, HTTP errors included."""
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return int(response.status), list(response.headers.items()), response.read()
    except urllib.error.HTTPError as error:
        return int(error.code), list(error.headers.items()), error.read()


class LabelStudioClient:
    """A logged-in Label Studio session: session cookie + CSRF, the usual way.

    1. ``GET /user/login/`` picks up the ``csrftoken`` cookie;
    2. ``POST /user-login/`` exchanges the credentials for a ``sessionid``,
       echoing the CSRF token in the ``X-CSRFToken`` header and a ``Referer``;
    3. the project is looked up **by title** (``handstand-hold-shapes``), so no
       hard-coded project id can import 200 tasks into the wrong project.

    ``http`` is injectable, so tests mock the whole HTTP conversation.
    """

    def __init__(self, base_url: str = DEFAULT_LS_URL, *, http: HttpCall | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._http: HttpCall = http or urllib_http
        self.cookies: dict[str, str] = {}

    def _call(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> bytes:
        cookie_header = "; ".join(f"{key}={value}" for key, value in self.cookies.items())
        merged = {"Cookie": cookie_header, **(headers or {})}
        status, response_headers, payload = self._http(method, url, body, merged)
        for name, value in response_headers:
            if name.lower() == "set-cookie":
                cookie = value.split(";", 1)[0]
                key, _, cookie_value = cookie.partition("=")
                if key.strip():
                    self.cookies[key.strip()] = cookie_value.strip()
        if status >= 400:
            snippet = payload[:200].decode("utf-8", "replace").replace("\n", " ")
            raise LabelStudioError(f"{method} {url} -> HTTP {status}: {snippet}")
        return payload

    def login(self, user: str, password: str) -> None:
        """Open a session; raises when the cookies say the login did not take."""
        self._call("GET", f"{self.base_url}/user/login/")
        csrf = self.cookies.get("csrftoken", "")
        form = urllib.parse.urlencode({"username": user, "password": password}).encode()
        self._call(
            "POST",
            f"{self.base_url}/user-login/",
            form,
            {
                "Content-Type": "application/x-www-form-urlencoded",
                "X-CSRFToken": csrf,
                "Referer": f"{self.base_url}/user/login/",
            },
        )
        if "sessionid" not in self.cookies:
            raise LabelStudioError(
                "Label Studio login failed: no session cookie came back (check LS_USER/LS_PASSWORD)"
            )

    def find_project_id(self, title: str) -> int:
        """The id of the one project whose title is exactly ``title``."""
        query = urllib.parse.urlencode({"title": title})
        payload = self._call("GET", f"{self.base_url}/api/projects?{query}")
        data = json.loads(payload)
        projects = data.get("projects") if isinstance(data, dict) else data
        for project in projects if isinstance(projects, list) else []:
            if isinstance(project, dict) and project.get("title") == title:
                return int(project["id"])
        raise LabelStudioError(f"no Label Studio project titled {title!r}")

    def import_tasks(self, project_id: int, tasks: Sequence[Mapping[str, Any]]) -> int:
        """POST the tasks into the project; returns how many were sent."""
        body = json.dumps(list(tasks)).encode()
        self._call(
            "POST",
            f"{self.base_url}/api/projects/{project_id}/import",
            body,
            {
                "Content-Type": "application/json",
                "X-CSRFToken": self.cookies.get("csrftoken", ""),
                "Referer": self.base_url,
            },
        )
        return len(tasks)


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class IngestReport:
    """What one ingest run produced — the structured half of the summary."""

    today: str
    dry_run: bool
    keep: bool
    inbox: pathlib.Path
    videos: pathlib.Path
    catalogue_path: pathlib.Path
    new: list[NewClip]
    #: ``clip_id ->`` where the file actually landed (empty in a dry run).
    placed: dict[str, pathlib.Path]
    duplicates: list[tuple[pathlib.Path, str, str]]
    others: list[pathlib.Path]
    steps: tuple[str, ...]
    #: ``step -> {clip_id: error}``, pipeline and beyond (place, catalogue,
    #: sample_frames, label_studio all use the same shape).
    failures: dict[str, dict[str, str]]
    holds_per_clip: dict[str, int]
    #: Holds queued for review by THIS run's pre-label.
    queued: int
    queued_per_shape: dict[str, int]
    #: New-clip holds nobody has reviewed yet — the to-do.
    todo: int
    import_file: pathlib.Path | None
    sample_file: pathlib.Path | None
    sampled: int
    trainer_present: list[str]
    no_hold: list[str]
    #: What ``--label-studio`` did (``""`` when it was not asked for).
    label_studio: str
    catalogue_added: int
    report_path: pathlib.Path
    #: The whole printed summary; also written to :attr:`report_path`.
    text: str

    @property
    def ok(self) -> bool:
        """Did everything that was asked for succeed?"""
        return not self.failures


def _fail(failures: dict[str, dict[str, str]], step: str, key: str, message: str) -> None:
    """Record one failure under its step (``key`` is a clip id or ``"-"``)."""
    failures.setdefault(step, {})[key] = message


def run_ingest(
    *,
    data: str | pathlib.Path | None = None,
    videos: str | pathlib.Path | None = None,
    dry_run: bool = False,
    keep: bool = False,
    label_studio: bool = False,
    sample_frames: int = 0,
    ls_url: str = DEFAULT_LS_URL,
    probe: catalogue_module.Probe | None = None,
    http: HttpCall | None = None,
    env_path: str | pathlib.Path | None = None,
    step_overrides: Mapping[str, StepRunner] | None = None,
    today: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> IngestReport:
    """Do the whole weekly ingest (or, with ``dry_run``, only describe it).

    Order: scan the inbox, decide duplicates by content, plan the renames, and
    — unless this is a dry run — move the files, append the catalogue rows,
    run the six pipeline steps over exactly those clips, queue their holds for
    the shape review, optionally sample keypoint frames and optionally import
    the new tasks into Label Studio. Every failure is recorded per step and
    never stops the rest of the run.

    ``step_overrides`` replaces step runners (tests stub the model steps);
    ``probe``/``http``/``env_path``/``today`` are the injectable seams for
    ffprobe, the Label Studio HTTP, the credentials file and the report name.
    """
    say = progress if progress is not None else (lambda message: print(message, flush=True))
    root = pathlib.Path(data) if data is not None else data_dir()
    videos_root = pathlib.Path(videos) if videos is not None else videos_dir()
    inbox = videos_root / INBOX_DIRNAME
    catalogue_path = root / CATALOGUE_NAME
    day = today or dt.date.today().isoformat()
    reports_dir = root / REPORTS_DIRNAME / INGEST_DIRNAME
    report_path = reports_dir / f"{day}.txt"
    import_path = reports_dir / f"{day}_hold_shapes_tasks.json"
    sample_path = reports_dir / f"{day}_keypoint_tasks.json"
    failures: dict[str, dict[str, str]] = {}

    if not dry_run:
        inbox.mkdir(parents=True, exist_ok=True)
    plan = plan_inbox(inbox, videos_root, catalogue_path)

    placed: dict[str, pathlib.Path] = {}
    catalogue_added = 0
    pipeline: PipelineReport | None = None
    steps = StepSet(runners={})
    holds: dict[str, int] = {}
    queued = 0
    queued_per_shape: dict[str, int] = {}
    todo = 0
    import_file: pathlib.Path | None = None
    sample_file: pathlib.Path | None = None
    sampled = 0
    trainer_present: list[str] = []
    no_hold: list[str] = []
    label_studio_message = ""

    if not dry_run and plan.new:
        # 1. Move (or copy) the files in; a failed one is recorded and dropped.
        for clip in plan.new:
            try:
                placed[clip.clip_id] = place_clip(clip, keep=keep)
                say(f"placed {clip.source.name} -> {placed[clip.clip_id].name}")
            except Exception as error:  # one file must not end the run
                _fail(failures, "place", clip.clip_id, f"{type(error).__name__}: {error}")

        # 2. Append the catalogue rows (the real probe; identity row on failure).
        try:
            catalogue_added, catalogue_failures = catalogue_new_clips(placed, catalogue_path, probe)
            for clip_id, message in catalogue_failures.items():
                _fail(failures, "catalogue", clip_id, message)
        except OSError as error:
            _fail(failures, "catalogue", "-", f"{type(error).__name__}: {error}")

        # 3. The pipeline, new clips only, in order.
        steps = build_runners(root, videos_root)
        runners = {**steps.runners, **(dict(step_overrides) if step_overrides else {})}
        pipeline = run_pipeline(list(placed), runners, progress=say)
        for step, per_clip in pipeline.failures.items():
            for clip_id, message in per_clip.items():
                _fail(failures, step, clip_id, message)

        # 4. What the run produced for the summary.
        holds = holds_per_clip(root, SOURCE, placed)
        labels = hold_shapes.load_hold_shapes(root)
        todo = sum(
            1
            for row in _read_new_holds(root, placed)
            if (row["clip_id"], row["hold_id"]) not in labels
        )
        report = steps.prelabel_report
        tasks: list[dict[str, Any]] = report.tasks if report is not None else []
        if report is not None:
            queued = len(tasks)
            queued_per_shape = {shape: count for shape, count in report.counts.items() if count}
        trainer_present = trainer_present_clips(root, SOURCE, ROTATE, pipeline.done)
        no_hold = [clip_id for clip_id in pipeline.done if not holds.get(clip_id)]

        if queued and not label_studio:
            # Without --label-studio: write the import JSON and say where it is.
            try:
                _atomic_write(import_path, json.dumps(tasks, indent=2) + "\n")
                import_file = import_path
            except OSError as error:
                _fail(failures, "review", "-", f"could not write {import_path}: {error}")

        # 5. Optional: keypoint-labelling frames for the new clips.
        if sample_frames > 0 and pipeline.done:
            rows, sample_failures = sample_clip_frames(
                root, videos_root, pipeline.done, sample_frames
            )
            for clip_id, message in sample_failures.items():
                _fail(failures, "sample_frames", clip_id, message)
            sampled = len(rows)
            if rows:
                try:
                    model = make_pose_model()
                    keypoint_report = prelabel_keypoints(
                        model=model,
                        data=root,
                        only=[str(row["image"]) for row in rows],
                        out_json=sample_path,
                        merge_review=True,
                    )
                    if keypoint_report.tasks:
                        sample_file = sample_path
                except Exception as error:  # one broken model is a report, not a crash
                    _fail(failures, "sample_frames", "-", f"{type(error).__name__}: {error}")

        # 6. Optional: import the NEW hold tasks into Label Studio.
        if label_studio:
            if queued:
                try:
                    user, password = read_ls_credentials(env_path)
                    client = LabelStudioClient(ls_url, http=http)
                    client.login(user, password)
                    project_id = client.find_project_id(PROJECT_TITLE)
                    client.import_tasks(project_id, tasks)
                    link = f"{client.base_url}/projects/{project_id}"
                    label_studio_message = (
                        f"imported {queued} new task(s) into {PROJECT_TITLE!r} "
                        f"(project {project_id}) — {link}"
                    )
                except Exception as error:  # a failed import must not lose the tasks
                    message = f"{type(error).__name__}: {error}"
                    _fail(failures, "label_studio", "-", message)
                    label_studio_message = f"import failed: {message}"
                    try:
                        # Fall back to the import file so the tasks are not lost.
                        _atomic_write(import_path, json.dumps(tasks, indent=2) + "\n")
                        import_file = import_path
                    except OSError as write_error:
                        _fail(
                            failures,
                            "review",
                            "-",
                            f"could not write {import_path}: {write_error}",
                        )
            else:
                label_studio_message = "nothing to import: no new hold was queued"
    elif not dry_run:
        say("inbox has no new clips")

    text = _render_summary(
        today=day,
        dry_run=dry_run,
        keep=keep,
        plan=plan,
        placed=placed,
        catalogue_path=catalogue_path,
        catalogue_added=catalogue_added,
        pipeline=pipeline,
        failures=failures,
        holds=holds,
        queued=queued,
        queued_per_shape=queued_per_shape,
        todo=todo,
        import_file=import_file,
        sample_file=sample_file,
        sampled=sampled,
        sample_frames=sample_frames,
        trainer_present=trainer_present,
        no_hold=no_hold,
        label_studio=label_studio_message,
        report_path=report_path,
        label_studio_wanted=label_studio,
    )
    return IngestReport(
        today=day,
        dry_run=dry_run,
        keep=keep,
        inbox=plan.inbox,
        videos=plan.videos,
        catalogue_path=catalogue_path,
        new=plan.new,
        placed=placed,
        duplicates=plan.duplicates,
        others=plan.others,
        steps=STEPS,
        failures=failures,
        holds_per_clip=holds,
        queued=queued,
        queued_per_shape=queued_per_shape,
        todo=todo,
        import_file=import_file,
        sample_file=sample_file,
        sampled=sampled,
        trainer_present=trainer_present,
        no_hold=no_hold,
        label_studio=label_studio_message,
        catalogue_added=catalogue_added,
        report_path=report_path,
        text=text,
    )


def _read_new_holds(root: pathlib.Path, placed: Mapping[str, pathlib.Path]) -> list[dict[str, Any]]:
    """The hold-summary rows of the clips this run placed (``[]`` if unreadable)."""
    path = features.output_dir(root, SOURCE) / features.HOLD_SUMMARY_NAME
    try:
        rows = hold_shapes.read_hold_summary(path)
    except FileNotFoundError:
        return []
    wanted = set(placed)
    return [row for row in rows if str(row["clip_id"]) in wanted]


def _render_summary(
    *,
    today: str,
    dry_run: bool,
    keep: bool,
    plan: Plan,
    placed: Mapping[str, pathlib.Path],
    catalogue_path: pathlib.Path,
    catalogue_added: int,
    pipeline: PipelineReport | None,
    failures: Mapping[str, Mapping[str, str]],
    holds: Mapping[str, int],
    queued: int,
    queued_per_shape: Mapping[str, int],
    todo: int,
    import_file: pathlib.Path | None,
    sample_file: pathlib.Path | None,
    sampled: int,
    sample_frames: int,
    trainer_present: Sequence[str],
    no_hold: Sequence[str],
    label_studio: str,
    report_path: pathlib.Path,
    label_studio_wanted: bool,
) -> str:
    """The summary text: everything the spec asks for, printed and written."""
    flag = " (DRY RUN: nothing was written)" if dry_run else ""
    lines = [f"weekly ingest {today}{flag}"]
    lines.append(f"  inbox: {plan.inbox}")
    lines.append(f"  videos: {plan.videos}")
    lines.append(f"  mode: {'copy (--keep)' if keep else 'move'} into the videos directory")
    lines.append(
        f"  scan: {len(plan.new) + len(plan.duplicates)} mp4 clip(s), "
        f"{len(plan.others)} other entries left alone"
    )
    for path in plan.others:
        lines.append(f"    left alone: {path.name}")

    if plan.duplicates:
        lines.append(f"  duplicate(s): {len(plan.duplicates)} (by content, kept in the inbox)")
        for path, _clip_id, reason in plan.duplicates:
            lines.append(f"    {path.name} — {reason}")
    else:
        lines.append("  duplicate(s): 0")

    renames = [clip for clip in plan.new if clip.renamed]
    placed_ok = set(placed)
    lines.append(f"  new clip(s): {len(plan.new)}")
    for clip in plan.new:
        holds_note = ""
        if not dry_run and clip.clip_id in placed_ok:
            holds_note = f"  holds={holds.get(clip.clip_id, 0)}"
        lines.append(f"    {clip.clip_id}  {clip.name}{holds_note}")
    if renames:
        lines.append(f"  rename(s): {len(renames)} (old -> new)")
        for clip in renames:
            target = placed.get(clip.clip_id, clip.path)
            lines.append(f"    {clip.source.name} -> {target.name}")
    else:
        lines.append("  rename(s): 0")

    if dry_run:
        lines.append(f"  catalogue: would append {len(plan.new)} row(s) to {catalogue_path}")
        lines.append(f"  pipeline: would run {', '.join(STEPS)} for {len(plan.new)} clip(s)")
        if sample_frames:
            lines.append(
                f"  sample-frames: would sample {sample_frames} keypoint frame(s) per new clip"
            )
        destination = "via the Label Studio API" if label_studio_wanted else "as an import file"
        lines.append(f"  review: would queue the new holds for shape review {destination}")
        lines.append("  report: not written (dry run)")
        return "\n".join(lines)

    lines.append(f"  catalogue: appended {catalogue_added} row(s) to {catalogue_path}")

    if pipeline is not None:
        lines.append(
            f"  pipeline: {len(pipeline.done)} of {len(pipeline.attempted)} clip(s) "
            f"completed {', '.join(pipeline.order)}"
        )

    if holds:
        lines.append("  holds per clip: " + " ".join(f"{c}={n}" for c, n in holds.items()))
    if queued_per_shape:
        shape_counts = " ".join(
            f"{shape}={queued_per_shape[shape]}"
            for shape in hold_shapes.SHAPES
            if queued_per_shape.get(shape)
        )
        lines.append(f"  queued this run: {queued} hold(s) per predicted shape: {shape_counts}")
    elif pipeline is not None:
        lines.append("  queued this run: 0 hold(s)")

    lines.append(
        "  trainer present: " + (", ".join(trainer_present) if trainer_present else "none")
    )
    lines.append("  no hold: " + (", ".join(no_hold) if no_hold else "none"))

    if failures:
        lines.append(f"  failures: {sum(len(v) for v in failures.values())} (per step)")
        for step in (*STEPS, "place", "catalogue", "sample_frames", "label_studio"):
            for clip_id, message in failures.get(step, {}).items():
                lines.append(f"    {step}: {clip_id}: {message}")
        for step, per_clip in failures.items():
            if step in STEPS or step in {
                "place",
                "catalogue",
                "sample_frames",
                "label_studio",
            }:
                continue
            for clip_id, message in per_clip.items():
                lines.append(f"    {step}: {clip_id}: {message}")
    else:
        lines.append("  failures: none")

    if label_studio_wanted:
        detail = label_studio or "nothing to import: no new hold was queued"
        lines.append(f"  review: {todo} new hold(s) to label — {detail}")
        if import_file is not None:
            lines.append(f"    import file also written: {import_file}")
    elif import_file is not None:
        lines.append(f"  review: {todo} new hold(s) to label — import file: {import_file}")
    elif todo:
        lines.append(f"  review: {todo} new hold(s) to label (queued in an earlier run)")
    else:
        lines.append("  review: no new hold to label")

    if sample_frames:
        lines.append(
            f"  sampled frames: {sampled} of {sample_frames}/clip requested"
            + (f" — tasks: {sample_file}" if sample_file else "")
        )
    lines.append(f"  report: {report_path}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    """The ``python -m handstand.ingest`` command line."""
    parser = argparse.ArgumentParser(
        prog="python -m handstand.ingest",
        description=(
            "Weekly ingest: catalogue every new clip in videos/inbox/, run the "
            "whole pipeline over them and queue their holds for the shape review."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print renames, duplicates and what would run; write NOTHING",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="copy the inbox files into videos/ instead of moving them",
    )
    parser.add_argument(
        "--label-studio",
        action="store_true",
        help=(
            "import the NEW hold-shape tasks into the handstand-hold-shapes "
            "project through the Label Studio API (credentials in "
            f"{LS_ENV_PATH}); without this flag the import JSON is written instead"
        ),
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=0,
        metavar="N",
        help=(
            "also sample N keypoint-labelling frames per new clip into a "
            "separate new-task import file (default: 0 = off)"
        ),
    )
    parser.add_argument(
        "--ls-url",
        default=DEFAULT_LS_URL,
        metavar="URL",
        help=f"Label Studio base URL (default: {DEFAULT_LS_URL})",
    )
    parser.add_argument(
        "--data",
        type=pathlib.Path,
        default=None,
        help="data directory (default: $HANDSTAND_DATA)",
    )
    parser.add_argument(
        "--videos",
        type=pathlib.Path,
        default=None,
        help="videos directory holding inbox/ (default: $HANDSTAND_VIDEOS)",
    )
    return parser


def main(argv: Sequence[str] | None = None, **seams: Any) -> int:
    """Command line entry point: ``python -m handstand.ingest``."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.sample_frames < 0:
        parser.error("--sample-frames must be >= 0")

    videos_root = (args.videos or videos_dir()).expanduser()
    if not videos_root.is_dir():
        print(f"ingest: not a videos directory: {videos_root}", file=sys.stderr)
        return 2

    report = run_ingest(
        data=(args.data or data_dir()).expanduser(),
        videos=videos_root,
        dry_run=args.dry_run,
        keep=args.keep,
        label_studio=args.label_studio,
        sample_frames=args.sample_frames,
        ls_url=args.ls_url,
        **seams,
    )
    print(report.text)
    if not report.dry_run:
        try:
            _atomic_write(report.report_path, report.text + "\n")
        except OSError as error:
            print(f"ingest: could not write the report: {error}", file=sys.stderr)
            return 1
        print(f"  written: {report.report_path}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
