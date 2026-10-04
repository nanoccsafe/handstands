# Weekly ingest

One command turns a folder of new WhatsApp exports into catalogued clips,
finished pipeline output and a shape-review queue waiting in Label Studio.

```
handstand.ingest  ->  videos/inbox/*.mp4 moved into videos/
                      catalogue.csv appended (new rows only)
                      pose_mediapipe -> athlete -> postprocess -> phases
                        -> features -> hold_shapes prelabel   (new clips only)
                      data/reports/ingest/<date>.txt + the hold-shape import
```

## Where to drop the clips

Into `videos/inbox/` (`$HANDSTAND_VIDEOS/inbox/`), created on the first run.
Only `*.mp4` files are clips; anything else — a WhatsApp `.jpeg` still, a
sub-folder — is reported in the summary and left exactly where it is.

## The command

Always look first: `--dry-run` prints every rename, every duplicate and which
steps would run, and writes **nothing** (no move, no catalogue row, no report).

```bash
cd pipeline
uv run python -m handstand.ingest --dry-run
uv run python -m handstand.ingest
```

The real run, in order:

1. **Inbox** — every `*.mp4`, decided by *content*: `clip_id` is the SHA-1 of
   the bytes (`catalogue.file_clip_id`), never the file name. A file whose id
   is already in `catalogue.csv` is a duplicate: reported, left in the inbox.
2. **Rename and move** — the WhatsApp `(N)` suffix becomes ` take{N+1}` (see
   below); the file moves into `videos/` (with `--keep` it is copied instead).
   Nothing is ever overwritten, deleted or skipped because of a name: a target
   that exists with *different* content gets the smallest free ` (ingest N)`
   suffix.
3. **Catalogue** — the new rows are appended through `catalogue.build_row`
   (the real ffprobe), hand-annotated columns empty, written atomically. The
   existing bytes of `catalogue.csv` are a byte-for-byte prefix of the new
   file, so an interrupted or buggy run cannot damage what you typed.
4. **Pipeline, new clips only, in order** — `pose_mediapipe` (rotate `best`,
   3 poses, IMAGE mode with 0.2 detection/presence: the settings every
   existing athlete input was made with), `athlete`, `postprocess`, `phases`,
   `features` (which updates `hold_summary.csv` for these clips only), then
   `hold_shapes.prelabel(clips=…, skip_existing=True)`. One clip failing a
   step is recorded and the other clips continue.
5. **Review queue** — every *new* hold gets a Label Studio task. A hold that
   already has an image or a reviewed row in `hold_shapes.csv` is never
   touched again, so a re-run only ever adds NEW holds. Without
   `--label-studio` the new tasks are written to
   `data/reports/ingest/<date>_hold_shapes_tasks.json` and the summary says
   where it is.
6. **Optional** — `--label-studio` imports those new tasks straight into the
   `handstand-hold-shapes` project (looked up **by title**, so no hard-coded
   project id) over the Label Studio API: session login + CSRF, credentials
   from `~/.config/handstand/label-studio.env` (`LS_USER=…`, `LS_PASSWORD=…`,
   `chmod 600`, never printed or committed), URL defaulting to
   `http://localhost:8080` (`--ls-url` to change it). Before the POST the
   project's existing tasks are read page by page and every
   `(clip_id, hold_id)` already there is skipped, so a re-run — or a file that
   was imported by hand — never queues a hold twice (#89).
7. **`--sample-frames N`** (default 0) additionally samples N keypoint-labelling
   frames per new clip and pre-labels them into a *separate* import file
   (`data/reports/ingest/<date>_keypoint_tasks.json`).

The summary is printed and written to `data/reports/ingest/<date>.txt`: the
new clips with their holds per clip and per predicted shape, duplicates,
renames (old → new), failures per step, clips with a trainer present, clips
with no hold, and the review to-do — N new holds to label, with the Label
Studio link or the import file. Exit code 0 when everything succeeded, 1 when
anything failed (the failures are listed).

## The rename rule: `(N)` is not a copy

WhatsApp appends `(1)` (or `(2)` …) to a filename when several videos are
exported quickly. It says **nothing** about the content: in the 2026-10-02
batch, `…12.47.45 PM.mp4` and `…12.47.45 PM(1).mp4` are two *different*
clips. Duplicates are decided only by the content hash, never by the name.

So the file is renamed as it moves into `videos/`, and nobody mistakes it for
a copy again:

| in the inbox | in `videos/` |
| --- | --- |
| `WhatsApp Video 2026-10-02 at 12.47.45 PM.mp4` | same name (take 1, implicitly) |
| `WhatsApp Video 2026-10-02 at 12.47.45 PM(1).mp4` | `… 12.47.45 PM take2.mp4` |
| `… 12.47.45 PM (1).mp4` (space before `(`) | `… 12.47.45 PM take2.mp4` |

If the target name already exists with different content, the file gets the
smallest free ` (ingest N)` suffix instead — never overwritten, never skipped.
`catalogue._FILENAME_RE` reads ` takeN` names like any other, so
`recorded_at` stays right; `--dry-run` shows every rename before it happens.

## What to review afterwards

1. Read `data/reports/ingest/<date>.txt` — failures first, then the new clips
   and the review to-do line.
2. Label the new holds: with `--label-studio` they are already in the
   `handstand-hold-shapes` project (project 3); otherwise import
   `data/reports/ingest/<date>_hold_shapes_tasks.json` into it — through the
   API with `--import-only` (see below) or by hand (Import → Import
   predictions). Hotkeys 1–9
   are `line`, `straddle`, `split_stag`, `tuck`, `pike`, `other`,
   `not_a_hold`, `walk`, `mexican` — see `docs/labeling.md`.
3. After the review, refresh the catalogue's skill column:

   ```bash
   uv run python -m handstand.hold_shapes summary --write-catalogue
   ```

4. Optionally regenerate the trainer report (the new clips enter it):
   `uv run python -m handstand.trainer_report --generate`.

## Recovering from a failed step

The summary lists every failure as `step: clip: error`, and the run's exit
code is 1 when there is any. Every step is **idempotent per clip** — a step
whose output already exists skips instead of redoing it — so recovery is
always "fix the cause, then re-run what failed":

| failed step | re-run |
| --- | --- |
| `place` / `catalogue` | the file is still in the inbox (nothing was lost): fix the cause and run the ingest again |
| `pose_mediapipe` | `uv run python -m handstand.pose_mediapipe --rotate best --num-poses 3 --running-mode image --min-detection 0.2 --min-presence 0.2` (add `--clips <video path>` for one file; missing model → `uv run python scripts/download_models.py`) |
| `athlete` | `uv run python -m handstand.athlete --rotate best --clips <clip_id>` |
| `postprocess` | `uv run python -m handstand.postprocess --clips <clip_id>` |
| `phases` | `uv run python -m handstand.phases --clips <clip_id>` |
| `features` | `uv run python -m handstand.features --clips <clip_id> --overwrite` (`--overwrite` is what rewrites `hold_summary.csv` for that clip while keeping every other clip's rows) |
| `prelabel` | `uv run python -m handstand.hold_shapes prelabel --clips <clip_id> --skip-existing` (only adds NEW holds and merges the manifest/tasks) |
| `label_studio` | the tasks were written to a fallback import file (the path is in the summary): re-import it with `--import-only` once Label Studio and the credentials are reachable again — the holds already in the project are skipped (see below) |
| `sample_frames` | re-run the ingest with `--sample-frames N` once the cause is fixed: sampling only ever touches the clips it is given |

Once a clip has been moved into `videos/` and catalogued, a fresh ingest run
will not pick it up again (its content is in the catalogue now) — that is what
the per-step commands above are for. If a run died *before* the move, just fix
it and run the ingest again; the file is still in the inbox.

`uv run python -m handstand.ingest --dry-run` afterwards shows the current
state of the inbox without changing anything.

## Label Studio import failed

`--label-studio` is the only part of the run that talks to another program,
so it is the only part that can fail while everything else worked — the
2026-10-04 run died at the login with `POST http://localhost:8080/user-login/
-> HTTP 404`, because that URL does not exist in Label Studio 1.23: its login
form lives at `/user/login/`. A run can also die later with
`GET /api/projects?page_size=100 -> HTTP 401 Authentication credentials were
not provided`, which means the session never became a logged-in one. The
summary records the failure as `label_studio: -: …` (exit code 1) and writes
the tasks to a fallback import file, so nothing is lost.

`--import-only` is the recovery: it reads ONE already-written tasks file,
logs in, finds `handstand-hold-shapes` by title and imports that file's
tasks, skipping every `(clip_id, hold_id)` the project already holds. It
touches nothing else — no inbox scan, no catalogue, no pipeline, no report:

```bash
cd pipeline
uv run python -m handstand.ingest \
  --import-only data/reports/ingest/2026-10-04_hold_shapes_tasks.json --label-studio
```

It prints what it did:

```
import-only data/reports/ingest/2026-10-04_hold_shapes_tasks.json
  project: handstand-hold-shapes (project 3) — http://localhost:8080/projects/3
  tasks: 32 in the file, 32 already present, 0 imported
```

so running it twice — or after the tasks were imported by hand — reports
`32 already present, 0 imported` instead of queueing every hold a second
time. Exit code 0, no report file, no other side effect.

How the login works (Label Studio 1.23, and why this command succeeds where
the first run did not):

1. `GET /user/login/` — picks up the `csrftoken` cookie, the page with its
   hidden `csrfmiddlewaretoken` and an *anonymous* `sessionid`.
2. `POST` to that **same** `/user/login/` with the form fields
   `csrfmiddlewaretoken` (from the hidden input, or from the `csrftoken`
   cookie), `email` and `password`, plus `Referer: <base>/user/login/`.
3. Redirects are **never followed**: a successful Django login answers the
   POST with a **302**, and the logged-in `sessionid` is set *on that 302* —
   following it would drop that cookie and pick up a fresh anonymous one
   from the page it points at (which is how a failed login once looked like
   a success and the API answered 401). So the login only counts as a
   success when the POST answers 302/303 with a `sessionid` **different**
   from the anonymous one the GET set. Anything else — a 200 (the login form
   re-rendered) or a redirect that keeps the anonymous cookie — is a failed
   login: the error names `LS_USER`/`LS_PASSWORD` and where the credentials
   file is, and never prints the password.
4. Every API call afterwards (`/api/projects?page_size=100`,
   `/api/tasks?project=…`, the import) sends `X-CSRFToken` (the `csrftoken`
   cookie), the logged-in `sessionid` cookie and `Referer: <base>`.

It needs `~/.config/handstand/label-studio.env` (`LS_USER=…`, `LS_PASSWORD=…`,
`chmod 600`) and `--ls-url` when Label Studio is not on
`http://localhost:8080`.
