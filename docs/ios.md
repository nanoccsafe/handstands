# The iOS app

`ios/` is the iPhone app: a SwiftUI shell around the `HandstandCore` package
(`swift/HandstandCore/`). The `.xcodeproj` is **generated** from
`ios/project.yml` with XcodeGen and is never committed — edit `project.yml`,
not the project. The project's one CocoaPod — Google's MediaPipe Tasks Vision,
the default pose backend (see **Pose backends** below) — hooks into that
generated project, so `pod install` follows every `xcodegen generate` and the
generated `.xcworkspace` is what you open (and, like the project, is never
committed).

## Generate and open (on the Mac)

```bash
export PATH="/opt/homebrew/bin:$PATH"   # XcodeGen and CocoaPods live in homebrew
cd ios
xcodegen generate
pod install                              # after every xcodegen generate
open Handstand.xcworkspace               # the workspace, not the .xcodeproj
```

Re-run `xcodegen generate` whenever `project.yml` changes (including after a
pull) and `pod install` right after it — CocoaPods re-hooks into the freshly
generated project. `Podfile.lock` is committed (it pins the MediaPipe
version); `Pods/` and `Handstand.xcworkspace` are git-ignored and re-created
by `pod install`, which needs no network once `Pods/` is installed.

Also fetch the pose model before the first build (see **Pose backends**):
`tools/ios/fetch_mediapipe_model.sh` puts it into `LocalResources/`, which
`project.yml` bundles when present and skips when absent.

## Build and test from Linux

```bash
tools/mac/ios_build.sh          # fetch the model if missing, rsync ios/ + swift/
                                 # to the Mac, xcodegen + pod install, simulator
                                 # build, HandstandAppTests
tools/mac/ios_build.sh wt/i44   # the same, spelled out
```

It runs `tools/ios/fetch_mediapipe_model.sh` first when the model is missing
(the model is never committed), then picks an available iPhone simulator from
`xcrun simctl list devices available` (nothing hard-coded) and exits
non-zero if generation, the pod install, the build or the tests fail, printing
`** BUILD SUCCEEDED **`, `** TEST SUCCEEDED **` and the `Executed … tests`
lines.

## Pose backends

The app runs one of two pose backends, decided by `PoseBackend.preferred`
(`ios/HandstandApp/Pose/PoseBackend.swift`):

- **MediaPipe — the default** (chainlink #45), whenever
  `pose_landmarker_full.task` is in the app bundle. The bake-off
  (**docs/bakeoff.md**, chainlink #16) picked it over Apple Vision: PCK@0.2
  on clean line holds **0.76 vs 0.43** at the common gate, MediaPipe wins at
  every gate, and at the 0.5 gate Vision leaves **84/179** clips
  unmeasurable against MediaPipe's **3/179**. It runs the *same way the
  pipeline does* — `pose_mediapipe.py --rotate best`: two VIDEO-mode
  landmarkers (MediaPipe's own 0.5 confidences, `numPoses = 1`), one fed
  upright frames and one fed 180°-turned frames, both on every frame, and
  `HandstandCore.OrientationChooser` picking the winner over the whole clip
  — because the thresholds, the reference and the scorer were all built on
  those keypoints. The extraction is clip-level
  (`MediaPipeClipExtractor.extract`), so `MediaPipePoseService.process`
  refuses a single frame honestly (`PoseServiceError.clipLevelOnly`) rather
  than guessing one.
- **Apple Vision — the fallback** (chainlink #82), always: built into iOS,
  needs no download, and what a build without the fetched model runs.

The **model file is not committed** (`*.task` is git-ignored). Fetch it into
`ios/LocalResources/models/pose_landmarker_full.task` with:

```bash
tools/ios/fetch_mediapipe_model.sh
```

It copies `pipeline/models/pose_landmarker_full.task` when the pipeline has
it, otherwise downloads the exact `MODEL_URL` `pose_mediapipe.py` uses.
`ios/LocalResources/` is git-ignored (like the scoring reference, #28) and an
optional resource folder, so a build without it works — the app then analyses
with Vision.

Two per-backend numbers live in `PoseBackend` (one place, so a later
recalibration changes one line): `minVisibility` — the post-process gate,
**0.5 for both** backends for now (MediaPipe's is the bake-off's gate) —
passed into `Analyzer.analyze(config:)`; and `analysisVersion` —
`"mediapipe-1"` for MediaPipe runs, `"vision-1"` for Vision — which is what
makes pose caches (#48) written by the old builds fail their version check
and recompute instead of replaying.

CocoaPods layout: `ios/Podfile` (committed; `MediaPipeTasksVision ~> 0.10.35`,
pinned by the committed `Podfile.lock`), `Pods/` and the generated
`Handstand.xcworkspace` git-ignored, and `pod install` after every
`xcodegen generate` — `tools/mac/ios_build.sh` does both in that order and
stays offline when `Pods/` is already installed.

## Signing with a free Apple ID

No App Store, no paid account: a free personal team is enough.

1. Find your team id: Xcode → Settings → Accounts → add your Apple ID →
   view the team (looks like `AB12CD34EF`). It is also shown in the target's
   **Signing & Capabilities → Team**.
2. Either put it in a file: copy `ios/Config/Local.xcconfig.example` to
   `ios/Config/Local.xcconfig` (same directory) and set
   `DEVELOPMENT_TEAM = <team id>`. `Local.xcconfig` is git-ignored —
   **never commit a team id**, the repo is public.
   …or in Xcode: select the **HandstandApp** target → Signing &
   Capabilities → Team → your personal team.
3. Free teams need a bundle id nobody else uses; `com.nanoccsafe.handstand`
   is fine unless Xcode says it is taken, in which case change it in
   `project.yml` to something like `com.<yourname>.handstand`.

Without a team id the simulator build and tests still run; only installing on
a phone needs one.

## Sideloading onto an iPhone

1. Connect the iPhone (cable, or wireless with the phone on the same network)
   and unlock it; tap **Trust** on the phone if asked.
2. In Xcode pick the iPhone as the run destination (top of the window) and
   press Run (⌘R). Xcode provisions it with your free team.
3. First time only: on the phone, **Settings → General → VPN & Device
   Management →** your developer app → **Trust**. The app opens after that.
4. Free-provisioning installs **expire after 7 days**; then the icon goes
   grey. Just repeat step 2 — reinstall the same way, nothing is lost.

## Hold type

The Record screen opens with a compact **Hold: Line ▾** menu at the top —
the *one* thing the user tells the app (chainlink #66). Wall, hand steps,
camera angle and phases are detected automatically (epic #72), so there is
no other setup question. Every hold is listed — Line, Tuck, Straddle, Stag,
Diamond, One arm — but only **Line** is recordable in the MVP; the rest are
disabled and read "coming soon" so the user can see what is planned. The
last choice is remembered per device via `@AppStorage("holdType")` and read
through `HoldType.resolved(...)`, so a fresh install shows Line. The choice
is frozen while recording, and the done screen shows it under Duration and
Frame.

Every successful recording gets a JSON **sidecar** with the same stem, in
the same `Recordings` folder:

```
20260928-143059.mov
20260928-143059.json   {"schema":1,"hold_type":"line",
                       "recorded_at":"2026-09-28T14:30:59Z","app_version":"0.1.0"}
```

Keys are snake_case, the date is ISO-8601 in UTC, and `hold_type` is the
hold that was selected when the take **started**. If the sidecar cannot be
written the video is kept — never deleted — and `errorMessage` explains the
problem. Analysis (chainlink #47) reads `hold_type` to pick the matching
scoring reference (`HoldType.referenceResourceName`, `"reference-line"` for
Line); see **Analysis** below for where that file comes from (it is never
committed — chainlink #28's real one is user data). Live mode (chainlink #91)
adds five more **optional** keys to the same sidecar — see **Live mode**
below.

## Live mode (spoken cues)

While the camera is up the phone says **one short cue** when you achieve a
line, and little else (chainlink #91, stage 1). The philosophy is the user's
(2026-10-06): *sparse encouraging audio, every cue optional*, because how much
is too much has to be tested at the gym. And the split is fixed:
**live cues only confirm — the analysis afterwards gives the critique.** No
correction is ever spoken mid-hold; the holds, the faults and the score come
from **Analyse**, which is unchanged and stays authoritative.

### What is spoken, and when

| Cue | Words | When |
| --- | --- | --- |
| Line | "Line. Hold it." | the frame has been a *line* for 0.5 s — upside down, wrist→ankle within **15°** of vertical, hips open at **≥ 160°**, wrists/hips/ankles confident. **Once per line.** |
| Time marks | "Five." / "Ten." / "Fifteen." … | every 5 s of line time while the line holds — only with **Time marks** on. |
| Framing hints | "Step back." · "Come closer." · "Move to the middle." · "Only you in the frame." · "Raise the phone." | **before the attempt**, upright and out of a line: at most one every 4 s, never the same one twice within 8 s, and never within 1.5 s of coming down out of inversion. Cut off at the top *only* while standing → "Raise the phone." (the phone is too low); any sideways edge → "Move to the middle."; too small → "Come closer."; two bodies → "Only you in the frame."; nobody → nothing (the border already says so). |
| Ready | "Ready." | the framing goes from wrong back to right while you are upright, once per transition. |

Nothing at all is said when **Voice cues** is off, except the framing hints —
they have their own toggle. Every cue is also shown as a small **banner** under
the border for **2 s**, the same words, so the screen works with the sound off
(and while the phone is muted).

### Settings

The **gear** on the home screen opens **Settings** with three toggles, stored
with `@AppStorage` and remembered per device:

- **Voice cues** — default **on**: the line cue, the time marks and "Ready.".
- **Time marks** — default **off**: the optional "Five." / "Ten." marks.
- **Framing hints** — default **on**: the pre-attempt hints above.

Under the toggles, the philosophy in one line:

> Live cues only confirm; the analysis afterwards gives the critique.

### The moving parts

- **Live pose.** MediaPipe runs on the camera frames **on the analysis queue**,
  throttled to **at most 10 fps** (configurable, and only while the Record
  screen is up), with its **own** VIDEO-mode landmarker instance — never the
  analysis one — and the capture clock's own monotonic `t_ms`, so variable
  frame rate is handled by timestamps, never by a fixed fps. One pass per
  frame only: the frame is fed 180°-turned when the *previous* live frame was
  judged upside down (`VisionPoseCore.AutoRotation`'s rule, made causal) —
  both passes are too expensive live. A build without the fetched model (the
  `.task` file is never committed) simply has no live cues and records
  exactly as it did before.
- **Pure policy.** `HandstandCore.LiveLineDetector` (the line, its 0.5 s
  onset / 0.4 s loss hysteresis, the marks) and `HandstandCore.CueScheduler`
  (every rule in the table) are pure and unit-tested — including the check
  that **every live cue on the golden fixtures falls inside a hold the offline
  `PhaseSegmenter` finds on the same input**, so the cue can never fire where
  the analysis will say "no hold".
- **Speech.** `AVSpeechSynthesizer` on an audio session of `.playback` with
  `.duckOthers` and `.mixWithOthers`: background music keeps playing (it ducks
  while a cue is said) and **an utterance is never interrupted** — the next cue
  waits its turn, which the debounce keeps short. The rate is slightly slower
  than the default so the words carry in a gym. The writer records **video
  only**, so nothing spoken reaches the take's file today; if a microphone is
  ever added to `RecordingWriter`, cues on the recording's audio are
  acceptable (they are the point of the feature, not a leak).

### Performance guard

The recording must never drop a frame because of live mode, so live analysis
drops **its own** frames first: at most one inference is ever in flight, new
frames are *skipped* while it runs (never queued), and the capture queue never
waits for MediaPipe. On top of that:

- if the **average inference over a 2 s window exceeds 80 ms**, the live rate
  **halves** (10 → 5 → 4 fps, floor 4);
- if **any frame is dropped from the recording** (`didDrop`), the live rate
  halves too.

Every halving is logged: `Handstand: live mode <reason> — rate halved to
N fps`.

### The per-take numbers

The sidecar (`20260928-143059.json`) gains five **optional** keys — schema
stays 1, and a take from a build without live mode reads exactly as before:

```json
{"schema": 1, "hold_type": "line", "recorded_at": "2026-09-28T14:30:59Z",
 "app_version": "0.1.0",
 "live_fps_target": 5.0, "live_fps_achieved": 4.8, "avg_inference_ms": 96.3,
 "recording_frames_dropped": 0,
 "cues_spoken": [{"t_ms": 1240, "cue": "line_hold"},
                 {"t_ms": 6020, "cue": "time_mark_5"}]}
```

- `live_fps_target` — the live rate in force at the end of the take (the
  guard may have halved it mid-take), `live_fps_achieved` — the live frames
  actually analysed over the take's length.
- `avg_inference_ms` — what one live inference cost on average.
- `recording_frames_dropped` — what the *recording* lost (the number the
  guard exists to keep at 0).
- `cues_spoken` — every cue spoken during the take: `t_ms` into the take and
  the cue's stable name (`line_hold`, `time_mark_5`, `step_back`, …), not the
  words, which you may reword in `CueText`.

They are shown in the session screen's **Details** disclosure (chainlink #93),
and they exist to tune one question: *how much feedback is too much?*

## History

Every recording becomes a **Session** row in a local SwiftData database
(chainlink #51), listed on the **History** screen (home → **History**):

- **What's stored**: the movie's *file name relative to the Recordings
  folder* (`20260928-143059.mov` — never an absolute path, the app
  container moves between installs while the folder name is the stable
  part), when it was taken, which hold it was for, and its duration and
  frame size as read from the file. The analysis columns (`analyzed_at`,
  `clip_score`, `hold_count`, `longest_hold_s`, `analysis_version`,
  `analysis_note`) are filled by **Analyse** (see **Analysis** below);
  until a take is analysed those rows read "Not analysed yet", and a take
  the app could not measure reads "Couldn't measure".
- **Where**: `SessionStore` in `ios/HandstandApp/Sessions/`, over the
  `ModelContainer` that `HandstandApp.swift` opens with
  `ModelConfiguration(cloudKitDatabase: .none)`.
- **Local only**: no CloudKit, no network — the rows live on the phone and
  nowhere else. The movies themselves are never moved or copied: they stay
  in `Application Support/Recordings`, and the database only points at
  them.

The header reads **"N sessions · M this week · total time mm:ss"** plus
"Best score X" once any take has been analysed (until then, "Scores appear
once analysis is available"). Rows show the date and time, the hold, the
duration and the score — or "Couldn't measure" when the last analysis could
not measure that take, "Not analysed yet" when there was none — newest
first; a row opens a detail screen that plays the video with the same
facts beside it.

`reconcile()` runs every time the screen appears: a `.mov` with no row
gets one (hold type and date from its sidecar when there is one, otherwise
Line and the file's creation date; duration and size from the file), a row
whose movie is gone is removed, and anything that is not a `.mov` is
ignored. That is how recordings taken *before* this feature — or while a
take failed to be recorded — appear as well.

**Deleting** a recording (swipe the row, or the button in the detail view;
both ask *"Delete this recording? The video is removed from the phone."*)
removes the `.mov`, its `.json` sidecar, its `.pose.json` pose cache (see
**Stress diagram** below) **and** the row. The store only
ever deletes files inside the Recordings folder — nothing else on the
phone is touched.

## Analysis

**Analyse** is the app's first end-to-end run (chainlink #47): a recording
(or a video picked from Photos) to phases, holds, a duration — and a score
when a scoring reference is available. All of it is on the phone.

What happens, in order:

1. The movie is read frame by frame by `VideoFrameSource`
   (`swift/VisionPose/…/VideoFrameSource.swift`), decoded exactly the way
   the macOS `vision-pose` runner decodes it — same `kCVPixelFormatType_32BGRA`
   pixel format, same `preferredTransform` → display orientation
   (`DisplayFrames`), same millisecond `tMs` rounding — and **capped at
   30 fps**: a frame is kept when it is at least `1000/30 − 0.5` ms after
   the last kept one (the first frame is always kept; `maxFps 0` keeps
   everything, which is what a parity check against the runner's CSV uses).
   Frames are *pulled* one at a time, so a long clip never piles up in
   memory while the model is still on frame three.
2. `PoseBackend.preferred`'s backend runs over every kept frame —
   **MediaPipe** (the default when its model is bundled, see **Pose
   backends** above) through the clip-level `MediaPipeClipExtractor`, both
   passes of the whole clip and the orientation choice; **Apple Vision**
   per frame through the kit's `PoseService` (chainlink #82), the same class
   the runner uses.
3. `Analyzer.analyze` (#42) runs the whole pipeline: post-process (with the
   backend's `minVisibility` gate), phases, features, and the scorer *if* a
   reference loaded (step 0).
4. The result is shown: **Holds: N**, **Longest hold: x.x s**, the score —
   or **"No score yet (no reference)"** — and up to three top faults with
   their names said in words (`hip_angle` → "Hip angle"). A clip the app
   *cannot* measure (no body length — for instance Vision found a person
   but never a confident enough leg) shows none of those rows: it says
   **"Couldn't measure your body in this video."** with the framing hint
   and **"Person found in X of Y frames"** (**"No person found in this
   video."** when nobody was found) instead.
5. For a recording, the result is saved to the session row's analysis
   columns (`analyzed_at`, `clip_score`, `hold_count`, `longest_hold_s`,
   `analysis_version` — the backend's own (`"mediapipe-1"` /
   `"vision-1"`, **Pose backends** above) — and `analysis_note`), which is
   what the detail screen and the History row then show. An unmeasurable clip is
   saved with hold count 0, no score and the reason in `analysis_note` —
   the screens then read "Couldn't measure" rather than a hold count of 0.
6. For a recording, the frames the model saw are also written to the
   **pose cache** beside the movie — `<basename>.pose.json`, same folder,
   same stem (see **Stress diagram** below). The next time the session
   screen opens it reads those frames instead of running Vision again.
7. For a recording, the run's numbers are written beside it too as
   `<basename>.diagnostics.json` (chainlink #93), and what the run found
   — including why it found no hold — is shown in words; both are spelled
   out under **Diagnostics and the "No hold found" explanation** below.

Worth knowing:

- **Progress and Cancel**: the run shows a percentage and a **Cancel**
  button; cancelling returns to the button and saves nothing. The screen
  stays awake while it runs (`isIdleTimerDisabled`, restored afterwards),
  and the heavy work runs off the main actor, so the UI never blocks. A
  video the model cannot measure — nobody in it, or nobody it can measure —
  finishes with the explanation above rather than a hold count of 0, so
  "Holds: 0" is never shown as if you had failed to hold; only a video that
  cannot be read fails, with the reason in words.
- **No score without a reference**: scores appear only when a scoring
  reference is available. The real one comes from your own #28 data — see
  below. Without one the analysis still finds phases, holds and features.
- **No fault classifier yet**: chainlink #34's classifier does not exist,
  so nothing classifies faults; the "top faults" are just the scorer's
  ranked features of the scored hold.
- A video picked from **Analyse a video** runs the same analysis with the
  Line hold and is *not* saved anywhere — a pick is not a recording, so it
  never appears in History.

### Diagnostics and the "No hold found" explanation

Every analysis of a **recording** also writes a few KB of numbers beside
it: `<basename>.diagnostics.json`, same folder, same stem as the movie
(chainlink #93). It is what the lead pulls off the phone with `xcrun
devicectl` to see why a take went the way it did, and what the session
screen's **Details** disclosure shows.

```json
{"schema": 1, "app_version": "0.1.0", "backend": "mediapipe",
 "analysis_version": "mediapipe-1", "analysis_wall_time_s": 6.42,
 "video": {"duration_s": 24.316, "fps": 59.94, "width": 1080, "height": 1920},
 "analysed_fps": 29.7, "frames_total": 714, "frames_with_person": 281,
 "frames_with_person_pct": 39.4,
 "unknown_reasons": {"no_visible_wrist": 609, "no_visible_ankle": 310},
 "out_of_frame": {"partly": 62, "not_in_frame": 433, "edges": {"top": 40}},
 "dominant_reason": "not_in_frame", "hold_count": 0, "hold_durations_s": [],
 "usable": true, "unusable_reason": ""}
```

- **Counts only.** No keypoints (those are `.pose.json`'s) and no
  per-frame rows, so the file is a few KB whatever the take — a thousand
  frames cost no more than sixty. Local to the phone, never sent
  anywhere; `SessionStore.delete` removes it with the recording and
  `reconcile()` ignores it (it keeps only the movie's own extension).
- **Two frame rates.** `video.fps` is what the camera recorded at (about
  60 on the iPhone), `analysed_fps` what the analysis was actually fed at
  (the 30 fps cap, measured over the frames' own timestamps);
  `frames_with_person_pct` is the share of frames with a person in them.
- **The out-of-frame reason.** The phase segmenter only ever reports the
  *symptom* — `no_visible_wrist`, `no_visible_ankle`, … — because it is
  not told how big the picture is. `Framing` puts the two together: a
  frame counts as *partly out of frame* when a wrist, ankle or hip is
  missing **and** the visible body touches a border (any joint within 3%
  of an edge), and as *not in frame* when nothing was detected. The
  edges it was cut off at (`top` / `bottom` / `left` / `right`) are in
  `out_of_frame.edges`.
- **`dominant_reason`** is the reason for *no hold*: the bucket the most
  unseen frames fell in when **more than half** the take could not be
  seen into, `not_inverted` otherwise (visible, just never upside down
  long enough), and `""` when there was a hold or the clip could not be
  measured. A frame that was out of frame counts as *out of frame* rather
  than as the missing joint, so the framing outranks its own symptom.

**"No hold found"** (the same words on the panel right after a run and in
the session screen for a take analysed earlier): `NoHoldReason` maps each
reason to one sentence, and the message ends in **one** setup tip from a
three-entry table — distance/height (`positionTip`) for anything out of
frame or never inverted, light/contrast (`lightTip`) for joints the model
could not read, `clearViewTip` for a blocked camera:

> **No hold found**
> You were partly out of frame (top and right edges) for most of the
> take. Place the phone about 3 m away at hip height, so your whole body
> fits upside down.

A take the app could *not measure* never reaches this: it says
**Couldn't measure** with its own words instead, as it always has.

### The scoring reference (never committed)

The repo is public and the reference is built from your own labelled clips
(#28), so it is **never committed**. To score, put your reference in the
app's resource folder on *your* clone before building:

```
ios/LocalResources/reference-line.json     # one file per hold:
                                           # reference-<hold>.json
```

`ios/LocalResources/` is in `.gitignore` and declared in `project.yml` as
an **optional** resource folder, so `xcodegen generate` and the build work
when the folder is absent — scores then read "No score yet (no reference)".
`ReferenceLoader` looks the file up in the built bundle
(`<hold.referenceResourceName>.json`), decodes it with
`ScoreReference.decode`, and returns `nil` when it is missing *or* invalid:
an invalid file is logged once, never a crash. The unit tests use the
synthetic `parity_reference.json` inlined as a string instead — no test
needs (or gets) the real one.

## Stress diagram

Open an **analysed** recording and the movie plays with the athlete's
skeleton drawn over it (chainlink #48): bones and joints coloured by how far
the form is off, joints *sized* by it, the vertical **stack line** through
the hands, and the **centre of mass** — a dot with an arrow to where it
projects onto the hand line, coloured by which side of the base of support
it is on.

### What the colours mean

- **Green** — on target, and the CoM inside the base of support.
- **Amber** — off, but not badly; the CoM in front of the fingertips is
  amber's twin the other way: *under* (behind the heel) is amber, *over*
  (past the fingers) is **red**.
- **Red** — badly off.
- **Grey** — not held or not measured: outside a hold the whole skeleton
  turns grey, because a warm-up is not a fault.

Joints are circles of `4 + 8 × severity` points (4 pt when grey), so how
bad a fault is reads as size as well as colour. The legend under the player
says all of this in a row of swatches, plus the note line: **"Colours
compare with your reference"** when the build has a scoring reference (#28),
or **"Colours use built-in form thresholds (no reference yet)"** until it
does. The exact severity rules (z-scores with a reference, `TOLERANCES`
without one) live in `HandstandCore.StressDiagram` and are documented in
docs/swift.md.

The **Diagram** button beside the scrubber hides and shows the overlay (on
by default whenever there is one); play/pause and the scrubber move the
playhead, and a `1/30 s` time observer picks the frame to draw — the same
timestamps the analysis was made from, so the skeleton follows the body.

**Side view only, for now**: every clip in the dataset is a side view, so
the diagram measures sagittal-plane alignment only; front/back alignment is
chainlink #83. The *ideal* "ghost" skeleton you would compare against is
chainlink #28's — the model has the slot for it (`DiagramFrame.ideal`) and
nothing fills it yet.

### The pose cache

Analysing a take runs the pose model over every frame — seconds of work that
must not be repeated just to *watch* the overlay again. So a finished run
writes the frames it read to **`<basename>.pose.json` next to the movie**
(`20260928-143059.mov` → `20260928-143059.pose.json`, same folder):

```json
{"schema": 1, "backend": "mediapipe", "analysis_version": "mediapipe-1",
 "max_fps": 30,
 "frames": [{"t_ms": 0, "detected": true,
             "joints": {"nose": [x, y, visibility], …}}, …]}
```

(`backend`/`analysis_version` follow whichever backend ran — `"vision"` /
`"vision-1"` while Vision is the fallback — and a cache whose
`analysis_version` is not this build's is not read: **Pose backends** above
is what makes old `"vision-1"` caches recompute once MediaPipe decides.)

- **Written** by `AnalysisService` when the extraction finishes — for a
  *recording* only; a video picked from Photos has no row in History and so
  gets no cache and no diagram (out of scope for now).
- **Read** on the session screen's appear (and after a run): the frames go
  through `Analyzer.analyze` again — milliseconds, no Vision — with this
  build's reference, so the overlay can never go stale when a new reference
  lands. `PoseCache.read` answers *nil* (and the screen simply waits for
  "Analyse") on a missing file, bad JSON, or a `schema` / `analysis_version`
  this build does not know.
- **Deleted** with the recording, like the `.json` sidecar. `reconcile()`
  ignores it — it is not a `.mov`, so it never becomes a History row.

It is local to the phone, exactly like the video: no network, no copies.

## Summary

Under the player of an **analysed** session — the same condition as the
diagram: an analysis exists — the screen shows three things
(chainlink #49, the logic in docs/swift.md's **SessionSummary** section):

1. **The heat strip** (`HeatStripView`): a full-width, 16 pt bar of the
   whole clip, one coloured bin per slice of it in the **same colours as
   the diagram** (grey = not held, green = on form, amber = off, red =
   far off — the bands come from `SessionSummary.heatStrip`, the same
   severity the overlay draws), a thin white **playhead** at the clock's
   current time, and the caption **"Form over time (holds coloured)"**.
   Tapping or dragging on the strip **seeks** the player to that moment —
   the x↔time mapping is the pure `HeatStripGeometry` (unit-tested), so
   the line you tap and the bins you see are the same scale.
2. **The worst moment** (`WorstMomentCard`): a thumbnail of the worst
   frame of the hold with the stress diagram drawn over it
   (`StressDiagramOverlay` aspect-fits any size), a label like
   **"Worst moment · 0:04.2"** plus what was off in plain words
   (`FaultLabel`: "Hip angle, Line deviation"), and a tap that seeks the
   player **there and pauses**. The card is hidden when there is no worst
   moment — no hold, or a hold with nothing wrong.
3. **"What to work on"**: up to three numbered coaching cues
   (`CoachingCues.cues` — the longest hold's faults, plus the balance
   cues), one line each, in plain language.

Everything is computed **once per analysis** — on opening an analysed
take and again after **Analyse again** — and nothing is recomputed while
the movie plays except the playhead. Side view only for now, like the
rest of the analysis (front/back is chainlink #83); the names are
view-neutral.

## Export

An **analysed** session — the same condition as the diagram and the
summary — gets an **Export video** button in its own section
(chainlink #50; the library side is docs/swift.md's **AnnotatedVideo**
section). It writes an annotated copy of the movie: the picture with the
**stress diagram burned into every frame** and the **heat strip along the
bottom**, as an H.264 `.mp4`.

- **What is in it**: every frame of the original video, at the source's own
  frame timing, upright at its display size (a sideways recording comes out
  upright too), with the skeleton, the dashed stack line and the centre of
  mass drawn exactly as the overlay draws them — the same band colours and
  sizes, scaled to the video's resolution (`scale = videoHeight / 1000`) —
  the heat strip along the bottom 3 % with a white playhead at each frame's
  time, and a small footer bottom-left: "Handstand · 2026-09-30", plus
  "Score 78" when the clip has one. The source's audio comes along when the
  container can carry it unchanged (AAC).
- **Where it is written**: the app's **Caches** directory, as
  `<basename>-annotated.mp4` — never in Recordings, never a History row,
  and nothing is uploaded anywhere.
- **Progress and Cancel**: the export reports 0…100 % and Cancel stops it
  at once; a cancelled (or failed) export deletes its own partial file, so
  nothing half-written is ever left behind.
- **Sharing is up to the user**: when it finishes the section shows
  **Share video** (`ShareLink`), which opens the share sheet — save it to
  Photos, send it to a coach, whatever happens after that is the user's.
  The `NSPhotoLibraryAddUsageDescription` string ("Save your annotated
  handstand video to Photos.") is what makes **Save Video** in that sheet
  work.
- **Deleted afterwards**: the cached export goes when the session screen
  goes away, when a new export starts, and when the recording is deleted —
  `SessionStore.delete` takes a leftover `<basename>-annotated.mp4` with the
  movie. The copy in Photos (or wherever it was shared to) is the user's;
  the app keeps none.

An analysis the pipeline could not measure still exports: the plain video
with no skeleton and a neutral strip, rather than a button that does
nothing.

## Testing recording on the phone

The camera does not exist in the simulator, so recording and its framing
guide are checked on the iPhone itself (sideload as above). The checklist:

1. Open **Record** and **grant camera access** when iOS asks. If access was
   refused earlier, the screen explains and offers **Open Settings** — turn
   the camera on there and come back.
2. With your **whole body in frame** the border around the preview turns
   **green** and the message reads "Looks good".
3. **Cut your feet off** (step close, or stand at the bottom of the
   picture): the border turns **amber** and the message names the problem —
   "Step back: your feet are out of frame", or "you are cut off at the
   bottom of the frame". Standing far away turns it amber with "Move
   closer"; a second person walking in turns it red.
4. **Record ~10 s**: tap the record button, wait, tap stop. The done screen
   shows about **0:10** and a frame of **1080 × 1920**.
5. **Record again** returns to the live camera; **Done** goes back home.
6. **Check the hold picker** at the top reads **Hold: Line** (the other
   holds are listed but disabled, "— coming soon"), and that after Stop a
   `.json` sidecar with the same name appears next to the `.mov` in the
   Recordings folder (`20260928-143059.json` beside
   `20260928-143059.mov`).
7. **Check History**: record two takes, open **History** and check both
   appear (newest first, with date and time, hold, duration and "Not
   analysed yet"), then delete one — confirm the dialog — and check it is
   gone from the list and so is its `.mov` in the Recordings folder.

Recordings land in `Application Support/Recordings/<yyyyMMdd-HHmmss>.mov`
on the phone and stay there — nothing is uploaded, and the folder is
deliberately *not* excluded from an iCloud backup.

## Testing live mode on the phone (the "Live mode test" checklist)

The simulator has no camera, so the cues are felt on the iPhone itself
(sideload as above). Five things to check:

1. **Record a line hold with voice on.** Open **Record** with **Voice cues**
   on (gear on Home), kick up into a line and hold it for a few seconds: you
   must hear **"Line. Hold it."** once, about half a second *after* you are
   straight — and only once for that hold. The same words appear as the small
   banner under the border.
2. **Check the cue timing.** The cue must come when you are actually in the
   line (not on the way up, not after you come down), a brief wobble must not
   produce a second cue, and **Analyse** on that take must still find the hold
   the cue announced — the analysis stays the authority on what you held.
3. **Try silent mode.** Turn **Voice cues** off: nothing is spoken during the
   attempt, and a pre-attempt framing hint (its own toggle) is still shown as
   a banner. With the phone's own mute switch on, check whether the cues are
   still audible (the `.playback` session says they should be) and that the
   banner always carries the words anyway.
4. **Try time marks.** Turn **Time marks** on and hold a line past 5 s: you
   should hear "Five." (and "Ten." past 10 s), each once, and never a mark
   while you are not in the line.
5. **Read the live stats in Details.** Open the take from **History** and
   expand **Details**: `Live rate` (target → achieved fps), `Live inference`
   (ms), `Recording dropped` (frames — should read 0) and `Cues` with each
   cue's name and second. The same numbers are in the sidecar
   (`<take>.json`) for `jq`.

## Testing analysis on the phone

Analysis needs a real recording and the Vision model on real frames, so
this checklist is for the iPhone (sideload as above):

1. **Record ~10 s** (the recording checklist above), then open the take
   from **History** → its row.
2. Tap **Analyse** (it reads "Analyse again" after the first run). The
   progress percentage must **move** — it starts at 0 %, climbs and reaches
   100 % — the screen must **stay awake** while it runs, and the result
   must show **Holds: 1** for a clip with one clean hold, and a
   **Longest hold** of a few seconds that matches how long you actually
   held. A clip the camera could not measure shows **"Couldn't measure your
   body in this video."** (with how many frames had a person in it)
   instead of any hold count.
3. Tap **Analyse again** and, while it is running, tap **Cancel**: the
   screen goes straight back to the button, and nothing was saved (the
   row's numbers, if any, are unchanged).
4. Go back to **History**: the row now shows the score (or still "Not
   analysed yet" when the build has no reference — see **Analysis**), and
   re-opening the session shows the run's saved numbers again — **Holds**,
   **Longest hold** and **Score** (the top faults are words, not stored on
   the row, so they are shown by the run that produced them).
5. Home → **Analyse a video** → pick a video from Photos → **Analyse**: the
   same progress, Cancel and result panel, with the duration and frame
   size still shown above it. Nothing appears in History — a pick is not a
   recording.
6. **Play an analysed session** with the diagram on (chainlink #48): tap
   ▶ and check the **skeleton follows your body** through the clip, and
   that it **turns grey outside the hold** (the kick-up and the landing)
   while the stack line stays down your hands. Toggle **Diagram** off and
   on, then leave and re-open the session: the overlay comes straight back
   from the `.pose.json` cache, without the analysis running again.
7. **Read the summary under the player** (chainlink #49): analyse a
   session, then check the **heat strip's colours match the diagram**
   while playing (grey outside the hold, green/amber/red in it, white
   playhead moving over it) and tap the strip to **seek**; **tap the
   worst moment** card and check the player jumps to that frame and
   **pauses**; and read the numbered cues under **"What to work on"**.
   Tap **Analyse again** and check the strip, card and cues all refresh.
8. **Export an analysed session** (chainlink #50): with an analysis on
   screen, tap **Export video** — the progress bar must climb from 0 to
   100 %, and **Cancel** mid-way must go straight back to the button with
   no file left behind. When it finishes tap **Share video** → **Save
   Video**, then open the clip **in Photos** and check the **skeleton
   lines up with your body** (upright and unmirrored, even for a recording
   made sideways), the **strip runs along the bottom** with its white
   playhead, and the footer shows the date. Leave the session and come
   back: the export is gone from the phone's Caches and can be made again
   with one tap.

Everything stays on the phone: no networking, no analytics, recordings are
not uploaded anywhere. The app ships the home, settings, record (with live
cues, chainlink #91), history, analysis, stress-diagram overlay, summary
(heat strip, worst moment, cues), annotated video export, video-pick and
about screens.
