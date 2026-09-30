# The iOS app

`ios/` is the iPhone app: a SwiftUI shell around the `HandstandCore` package
(`swift/HandstandCore/`). The `.xcodeproj` is **generated** from
`ios/project.yml` with XcodeGen and is never committed — edit `project.yml`,
not the project.

## Generate and open (on the Mac)

```bash
export PATH="/opt/homebrew/bin:$PATH"   # XcodeGen lives in homebrew
cd ios
xcodegen generate
open Handstand.xcodeproj
```

Re-run `xcodegen generate` whenever `project.yml` changes (including after a
pull); Xcode picks up the regenerated project.

## Build and test from Linux

```bash
tools/mac/ios_build.sh          # rsync ios/ + swift/ to the Mac, xcodegen,
                                # simulator build, HandstandAppTests
tools/mac/ios_build.sh wt/i44   # the same, spelled out
```

It picks an available iPhone simulator from `xcrun simctl list devices
available` (nothing hard-coded) and exits non-zero if generation, the build
or the tests fail, printing `** BUILD SUCCEEDED **`, `** TEST SUCCEEDED **`
and the `Executed … tests` lines.

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
committed — chainlink #28's real one is user data).

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
removes the `.mov`, its `.json` sidecar **and** the row. The store only
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
   memory while Vision is still on frame three.
2. Apple Vision — the `PoseService` of chainlink #82, the same class the
   runner uses — runs on every kept frame.
3. `Analyzer.analyze` (#42) runs the whole pipeline: post-process, phases,
   features, and the scorer *if* a reference loaded (step 0).
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
   `analysis_version = "vision-1"`, `analysis_note`), which is what the
   detail screen and the History row then show. An unmeasurable clip is
   saved with hold count 0, no score and the reason in `analysis_note` —
   the screens then read "Couldn't measure" rather than a hold count of 0.

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

Everything stays on the phone: no networking, no analytics, recordings are
not uploaded anywhere. The app ships the home, record, history, analysis,
video-pick and about screens; the stress-diagram overlay (chainlink #48)
comes next.
