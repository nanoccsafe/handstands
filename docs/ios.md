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
Line); no reference is bundled yet (chainlink #28).

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

Recordings land in `Application Support/Recordings/<yyyyMMdd-HHmmss>.mov`
on the phone and stay there — nothing is uploaded, and the folder is
deliberately *not* excluded from an iCloud backup.

Everything stays on the phone: no networking, no analytics, recordings are
not uploaded anywhere. The app ships the home, record, video-pick and about
screens; analysis (chainlink #47), overlay (#48) and storage (#51) come
later.
