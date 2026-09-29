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

Everything stays on the phone: no networking, no analytics, recordings are
not uploaded anywhere. The scaffold ships the home, record-placeholder,
video-pick and about screens; capture (chainlink #46), analysis (#47),
overlay (#48) and storage (#51) come later.
