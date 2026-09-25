# handstands

Handstand form analysis for iPhone: record an attempt, estimate 2D pose (upside-down frames rotated 180°),
extract biomechanical features, and score the hold against a personal "ideal handstand" reference.

## Layout

| Path | Contents |
|---|---|
| `pipeline/` | Python reference pipeline: pose extraction, features, phases, scoring, classifier training |
| `swift/` | `HandstandCore` Swift package, the on-device port of the pipeline (parity-tested against Python fixtures) |
| `ios/` | iOS app (SwiftUI) |
| `tools/` | Labeling helpers, visualization, one-off scripts |
| `docs/` | Design notes, bake-off results, recording protocol |
| `data/` | Local only (git-ignored) except `splits.json` |

Raw videos are kept out of git. Work is tracked with chainlink in `.chainlink/`.

## Machines

- Linux workstation: Python pipeline, training.
- Mac mini M1 (Xcode 27): Apple Vision runner, Swift package, iOS builds.
