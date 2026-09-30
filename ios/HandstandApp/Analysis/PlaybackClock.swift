import Foundation
import Observation

// --------------------------------------------------------------------------- #
// Where playback is, for the diagram (#48).
//
// The overlay is drawn from the clip's own clock — the `t_ms` of the frames
// the analysis was built from, the same timeline the pose cache was written
// on, with video time starting at 0. The player's periodic time observer
// (every 1/30 s, `SessionDetailView`) feeds this class, and the view reads
// `playheadMs` to ask `StressDiagram.frameIndex` which frame to draw.
//
// A class rather than view state because the AVFoundation callback is not a
// SwiftUI context: it hops onto the main actor (`MainActor.assumeIsolated`
// — the observer is installed on the main queue) and updates *this*, and
// `@Observable` is what repaints the view when it changes.
// --------------------------------------------------------------------------- #

/// The playhead in both spellings the overlay needs, updated by the player's
/// time observer.
@MainActor
@Observable
final class PlaybackClock {
    /// The playhead in milliseconds on the clip's clock — what
    /// `StressDiagram.frameIndex(atMs:in:)` looks a frame up in.
    var playheadMs = 0
    /// The same playhead in seconds, for the scrubber to read and write.
    var seconds = 0.0

    /// Moves the clock to `seconds` of playback — the one place the two
    /// columns are kept in step (`t_ms = seconds × 1000`, truncated, the
    /// way a timestamp of a moment is a whole millisecond).
    func tick(_ seconds: Double) {
        let clamped = seconds.isFinite ? Swift.max(0, seconds) : 0
        self.seconds = clamped
        playheadMs = Int(clamped * 1000)
    }
}
