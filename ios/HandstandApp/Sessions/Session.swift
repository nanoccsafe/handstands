import Foundation
import HandstandCore
import SwiftData

/// One recording as a row in the local database (chainlink #51): what the
/// History screen lists. The row stores the movie's **file name relative to
/// the Recordings folder** — `20260928-143059.mov`, never an absolute path,
/// because the app container moves between installs while
/// `Application Support/Recordings/<name>` (`RecordingFile.swift`) is the
/// stable part. The database only *points* at the movie: the bytes stay
/// where chainlink #46 wrote them, and deleting the row takes the file with
/// it (`SessionStore.delete`).
///
/// Everything is on the phone: the container is opened with
/// `cloudKitDatabase: .none`, there is no network code behind this model.
///
/// The analysis columns (`analyzedAt` … `analysisVersion`) are filled in by
/// chainlink #47 once scoring runs on the device; until then they are `nil`,
/// which the History screens read as "Not analysed yet".
@Model
final class Session {
    /// The movie's name inside the Recordings folder, e.g.
    /// `20260928-143059.mov`. Unique — one row per file, whoever adds it
    /// (the recorder right after a take, or `reconcile()` for a file made
    /// before this feature existed).
    @Attribute(.unique) var movieFilename: String
    /// When the take started: the sidecar's `recorded_at` when there is one,
    /// otherwise the movie's creation date.
    var recordedAt: Date
    /// The hold as a `HoldType` raw value (`"line"`); read through the
    /// `holdType` accessor, so a value written by a future build falls back
    /// to Line instead of failing.
    var holdTypeRaw: String
    /// Length in seconds and the frame size in pixels, read from the movie
    /// itself (`VideoInfoReader`) — the same numbers the done screen shows.
    var durationS: Double
    var width: Int
    var height: Int

    // MARK: - Analysis (chainlink #47; nil = not analysed yet)

    var analyzedAt: Date?
    /// The clip score, the one number History shows next to a take.
    var clipScore: Double?
    var holdCount: Int?
    var longestHoldS: Double?
    /// Which scoring run wrote the values above, for when the algorithm
    /// changes and old scores must be told apart from new ones.
    var analysisVersion: String?

    /// The hold as a `HoldType`: unknown raw values (and holds the MVP
    /// cannot record yet) read as Line, via `HoldType.resolved`.
    var holdType: HoldType { HoldType.resolved(holdTypeRaw) }

    init(
        movieFilename: String,
        recordedAt: Date,
        holdTypeRaw: String = HoldType.default.rawValue,
        durationS: Double = 0,
        width: Int = 0,
        height: Int = 0,
        analyzedAt: Date? = nil,
        clipScore: Double? = nil,
        holdCount: Int? = nil,
        longestHoldS: Double? = nil,
        analysisVersion: String? = nil
    ) {
        self.movieFilename = movieFilename
        self.recordedAt = recordedAt
        self.holdTypeRaw = holdTypeRaw
        self.durationS = durationS
        self.width = width
        self.height = height
        self.analyzedAt = analyzedAt
        self.clipScore = clipScore
        self.holdCount = holdCount
        self.longestHoldS = longestHoldS
        self.analysisVersion = analysisVersion
    }
}
