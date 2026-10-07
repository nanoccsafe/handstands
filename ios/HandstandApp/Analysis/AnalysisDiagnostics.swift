import Foundation
import HandstandCore

// --------------------------------------------------------------------------- #
// What one analysis saw (chainlink #93): a few KB of numbers beside the
// recording — `<basename>.diagnostics.json`, same folder, same stem as the
// movie, the `.json` sidecar and the `.pose.json` cache.
//
//     {"schema": 1, "app_version": "0.1.0", "backend": "mediapipe",
//      "analysis_version": "mediapipe-1", "analysis_wall_time_s": 6.42,
//      "video": {"duration_s": 24.316, "fps": 59.94, "width": 1080,
//                "height": 1920},
//      "analysed_fps": 29.7, "frames_total": 714, "frames_with_person": 281,
//      "frames_with_person_pct": 39.4,
//      "unknown_reasons": {"no_visible_wrist": 609, "no_visible_ankle": 310},
//      "out_of_frame": {"partly": 62, "not_in_frame": 433,
//                       "edges": {"top": 40, "right": 31}},
//      "dominant_reason": "not_in_frame",
//      "hold_count": 0, "hold_durations_s": [], "usable": true,
//      "unusable_reason": ""}
//
// Counts only: no keypoints (those are the pose cache's) and no per-frame
// rows, so the file stays a few KB whatever the take — the point of it is
// that it can be read off the phone (`xcrun devicectl`) without becoming a
// second copy of the take. It is local to the phone and never sent
// anywhere: `SessionStore.delete` removes it with the recording, and
// `reconcile()` ignores it (it keeps only the movie's own extension).
// --------------------------------------------------------------------------- #

/// The pixel size the "out of frame" test is measured against — the display
/// frame the model saw its keypoints in, which is the movie's own size once
/// the track's transform has turned it upright.
struct FrameSize: Equatable, Sendable {
    var width: Int
    var height: Int

    init(width: Int, height: Int) {
        self.width = width
        self.height = height
    }

    /// How close to an edge — as a fraction of the frame's width (left and
    /// right) or height (top and bottom) — a joint may sit before the body
    /// counts as touching that edge. 3%: far enough that a person standing
    /// calmly in the middle of the picture never trips it, near enough that
    /// a hand or foot already clipped by the border does.
    static let edgeMargin = 0.03

    /// The frame edges `x, y` is standing on (or over — a coordinate past
    /// the edge is beyond it on that side, and the comparisons below catch
    /// it too). Empty for a size the analysis never had.
    func edges(x: Double, y: Double) -> Set<Edge> {
        guard width > 0, height > 0, x.isFinite, y.isFinite else { return [] }
        let marginX = Double(width) * Self.edgeMargin
        let marginY = Double(height) * Self.edgeMargin
        var found: Set<Edge> = []
        if y <= marginY { found.insert(.top) }
        if y >= Double(height) - marginY { found.insert(.bottom) }
        if x <= marginX { found.insert(.left) }
        if x >= Double(width) - marginX { found.insert(.right) }
        return found
    }
}

/// What one frame says about the framing (chainlink #93's out-of-frame
/// reason), read off the keypoints and the phase segmenter's own answer
/// about that frame.
///
/// The segmenter can only report the *symptom* — which joint it could not
/// see (`no_visible_wrist`, …) — because it is not told how big the frame
/// is. This is where the symptom and the picture are put together: a take
/// where the athlete was at the edge of a floor-level phone reads as
/// "missing wrist" frame by frame and as "out of frame" over the take.
enum Framing: Equatable, Sendable {
    /// Nothing was detected: nobody in the picture at all.
    case notInFrame
    /// A person was found, at least one of the wrists, ankles or hips was
    /// not, and the visible body touches a frame border — the take is cut
    /// off at these edges.
    case outOfFrame([Edge])
    /// Anything else: a person wholly inside the picture, or a frame lost
    /// to something that is not the framing (a trainer in front of the
    /// camera), or a frame whose size the analysis did not have — without
    /// the size there is no border to touch, so nothing is claimed.
    case inFrame

    /// The joints a phase is read off: when one of these is missing and the
    /// body touches a border, the joint did not wander off the picture —
    /// the picture ended. Spelled once from `NoHoldReason`, so the reason
    /// and the diagnosis can never disagree.
    static let missingJointReasons = [
        NoHoldReason.noVisibleWrist.rawValue,
        NoHoldReason.noVisibleAnkle.rawValue,
        NoHoldReason.noVisibleHip.rawValue,
    ]

    /// - Parameters:
    ///   - frame: the keypoints as the model reported them, in display pixels.
    ///   - phaseReason: why the segmenter could not see into this frame —
    ///     `""` when it could (`FrameSignals.unknownReason`).
    ///   - size: the frame's pixel size, `nil` when the movie could not be
    ///     read (then only "nobody detected" is claimed).
    init(frame: PostProcessInputFrame, phaseReason: String, size: FrameSize?) {
        guard frame.detected else {
            self = .notInFrame
            return
        }
        guard Self.missingJointReasons.contains(phaseReason), let size else {
            self = .inFrame
            return
        }
        var hits: Set<Edge> = []
        for point in frame.joints.values {
            hits.formUnion(size.edges(x: point.x, y: point.y))
            if hits.count == Edge.allCases.count { break }
        }
        guard !hits.isEmpty else {
            self = .inFrame
            return
        }
        // `Edge.allCases` order, never `Set` iteration order: the same clip
        // must always name its edges the same way round.
        self = .outOfFrame(Edge.allCases.filter(hits.contains))
    }
}

/// The analysis of one take as the small JSON beside its recording
/// (chainlink #93) — numbers only, local only, never sent anywhere.
struct AnalysisDiagnostics: Codable, Equatable, Sendable {
    /// The document's schema version — bumped when the *shape* changes.
    static let schema = 1

    var schema: Int
    /// The app's marketing version, exactly as the sidecar spells it.
    var appVersion: String
    /// Which pose backend ran (`"mediapipe"` / `"vision"`).
    var backend: String
    /// Which scoring run wrote these numbers (`"mediapipe-1"`, …).
    var analysisVersion: String
    /// The whole run's wall time in seconds — decode, model and pipeline,
    /// the number the "30 s analysed in 5–10 s" baseline is read against.
    var analysisWallTimeS: Double
    /// The movie: its length, its own frame rate (what the camera recorded
    /// at) and its frame size.
    var video: Video
    /// The rate the frames were actually *analysed* at — the app keeps at
    /// most 30 of them per second, so a 60 fps recording is analysed at
    /// about half that. Recorded fps is `video.fps`; both are here so the
    /// gap is one line to read.
    var analysedFps: Double
    /// How many frames went into the pipeline, and how many of them had a
    /// person in them.
    var framesTotal: Int
    var framesWithPerson: Int
    /// `framesWithPerson` as a percentage of `framesTotal`, 1 decimal.
    var framesWithPersonPct: Double
    /// Why the segmenter could not see into a frame, counted verbatim
    /// (`no_visible_wrist`, `no_visible_ankle`, `no_visible_hip`,
    /// `trainer_contact`, …) — one key per reason it gave.
    var unknownReasons: [String: Int]
    /// The out-of-frame reason, computed from the keypoints: frames with a
    /// person and a missing wrist/ankle/hip that touch a border (`partly`),
    /// frames with nobody in them at all (`notInFrame`), and how many of
    /// the partly frames touched each edge.
    var outOfFrame: OutOfFrame
    /// Why this take had no hold: `NoHoldReason`'s raw value, or `""` when
    /// there was a hold to show or the clip could not be measured (then
    /// `unusableReason` is the answer).
    var dominantReason: String
    var holdCount: Int
    /// Every hold's length in seconds, in time order.
    var holdDurationsS: [Double]
    /// Was the clip measurable at all? `unusableReason` is `""` exactly
    /// when this is `true` — the same two facts the screens read.
    var usable: Bool
    var unusableReason: String

    // MARK: - Building

    /// The movie's own numbers, as the document spells them.
    struct Video: Codable, Equatable, Sendable {
        var durationS: Double
        var fps: Double
        var width: Int
        var height: Int

        private enum CodingKeys: String, CodingKey {
            case durationS = "duration_s"
            case fps
            case width
            case height
        }
    }

    /// The out-of-frame counts, as the document spells them.
    struct OutOfFrame: Codable, Equatable, Sendable {
        /// Frames with a person, a missing wrist/ankle/hip and the visible
        /// body touching a border.
        var partly: Int
        /// Frames with nobody detected in them at all.
        var notInFrame: Int
        /// How many partly-out-of-frame frames touched each edge, keyed by
        /// `Edge.rawValue` (`"top"`, `"bottom"`, `"left"`, `"right"`).
        var edges: [String: Int]

        private enum CodingKeys: String, CodingKey {
            case partly
            case notInFrame = "not_in_frame"
            case edges
        }
    }

    /// One pass over the take: every count the document carries, plus the
    /// dominant reason behind `dominantReason` and the no-hold message.
    ///
    /// Built once per document and once per "No hold found" explanation —
    /// one linear pass over a few thousand frames, no allocation per frame
    /// beyond the dictionaries themselves.
    struct Tally: Equatable, Sendable {
        /// The segmenter's own words, counted verbatim.
        var unknownReasons: [String: Int] = [:]
        /// Frames nobody was detected in.
        var notInFrame: Int = 0
        /// Frames with a person, a missing wrist/ankle/hip and the visible
        /// body touching a border.
        var partlyOutOfFrame: Int = 0
        /// How many of those touched each edge, keyed by `Edge.rawValue`.
        var outOfFrameEdges: [String: Int] = [:]
        /// One count per reason, over the frames that could not be seen
        /// into — **exclusive**: a frame the athlete was out of frame in
        /// counts as out of frame rather than as the missing joint the
        /// segmenter had to report. That is what lets the framing outrank
        /// `no_visible_wrist` when it is what the take was really about.
        var buckets: [String: Int] = [:]
        /// The bucket the most frames fell in, `""` when nothing was
        /// unknown. Ties go to the earlier `NoHoldReason` case — the
        /// framing first, because on a phone the missing joint is usually
        /// the symptom of it.
        var dominant: String = ""
        /// The edges the dominant `out_of_frame` bucket was cut off at, in
        /// `Edge.allCases` order.
        var dominantEdges: [Edge] = []

        init(analysis: Analysis, frames: [PostProcessInputFrame], size: FrameSize?) {
            let signals = analysis.phases.signals
            for (index, frame) in frames.enumerated() {
                let seen = index < signals.known.count
                let known = seen ? signals.known[index] : true
                var reason = ""
                if index < signals.unknownReason.count {
                    reason = signals.unknownReason[index]
                }
                if seen, !known {
                    unknownReasons[reason, default: 0] += 1
                }
                switch Framing(frame: frame, phaseReason: reason, size: size) {
                case .notInFrame:
                    notInFrame += 1
                    buckets[NoHoldReason.notInFrame.rawValue, default: 0] += 1
                case .outOfFrame(let edges):
                    partlyOutOfFrame += 1
                    for edge in edges {
                        outOfFrameEdges[edge.rawValue, default: 0] += 1
                    }
                    buckets[NoHoldReason.outOfFrame.rawValue, default: 0] += 1
                case .inFrame:
                    // A frame the module *could* see into has no reason to
                    // explain: it is not part of why there was no hold.
                    if !known, !reason.isEmpty {
                        buckets[reason, default: 0] += 1
                    }
                }
            }
            dominant = Self.largest(buckets)
            if dominant == NoHoldReason.outOfFrame.rawValue {
                dominantEdges = Edge.allCases.filter {
                    (outOfFrameEdges[$0.rawValue] ?? 0) > 0
                }
            }
        }

        /// The reason to say when this take had no hold: the dominant
        /// bucket when **more than half** the take could not be seen into —
        /// only then is "…for most of the take" true — and "never upside
        /// down long enough" when the take was visible well enough for that
        /// to be the answer.
        func noHoldReason(frameCount: Int) -> NoHoldReason {
            let unseen = buckets.values.reduce(0, +)
            guard Double(unseen) > Double(frameCount) * 0.5,
                let reason = NoHoldReason(rawValue: dominant)
            else { return .notInverted }
            return reason
        }

        /// The whole "No hold found" message: the reason in plain words,
        /// then one setup tip.
        func noHoldMessage(frameCount: Int) -> String {
            let reason = noHoldReason(frameCount: frameCount)
            let edges = reason == .outOfFrame ? dominantEdges : []
            return NoHoldExplanation(reason: reason, edges: edges).text
        }

        /// The bucket the most frames fell in — ties to the earlier
        /// `NoHoldReason` case, then alphabetically, so the answer never
        /// depends on dictionary order.
        private static func largest(_ buckets: [String: Int]) -> String {
            guard let best = buckets.values.max(), best > 0 else { return "" }
            let tied = buckets.filter { $0.value == best }.map(\.key)
            let rank = { (reason: String) in
                NoHoldReason.allCases.firstIndex { $0.rawValue == reason }
                    ?? NoHoldReason.allCases.count
            }
            return tied.min { first, second in
                rank(first) == rank(second) ? first < second : rank(first) < rank(second)
            } ?? ""
        }
    }

    /// The document for one run.
    ///
    /// - Parameters:
    ///   - analysis: what the pipeline answered.
    ///   - frames: the frames that went into it (their timestamps give
    ///     `analysedFps`, their `detected` flag `framesWithPerson`).
    ///   - video: the movie's own duration, frame rate and size — read once
    ///     before the run, never guessed.
    ///   - wallTimeS: how long the whole run took.
    ///   - appVersion / backend / analysisVersion: who wrote the numbers.
    static func make(
        analysis: Analysis,
        frames: [PostProcessInputFrame],
        video: VideoInfo,
        wallTimeS: Double,
        appVersion: String,
        backend: String,
        analysisVersion: String
    ) -> AnalysisDiagnostics {
        let tally = Tally(
            analysis: analysis, frames: frames,
            size: FrameSize(width: video.width, height: video.height))
        let unusable = AnalysisSummary.unusableReason(of: analysis)
        let total = frames.count
        let withPerson = frames.filter(\.detected).count
        var personPct = 0.0
        if total > 0 {
            personPct = round(100.0 * Double(withPerson) / Double(total), decimals: 1)
        }
        // The reason is only *for* something when the take measurably had
        // no hold: a clip with a hold had no dominant reason, and an
        // unmeasurable one is explained by `unusableReason`.
        let explainsNoHold = unusable == nil && analysis.phases.holdCount == 0

        return AnalysisDiagnostics(
            schema: Self.schema,
            appVersion: appVersion,
            backend: backend,
            analysisVersion: analysisVersion,
            analysisWallTimeS: round(wallTimeS, decimals: 2),
            video: Video(
                durationS: round(video.duration, decimals: 3),
                fps: round(video.frameRate, decimals: 1),
                width: video.width,
                height: video.height
            ),
            analysedFps: round(Self.analysedFps(frames), decimals: 1),
            framesTotal: total,
            framesWithPerson: withPerson,
            framesWithPersonPct: personPct,
            unknownReasons: tally.unknownReasons,
            outOfFrame: OutOfFrame(
                partly: tally.partlyOutOfFrame,
                notInFrame: tally.notInFrame,
                edges: tally.outOfFrameEdges
            ),
            dominantReason: explainsNoHold ? tally.noHoldReason(frameCount: total).rawValue : "",
            holdCount: analysis.phases.holdCount,
            holdDurationsS: analysis.phases.holdDurationsS().map { round($0, decimals: 2) },
            usable: unusable == nil,
            unusableReason: unusable ?? ""
        )
    }

    /// The rate the pipeline was actually fed at: the frames' own span, not
    /// the movie's — the extraction keeps at most 30 fps of a 60 fps
    /// recording, and this is the number that says so. `0` when there are
    /// fewer than two frames to measure a span with.
    static func analysedFps(_ frames: [PostProcessInputFrame]) -> Double {
        guard frames.count > 1, let first = frames.first, let last = frames.last,
            last.tMs > first.tMs
        else { return 0 }
        return Double(frames.count - 1) * 1000.0 / Double(last.tMs - first.tMs)
    }

    /// The "No hold found" message this document stands for (what the
    /// session screen shows for a take analysed earlier), or `nil` when
    /// there is nothing to explain — a hold was found, or the clip could
    /// not be measured and says that instead.
    var noHoldExplanation: String? {
        guard usable, holdCount == 0, !dominantReason.isEmpty,
            let reason = NoHoldReason(rawValue: dominantReason)
        else { return nil }
        var edges: [Edge] = []
        if reason == .outOfFrame {
            edges = Edge.allCases.filter { (outOfFrame.edges[$0.rawValue] ?? 0) > 0 }
        }
        return NoHoldExplanation(reason: reason, edges: edges).text
    }

    /// Round to `decimals` places — so a document reads `39.4` and `6.42`,
    /// not `39.39999999999999`. Anything that is not a number reads `0`
    /// rather than poisoning the file with `NaN` (JSON has no way to spell
    /// it, and a diagnostics file must not be the reason a read fails).
    private static func round(_ value: Double, decimals: Int) -> Double {
        guard value.isFinite else { return 0 }
        let factor = pow(10.0, Double(decimals))
        return (value * factor).rounded() / factor
    }

    // MARK: - The file

    /// The diagnostics for `movie`: same folder, same stem,
    /// `.diagnostics.json`.
    static func url(for movie: URL) -> URL {
        movie.deletingPathExtension().appendingPathExtension("diagnostics.json")
    }

    /// Writes `diagnostics` next to `movie`, atomically: a crash mid-write
    /// leaves either the old file or none, never half a document. Keys are
    /// sorted, so two runs over one take differ only where the numbers do.
    static func write(_ diagnostics: AnalysisDiagnostics, for movie: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(diagnostics).write(to: url(for: movie), options: .atomic)
    }

    /// The file beside `movie`, or `nil` — no file, unreadable bytes, or a
    /// `schema` this build does not know (a document written by a different
    /// shape of the app is a different question, not an answer).
    static func read(for movie: URL) -> AnalysisDiagnostics? {
        guard let data = try? Data(contentsOf: url(for: movie)),
            let document = try? JSONDecoder().decode(AnalysisDiagnostics.self, from: data),
            document.schema == Self.schema
        else { return nil }
        return document
    }

    // MARK: - Coding keys

    /// The document's keys, snake_case — the shape the file on the phone
    /// is spelled in, read with `jq` as often as with this app.
    private enum CodingKeys: String, CodingKey {
        case schema
        case appVersion = "app_version"
        case backend
        case analysisVersion = "analysis_version"
        case analysisWallTimeS = "analysis_wall_time_s"
        case video
        case analysedFps = "analysed_fps"
        case framesTotal = "frames_total"
        case framesWithPerson = "frames_with_person"
        case framesWithPersonPct = "frames_with_person_pct"
        case unknownReasons = "unknown_reasons"
        case outOfFrame = "out_of_frame"
        case dominantReason = "dominant_reason"
        case holdCount = "hold_count"
        case holdDurationsS = "hold_durations_s"
        case usable
        case unusableReason = "unusable_reason"
    }
}
