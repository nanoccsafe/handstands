import Foundation

// --------------------------------------------------------------------------- #
// Which way the camera faced, per hold (chainlink #102, port of #74).
//
// The scoring reference is built from side-view holds only — the side-view
// features (hip angle, leg separation, knee) are honest only from the side —
// so a number produced from a front or angled take is misleading. This is the
// on-device port of `pipeline/handstand/view.py`: it measures the camera from
// the pose instead of trusting a label, per hold, and the app uses it to stop
// showing a score for a take that was not filmed from the side.
//
// The rule is the lead's measurement: from the side the left and right
// shoulders (and hips) sit on top of each other in the image, and from the
// front or at an angle they are apart. Per frame the width between the two
// shoulders and between the two hips is measured in display pixels and the
// mean of the ones that are visible is divided by the clip's body length `L`::
//
//     view_width = mean(shoulder width, hip width) / L
//
// A pair contributes only when **both** its joints are valid and have a
// position, so a frame that sees one shoulder and not the other has no
// shoulder width; a frame that sees neither pair, or a clip with no usable
// `L`, has no `view_width` at all.
//
// Per hold, the median of the finite values classifies the view::
//
//     median <  SIDE_MAX (0.035)       -> "side"
//     median >= NOT_SIDE_MIN (0.042)   -> "not_side"
//     otherwise, or under MIN_VIEW_FRAMES (15) finite frames -> "unsure"
//
// The gap between the two thresholds is deliberate: the labelled side holds
// measured 0.013–0.030 and the front holds 0.045–0.113, and a median in the
// band between them is not evidence either way.
//
// The file I/O, the hold_views.csv table, the validation report and the CLI of
// the Python module are **not** ported: this is the measurement the app reads,
// nothing else. `ViewDetectorTests` pins the constants and the cases down.
// --------------------------------------------------------------------------- #

/// Which way the camera faced for a hold — `handstand.view.VIEWS`.
///
/// `rawValue` is the Python string, so a value stored on a `Session` row and
/// the CSV the pipeline writes spell the same word.
public enum CameraView: String, Sendable, Equatable, CaseIterable {
    /// The athlete was seen from the side: the side-view features are honest.
    case side
    /// Seen from the front or at an angle — `"not_side"` in Python.
    case notSide = "not_side"
    /// Not enough evidence either way: under `ViewDetector.minViewFrames`
    /// finite frames, or a median in the band between the two thresholds.
    case unsure
}

/// One hold's detected view — one row of `handstand.view`'s `HOLD_VIEW_COLUMNS`
/// (`clip_id` is not carried: the app has one clip in hand).
public struct HoldView: Sendable, Equatable {
    /// The hold's number — `hold_id`.
    public var holdId: Int
    /// How many finite `view_width` frames the hold had — `view_frames`.
    public var viewFrames: Int
    /// The hold's median `view_width`, `NaN` when no frame had one —
    /// `view_width`.
    public var viewWidth: Double
    /// The classification — `view`.
    public var view: CameraView

    public init(holdId: Int, viewFrames: Int, viewWidth: Double, view: CameraView) {
        self.holdId = holdId
        self.viewFrames = viewFrames
        self.viewWidth = viewWidth
        self.view = view
    }
}

/// Port of the measurement half of `pipeline/handstand/view.py`: the constants,
/// the per-frame `view_width` and the per-hold classification.
public enum ViewDetector {
    // MARK: - Constants, one per Python module constant

    /// `SIDE_MAX` — a hold whose median `view_width` is under this is a side
    /// view. `= pipeline/handstand/view.py`.
    public static let sideMax = 0.035

    /// `NOT_SIDE_MIN` — a hold whose median `view_width` is at or over this is
    /// not a side view. `= pipeline/handstand/view.py`.
    public static let notSideMin = 0.042

    /// `MIN_VIEW_FRAMES` — a hold with fewer finite `view_width` frames than
    /// this is never classified: a median of a handful of frames is not a
    /// camera angle. `= pipeline/handstand/view.py`.
    public static let minViewFrames = 15

    /// `_WIDTH_PAIRS` — the joint pairs whose image separation says which way
    /// the athlete faces the camera, in the order they are averaged.
    static let widthPairs: [(Joint, Joint)] = [
        (.leftShoulder, .rightShoulder),
        (.leftHip, .rightHip),
    ]

    // MARK: - The measurement, frame by frame

    /// The per-frame `view_width` of a clip: one value per frame, `NaN` where
    /// unmeasured — `view_width`.
    ///
    /// The mean of the shoulder width and the hip width that this frame can
    /// see, each the 2D pixel distance between the two joints of the pair and
    /// counted only when **both** joints are valid and have a position,
    /// divided by the clip's body length `L`. A frame that sees neither pair,
    /// and every frame of a clip whose `L` is not a positive finite number, is
    /// `NaN`: a width over no body length is not a ratio.
    public static func viewWidths(_ processed: ProcessedClip) -> [Double] {
        let frames = processed.frames.count
        guard let totalPx = processed.bodyLength.totalPx, totalPx > 0.0 else {
            // `totalPx` is nil for an unusable clip and already finite and
            // positive otherwise (a sum of three positive percentiles).
            return [Double](repeating: .nan, count: frames)
        }
        return processed.frames.map { frame in
            var widths: [Double] = []
            for (left, right) in widthPairs {
                guard frame.valid.contains(left), frame.valid.contains(right),
                    let first = frame.joints[left], let second = frame.joints[right]
                else { continue }
                widths.append(hypot(first.x - second.x, first.y - second.y))
            }
            guard !widths.isEmpty else { return .nan }
            return widths.reduce(0.0, +) / Double(widths.count) / totalPx
        }
    }

    /// A hold's median `view_width` and its view, by the threshold rule —
    /// `classify`.
    ///
    /// The median is of the finite values, `NaN` when there are none. A hold
    /// with fewer than `minViewFrames` finite values is `.unsure` whatever its
    /// median; otherwise the median decides, with the band between `sideMax`
    /// and `notSideMin` left `.unsure` on purpose.
    public static func classify(_ widths: [Double]) -> (median: Double, view: CameraView) {
        let finite = widths.filter { $0.isFinite }
        let median = Self.median(finite)
        guard finite.count >= minViewFrames else { return (median, .unsure) }
        if median < sideMax { return (median, .side) }
        if median >= notSideMin { return (median, .notSide) }
        return (median, .unsure)
    }

    /// One view per hold of a clip, in `hold_id` order — `clip_hold_views`'
    /// per-hold loop.
    ///
    /// The widths are the frames that are in the hold **and** valid, and
    /// `viewFrames` counts the finite ones among them — a hold with no usable
    /// frame is reported and called `.unsure` rather than dropped. A processed
    /// clip and a features table of different lengths are not evidence of
    /// anything, so `[]` comes back rather than a crash.
    public static func holdViews(processed: ProcessedClip, features: ClipFeatures) -> [HoldView] {
        guard processed.frames.count == features.holdId.count else { return [] }
        let widths = viewWidths(processed)
        return features.holdIds().map { id in
            let holdWidths = features.holdId.indices.filter {
                features.holdId[$0] == id && features.valid[$0]
            }.map { widths[$0] }
            let (median, view) = classify(holdWidths)
            return HoldView(
                holdId: id,
                viewFrames: holdWidths.filter { $0.isFinite }.count,
                viewWidth: median,
                view: view
            )
        }
    }

    /// `np.median` of the finite values: the middle value, or the mean of the
    /// two middle ones — `NaN` for an empty list (`Features.median`, kept
    /// local so this module reads on its own).
    static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        guard !sorted.isEmpty else { return .nan }
        if sorted.count % 2 == 0 {
            return (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2.0
        }
        return sorted[sorted.count / 2]
    }
}
