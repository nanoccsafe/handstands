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
// front or at an angle they are apart. The phone sits low — on the floor,
// looking up — which spreads the near and the far joint of a pair *along* the
// body, so the separation is measured only **across** the torso, projected on
// the axis perpendicular to the shoulder-midpoint-to-hip-midpoint line. Per
// frame::
//
//     S = midpoint(shoulders), H = midpoint(hips), axis = H - S
//     n = unit vector perpendicular to axis
//     shoulder_across = |(left_shoulder - right_shoulder) . n|
//     hip_across      = |(left_hip     - right_hip)     . n|
//     view_width      = (shoulder_across + hip_across) / 2 / L
//
// A frame has a value only when **all four** joints are valid and have a
// position and the torso axis has a non-zero finite length; one pair alone is
// not the across-the-body measure. A clip with no positive finite `L` has no
// `view_width` at all.
//
// Per hold, the median of the finite values classifies the view::
//
//     median <  SIDE_MAX (0.030)       -> "side"
//     median >= NOT_SIDE_MIN (0.036)   -> "not_side"
//     otherwise, or under MIN_VIEW_FRAMES (15) finite frames -> "unsure"
//
// The gap between the two thresholds is deliberate: the labelled side holds
// measured 0.005–0.026 and the front holds 0.038–0.112, and a median in the
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
    public static let sideMax = 0.030

    /// `NOT_SIDE_MIN` — a hold whose median `view_width` is at or over this is
    /// not a side view. `= pipeline/handstand/view.py`.
    public static let notSideMin = 0.036

    /// `MIN_VIEW_FRAMES` — a hold with fewer finite `view_width` frames than
    /// this is never classified: a median of a handful of frames is not a
    /// camera angle. `= pipeline/handstand/view.py`.
    public static let minViewFrames = 15

    /// The four joints the across-the-body measure is built from (`_SIDE_JOINTS`
    /// in Python). All four are read together: one pair alone is not the
    /// measure, so a frame missing any of them is unmeasurable.
    static let sideJoints: [Joint] = [
        .leftShoulder, .rightShoulder, .leftHip, .rightHip,
    ]

    // MARK: - The measurement, frame by frame

    /// The per-frame `view_width` of a clip: one value per frame, `NaN` where
    /// unmeasured — `view_width`.
    ///
    /// The two pair separations projected **across** the torso axis — the line
    /// from the shoulder midpoint to the hip midpoint — averaged and divided by
    /// the clip's body length `L`. Projecting takes out the component along the
    /// body, which a low camera spreads the near and far joint over, so only
    /// the genuinely side-on separation remains. A frame needs **all four**
    /// joints valid with a position and a torso axis of non-zero finite length,
    /// and a clip needs a positive finite `L`; anything else is `NaN`.
    public static func viewWidths(_ processed: ProcessedClip) -> [Double] {
        let frames = processed.frames.count
        guard let totalPx = processed.bodyLength.totalPx, totalPx.isFinite, totalPx > 0.0 else {
            // `totalPx` is nil for an unusable clip and already finite and
            // positive otherwise (a sum of three positive percentiles).
            return [Double](repeating: .nan, count: frames)
        }
        return processed.frames.map { frame in
            guard let across = acrossBodyWidth(frame) else { return .nan }
            return across / totalPx
        }
    }

    /// One frame's width **across** the torso, in display pixels, or `nil` when
    /// the frame cannot be measured — the numerator of `view_width`.
    ///
    /// The shoulders and hips are read together: every one of the four joints
    /// must be valid and have a position, and the shoulder-midpoint-to-hip-
    /// midpoint axis must have a non-zero finite length. Each pair's separation
    /// is projected on the axis turned 90 degrees, so a pair spread along the
    /// body contributes nothing; the two across widths are averaged.
    static func acrossBodyWidth(_ frame: ProcessedFrame) -> Double? {
        guard
            sideJoints.allSatisfy({ frame.valid.contains($0) && frame.joints[$0] != nil }),
            let leftShoulder = frame.joints[.leftShoulder],
            let rightShoulder = frame.joints[.rightShoulder],
            let leftHip = frame.joints[.leftHip],
            let rightHip = frame.joints[.rightHip]
        else { return nil }

        let shoulderMidX = (leftShoulder.x + rightShoulder.x) / 2.0
        let shoulderMidY = (leftShoulder.y + rightShoulder.y) / 2.0
        let hipMidX = (leftHip.x + rightHip.x) / 2.0
        let hipMidY = (leftHip.y + rightHip.y) / 2.0
        let axisX = hipMidX - shoulderMidX
        let axisY = hipMidY - shoulderMidY
        let axisLength = hypot(axisX, axisY)
        guard axisLength.isFinite, axisLength > 0.0 else { return nil }

        // The unit vector across the torso: the axis turned 90 degrees.
        let acrossX = -axisY / axisLength
        let acrossY = axisX / axisLength
        let shoulderAcross = abs(
            (leftShoulder.x - rightShoulder.x) * acrossX
                + (leftShoulder.y - rightShoulder.y) * acrossY)
        let hipAcross = abs(
            (leftHip.x - rightHip.x) * acrossX + (leftHip.y - rightHip.y) * acrossY)
        return (shoulderAcross + hipAcross) / 2.0
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
