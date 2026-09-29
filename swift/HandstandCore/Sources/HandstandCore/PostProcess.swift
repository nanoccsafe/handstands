import Foundation

/// The tuning knobs of the post-process, one per module constant of
/// `pipeline/handstand/postprocess.py` (each property names its Python twin
/// in a comment). `.init()` is exactly the Python pipeline's configuration,
/// which is what the golden fixtures of chainlink #25 were computed with.
public struct PostProcessConfig: Sendable, Equatable {
    /// `MIN_VISIBILITY` — a joint at or above this score is a position.
    public var minVisibility = 0.5
    /// `BODY_LENGTH_PERCENTILE` — the percentile of each span that counts as
    /// its full length.
    public var bodyLengthPercentile = 90.0
    /// `MIN_BODY_FRAMES` — how many frames a body-length part needs before it
    /// is measured.
    public var minBodyFrames = 10
    /// `MAX_SPEED_L_PER_S` — a sample faster than this has jumped, not travelled.
    public var maxSpeedLPerS = 8.0
    /// `MAX_GAP_S` — the longest gap that is bridged by interpolation.
    public var maxGapS = 0.2
    /// `MIN_CUTOFF` — the One-Euro filter's base cutoff, in Hz.
    public var minCutoff = 1.0
    /// `BETA` — how much the One-Euro cutoff rises with speed.
    public var beta = 0.3
    /// `D_CUTOFF` — the cutoff the One-Euro filter smooths its speed with, in Hz.
    public var dCutoff = 1.0
    /// `MIN_SAMPLE_DT` — the smallest timestep the One-Euro filter sees, seconds.
    public var minSampleDt = 1e-3

    public init() {}
}

/// One frame of the post-process input: the clock, the two frame-level flags
/// and the keypoints the model reported.
///
/// A joint with **no entry** was not seen — the same answer as
/// `[null, null, null]` in a golden fixture and as a NaN column in
/// `handstand.postprocess`. That includes a joint whose coordinates exist but
/// whose `visibility` is missing: no score is no position.
public struct PostProcessInputFrame: Sendable, Equatable {
    /// Presentation timestamp, milliseconds from the start of the clip —
    /// the `t_ms` column, the clip's only clock (these clips are variable
    /// frame rate, so nothing may assume a fixed step).
    public var tMs: Int
    /// Did the model find a pose in this frame (`detected`).
    public var detected: Bool
    /// Did the trainer overlap or touch the athlete (`trainer_contact`).
    public var trainerContact: Bool
    /// The keypoints of this frame, by joint; a missing joint was not seen.
    public var joints: [Joint: Keypoint]

    public init(
        tMs: Int,
        detected: Bool,
        trainerContact: Bool,
        joints: [Joint: Keypoint]
    ) {
        self.tMs = tMs
        self.detected = detected
        self.trainerContact = trainerContact
        self.joints = joints
    }
}

/// One clip's scale in display pixels, and why it has none when `reason` says
/// so — `handstand.postprocess.BodyLength`.
///
/// `L` is the sum of the three parts, each the percentile of its span over the
/// frames where *that part* could be measured, so a foreshortened limb is
/// measured against the frames where it was not. An unusable clip is a fact
/// about the clip, recorded in `reason` rather than thrown away, and Python
/// spells the same fact `reason == ""` for a usable one — here that is `nil`.
public struct BodyLength: Sendable, Equatable {
    /// Why the clip has no body length; `nil` when it has one — Python's
    /// `reason` (empty string when usable), i.e. `usable == reason is nil`.
    public var reason: String?
    /// The torso percentile: shoulder midpoint to hip midpoint, pixels.
    public var torsoPx: Double?
    /// The thigh percentile: the longer hip-to-knee of a frame, pixels.
    public var thighPx: Double?
    /// The shin percentile: the longer knee-to-ankle of a frame, pixels.
    public var shinPx: Double?
    /// How many frames each part was measurable on, keyed `"torso"`,
    /// `"thigh"`, `"shin"` — Python's `BodyLength.frames()`, recorded even
    /// when the clip is unusable.
    public var frames: [String: Int]

    /// Is there a body length to normalise by — Python's `usable` property
    /// (`not self.reason`).
    public var usable: Bool { reason == nil }

    /// `L` in pixels, `nil` when the clip is unusable — Python's `total`
    /// property (NaN there, `null` in a fixture).
    public var totalPx: Double? {
        guard usable, let torsoPx, let thighPx, let shinPx else { return nil }
        return torsoPx + thighPx + shinPx
    }
}

/// A processed position in display pixels: the `(x, y)` pair Python keeps as
/// two parallel arrays.
public struct Point2: Sendable, Equatable {
    /// Horizontal position, display pixels, rightwards.
    public var x: Double
    /// Vertical position, display pixels, downwards.
    public var y: Double

    public init(x: Double, y: Double) {
        self.x = x
        self.y = y
    }
}

/// One frame of the processed trajectory: where every joint with a position
/// ended up, and the two answers `handstand.postprocess` writes per
/// `(frame, joint)` — `valid` (has a position at all) and `filled` (the
/// position is a bridge interpolated across a short gap, not a measurement).
public struct ProcessedFrame: Sendable, Equatable {
    /// The processed positions — **only** the valid joints; a joint Python
    /// writes as NaN has no entry here.
    public var joints: [Joint: Point2]
    /// The joints with a position at all (Python's `valid` column).
    public var valid: Set<Joint>
    /// The joints whose position is an interpolation across a gap (Python's
    /// `filled` column). Always a subset of `valid`.
    public var filled: Set<Joint>
}

/// One clip's processed trajectory and the scale it was measured in — the
/// per-frame part of `handstand.postprocess.ProcessedClip` (gating, outliers,
/// gap fill and smoothing; `hold_like`, phases and features are not ported
/// here, chainlink #40).
public struct ProcessedClip: Sendable, Equatable {
    /// One processed row per input frame, in frame order.
    public var frames: [ProcessedFrame]
    /// The clip's body length, usable or not.
    public var bodyLength: BodyLength

    /// Does this clip have a body length, i.e. any usable scale —
    /// `ProcessedClip.usable` in Python.
    public var usable: Bool { bodyLength.usable }
}

/// The constants of `pipeline/handstand/postprocess.py` (and the one of
/// `handstand.athlete` the body length is checked against), spelled out once
/// so every helper below defaults to the numbers the Python functions default
/// to. `PostProcessTests.testTheConstantsAreThePythonOnes` pins them.
enum PostProcessConstants {
    static let minVisibility = 0.5  // MIN_VISIBILITY
    static let bodyLengthPercentile = 90.0  // BODY_LENGTH_PERCENTILE
    static let minBodyFrames = 10  // MIN_BODY_FRAMES
    static let maxSpeedLPerS = 8.0  // MAX_SPEED_L_PER_S
    static let maxGapS = 0.2  // MAX_GAP_S
    static let minCutoff = 1.0  // MIN_CUTOFF
    static let beta = 0.3  // BETA
    static let dCutoff = 1.0  // D_CUTOFF
    static let minSampleDt = 1e-3  // MIN_SAMPLE_DT
    static let minBodyLengthPixels = 1.0  // handstand.athlete.MIN_BODY_LENGTH_PIXELS

    /// `TORSO_ENDS` — the two joint pairs the torso span runs between.
    static let torsoEnds: [(Joint, Joint)] = [
        (.leftShoulder, .rightShoulder),
        (.leftHip, .rightHip),
    ]

    /// `LEG_SEGMENTS` — `(name, first joint role, second joint role)`, each
    /// measured on both sides.
    static let legSegments: [(name: String, first: String, second: String)] = [
        ("thigh", "hip", "knee"),
        ("shin", "knee", "ankle"),
    ]

    /// `LEG_SIDES` — the sides each leg segment is measured on.
    static let legSides = ["left", "right"]

    /// `LENGTH_PARTS` — every part of the body length, and the order the
    /// sidecar and the usability checks walk them in.
    static let lengthParts = ["torso", "thigh", "shin"]
}

/// One-Euro filter for one scalar — `handstand.postprocess.OneEuroFilter`
/// (Casiez, Roussel & Vogel, CHI 2012).
///
/// A one-pole low-pass whose cutoff rises with the speed of the signal, so a
/// still signal is smoothed hard and a moving one hardly at all: the jitter of
/// a handstand at the top goes, while the boundary between the kick-up and the
/// hold stays where it was. `beta` is that coupling; `dCutoff` smooths the
/// speed estimate itself so a single noisy sample cannot swing the cutoff.
///
/// The filter holds no state across a gap: `reset()` starts it again from the
/// next sample, which is what the post-process wants after an occlusion — its
/// own memory is what would drag the athlete towards the trainer.
struct OneEuroFilter {
    /// `MIN_CUTOFF` — the base cutoff, Hz.
    let minCutoff: Double
    /// `BETA` — cutoff added per (signal unit / second) of speed.
    let beta: Double
    /// `D_CUTOFF` — the cutoff the speed estimate is filtered with, Hz.
    let dCutoff: Double
    /// `MIN_SAMPLE_DT` — the smallest timestep allowed, seconds.
    let minSampleDt: Double

    private var started = false
    private var value = 0.0
    private var speed = 0.0
    private var time = 0.0

    init(
        minCutoff: Double = PostProcessConstants.minCutoff,
        beta: Double = PostProcessConstants.beta,
        dCutoff: Double = PostProcessConstants.dCutoff,
        minSampleDt: Double = PostProcessConstants.minSampleDt
    ) {
        precondition(minCutoff > 0.0, "min_cutoff and d_cutoff must be positive")
        precondition(dCutoff > 0.0, "min_cutoff and d_cutoff must be positive")
        self.minCutoff = minCutoff
        self.beta = beta
        self.dCutoff = dCutoff
        self.minSampleDt = minSampleDt
    }

    /// Forget everything, so the next sample starts the filter —
    /// `OneEuroFilter.reset`.
    mutating func reset() {
        started = false
        value = 0.0
        speed = 0.0
        time = 0.0
    }

    /// The one-pole smoothing factor for a cutoff and a timestep —
    /// `OneEuroFilter._alpha`.
    private func alpha(_ cutoff: Double, dt: Double) -> Double {
        let tau = 1.0 / (2.0 * Double.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
    }

    /// Filter one sample measured at `tSeconds` and return the new value —
    /// `OneEuroFilter.__call__`.
    ///
    /// The first sample after a `reset` is passed through untouched, the way
    /// every One-Euro implementation seeds itself: there is nothing to compare
    /// it against yet, and starting a filter from an average would put a spike
    /// in front of the first real sample.
    mutating func callAsFunction(_ value: Double, at tSeconds: Double) -> Double {
        guard started else {
            started = true
            self.value = value
            speed = 0.0
            time = tSeconds
            return self.value
        }
        let dt = max(tSeconds - time, minSampleDt)
        let instantaneousSpeed = (value - self.value) / dt
        let alphaD = alpha(dCutoff, dt: dt)
        speed = alphaD * instantaneousSpeed + (1.0 - alphaD) * speed
        let a = alpha(minCutoff + beta * abs(speed), dt: dt)
        self.value = a * value + (1.0 - a) * self.value
        time = tSeconds
        return self.value
    }
}

/// Port of the keypoint post-process of `pipeline/handstand/postprocess.py`:
/// gating, body length, speed outliers, gap fill and One-Euro smoothing, in
/// that order — `gated_valid`, `estimate_body_length`, `remove_speed_outliers`,
/// `fill_gaps`, `smooth_track` and `process_clip`.
///
/// The rules that make the numbers match Python:
///
/// * time comes from each frame's `t_ms` — these clips are variable frame
///   rate, so a gap is measured in **seconds** and the filter's `dt` is the
///   real one, never `1 / fps`;
/// * the percentile is numpy's default `"linear"` method: sort the finite
///   per-frame lengths, take index `(n - 1) * p / 100`, lerp between the two
///   neighbours;
/// * the One-Euro filter restarts after every invalid run, exactly where
///   `smooth_track` resets it;
/// * a clip with no body length comes back with every joint invalid and no
///   positions at all, as Python writes it.
///
/// `hold_like_frames`, the `ClipStats` summary, the parquet/JSON file I/O and
/// the CLI are deliberately **not** ported: this is the maths, nothing else.
public enum PostProcess {
    /// Run the five steps over one clip, in order — `process_clip`.
    ///
    /// A clip with no body length — too few valid frames to measure one —
    /// comes back with every joint invalid and `usable == false`: without `L`
    /// there is no scale to normalise by, no speed limit to judge a jump
    /// against and no meaningful unit for the filter, so the honest output is
    /// a clip that says so.
    public static func process(
        _ frames: [PostProcessInputFrame],
        config: PostProcessConfig = .init()
    ) -> ProcessedClip {
        let order = Joint.allCases
        let columns = order.count
        let rows = frames.count
        var x = [[Double]](repeating: [Double](repeating: .nan, count: columns), count: rows)
        var y = [[Double]](repeating: [Double](repeating: .nan, count: columns), count: rows)
        var visibility = [[Double]](repeating: [Double](repeating: .nan, count: columns), count: rows)
        var frameValid = [Bool](repeating: false, count: rows)
        let tSeconds = frames.map { Double($0.tMs) / 1000.0 }

        for (index, frame) in frames.enumerated() {
            // Python's `clip.detected & ~clip.trainer_contact`.
            frameValid[index] = frame.detected && !frame.trainerContact
            for (column, joint) in order.enumerated() {
                guard let keypoint = frame.joints[joint] else { continue }
                x[index][column] = keypoint.x
                y[index][column] = keypoint.y
                visibility[index][column] = keypoint.visibility
            }
        }

        let gated = gatedValid(
            x: x, y: y, visibility: visibility, frameValid: frameValid,
            minVisibility: config.minVisibility
        )
        let body = estimateBodyLength(
            joints: order, x: x, y: y, valid: gated,
            percentile: config.bodyLengthPercentile, minFrames: config.minBodyFrames
        )
        guard body.usable, let length = body.totalPx else {
            return ProcessedClip(
                frames: (0..<rows).map { _ in
                    ProcessedFrame(joints: [:], valid: [], filled: [])
                },
                bodyLength: body
            )
        }

        let afterOutliers = removeSpeedOutliers(
            tSeconds: tSeconds, x: x, y: y, valid: gated,
            bodyLength: length, maxSpeed: config.maxSpeedLPerS
        )
        let bridged = fillGaps(
            tSeconds: tSeconds, x: x, y: y, valid: afterOutliers, maxGapS: config.maxGapS
        )
        let smoothed = smoothTrack(
            tSeconds: tSeconds, x: bridged.x, y: bridged.y, valid: bridged.valid,
            bodyLength: length,
            minCutoff: config.minCutoff, beta: config.beta, dCutoff: config.dCutoff,
            minSampleDt: config.minSampleDt
        )

        var processed: [ProcessedFrame] = []
        processed.reserveCapacity(rows)
        for index in 0..<rows {
            var joints: [Joint: Point2] = [:]
            var valid: Set<Joint> = []
            var filled: Set<Joint> = []
            for (column, joint) in order.enumerated() where bridged.valid[index][column] {
                valid.insert(joint)
                joints[joint] = Point2(
                    x: smoothed.x[index][column],
                    y: smoothed.y[index][column]
                )
                if bridged.filled[index][column] {
                    filled.insert(joint)
                }
            }
            processed.append(ProcessedFrame(joints: joints, valid: valid, filled: filled))
        }
        return ProcessedClip(frames: processed, bodyLength: body)
    }

    // MARK: - Step 1: gating

    /// Which `(frame, joint)` samples the rest of the pipeline may use —
    /// `gated_valid`.
    ///
    /// A sample survives when all of these hold: its frame carries a pose that
    /// is not in trainer contact (`frameValid`, i.e. `detected and not
    /// trainer_contact`), its coordinates are finite, and its visibility is at
    /// or above `minVisibility`. A `NaN` visibility fails the comparison, which
    /// is the same answer the Python gives: a joint the model refused to score
    /// is not a position.
    ///
    /// Python raises `ValueError` on a shape mismatch; the precondition here
    /// is the same refusal with the same message text.
    static func gatedValid(
        x: [[Double]],
        y: [[Double]],
        visibility: [[Double]],
        frameValid: [Bool],
        minVisibility: Double = PostProcessConstants.minVisibility
    ) -> [[Bool]] {
        precondition(
            sameShape(x, y) && sameShape(x, visibility),
            "x, y and visibility must have the same (frames, joints) shape"
        )
        precondition(
            frameValid.count == x.count,
            "frame_valid must have \(x.count) entries, got \(frameValid.count)"
        )
        return x.indices.map { index in
            x[index].indices.map { column in
                frameValid[index]
                    && x[index][column].isFinite
                    && y[index][column].isFinite
                    && visibility[index][column] >= minVisibility
            }
        }
    }

    // MARK: - Step 2: body length

    /// Measure the clip's body length `L` in pixels, or say why it has none —
    /// `estimate_body_length`.
    ///
    /// Each part of `LENGTH_PARTS` is the `percentile`-th of its per-frame
    /// length over the frames where it could be measured: the torso midpoint
    /// to midpoint, and for the thigh and shin the **longer** leg of the
    /// frame, so a split, a stag or a straddle — which foreshortens one leg
    /// and never lengthens it — cannot shrink the yardstick. A part measured
    /// on fewer than `minFrames` frames has no percentile worth the name, and
    /// neither has a clip whose three parts do not add up to
    /// `handstand.athlete.MIN_BODY_LENGTH_PIXELS`; either way the result is
    /// unusable and says so in `reason`.
    static func estimateBodyLength(
        joints: [Joint],
        x: [[Double]],
        y: [[Double]],
        valid: [[Bool]],
        percentile: Double = PostProcessConstants.bodyLengthPercentile,
        minFrames: Int = PostProcessConstants.minBodyFrames
    ) -> BodyLength {
        precondition(sameShape(x, y) && sameShape(x, valid), "x, y and valid must have the same (frames, joints) shape")

        var spans: [String: [Double]] = [
            "torso": torsoSpan(joints: joints, x: x, y: y, valid: valid),
        ]
        for (name, first, second) in PostProcessConstants.legSegments {
            spans[name] = legSpan(
                joints: joints, x: x, y: y, valid: valid, first: first, second: second
            )
        }

        // An unusable clip still records how many frames each part *was*
        // measurable on: a clip that is unusable because the trainer is in
        // every frame of it is worth telling apart from one whose model saw
        // nothing.
        var counts: [String: Int] = [:]
        var values: [String: Double] = [:]
        for name in PostProcessConstants.lengthParts {
            let (value, count) = percentileOf(spans[name] ?? [], p: percentile)
            counts[name] = count
            values[name] = value
        }
        func unusable(_ reason: String) -> BodyLength {
            BodyLength(
                reason: reason, torsoPx: nil, thighPx: nil, shinPx: nil, frames: counts
            )
        }
        for name in PostProcessConstants.lengthParts {
            let count = counts[name] ?? 0
            if count < minFrames {
                return unusable("\(name) was measurable on \(count) frame(s), need \(minFrames)")
            }
            let value = values[name] ?? .nan
            if !value.isFinite || value <= 0.0 {
                return unusable("\(name) has no measurable length")
            }
        }
        let total = (values["torso"] ?? .nan) + (values["thigh"] ?? .nan) + (values["shin"] ?? .nan)
        if total < PostProcessConstants.minBodyLengthPixels {
            return unusable(
                "body length \(String(format: "%.3f", total)) px is below "
                    + "\(PostProcessConstants.minBodyLengthPixels) px"
            )
        }
        return BodyLength(
            reason: nil,
            torsoPx: values["torso"],
            thighPx: values["thigh"],
            shinPx: values["shin"],
            frames: counts
        )
    }

    /// `(value, how many finite samples)` of the percentile of `values` —
    /// `_percentile`.
    ///
    /// The interpolation is numpy's default `"linear"` method: among the
    /// sorted finite samples, index `(n - 1) * p / 100`, lerped towards its
    /// upper neighbour.
    static func percentileOf(_ values: [Double], p: Double) -> (value: Double, count: Int) {
        let finite = values.filter { $0.isFinite }
        guard !finite.isEmpty else { return (.nan, 0) }
        let sorted = finite.sorted()
        let rank = Double(sorted.count - 1) * (Swift.min(Swift.max(p, 0.0), 100.0)) / 100.0
        let low = Int(rank.rounded(.down))
        let high = Int(rank.rounded(.up))
        let gamma = rank - Double(low)
        return (sorted[low] + (sorted[high] - sorted[low]) * gamma, finite.count)
    }

    /// The per-frame shoulder-midpoint to hip-midpoint span, NaN where
    /// unmeasured — `_torso_span`.
    ///
    /// The ends are `TORSO_ENDS`, two *pairs* of joints, so both shoulders
    /// and both hips have to be valid in a frame for that frame to count: a
    /// frame with one shoulder missing has no shoulder midpoint, and using
    /// the one shoulder on its own would make the measurement depend on which
    /// side happened to be visible.
    static func torsoSpan(
        joints: [Joint],
        x: [[Double]],
        y: [[Double]],
        valid: [[Bool]]
    ) -> [Double] {
        let shoulders = midpoint(
            joints: joints, x: x, y: y, valid: valid, PostProcessConstants.torsoEnds[0].0,
            PostProcessConstants.torsoEnds[0].1
        )
        let hips = midpoint(
            joints: joints, x: x, y: y, valid: valid, PostProcessConstants.torsoEnds[1].0,
            PostProcessConstants.torsoEnds[1].1
        )
        return x.indices.map { index in
            guard shoulders.both[index], hips.both[index] else { return .nan }
            return hypot(shoulders.x[index] - hips.x[index], shoulders.y[index] - hips.y[index])
        }
    }

    /// The longer leg of a frame, as the per-frame `first`-to-`second` span —
    /// `_leg_span`.
    ///
    /// Per side, then the max over the sides: whichever leg is seen side-on
    /// that frame sets the length, and a frame where only one leg is visible
    /// contributes that leg rather than nothing.
    static func legSpan(
        joints: [Joint],
        x: [[Double]],
        y: [[Double]],
        valid: [[Bool]],
        first: String,
        second: String
    ) -> [Double] {
        let sides = PostProcessConstants.legSides.map { side -> [Double] in
            guard let a = Joint(rawValue: "\(side)_\(first)"),
                let b = Joint(rawValue: "\(side)_\(second)")
            else {
                return [Double](repeating: .nan, count: x.count)
            }
            return pairSpan(joints: joints, x: x, y: y, valid: valid, a, b)
        }
        return x.indices.map { index in
            var longest = -Double.infinity
            var known = false
            for side in sides where side[index].isFinite {
                known = true
                longest = Swift.max(longest, side[index])
            }
            return known ? longest : .nan
        }
    }

    /// Per-frame distance between two joints, NaN where either is invalid —
    /// `_pair_span`. A joint with no column in `joints` measures nothing at
    /// all, which is how a source without a `foot_index` stays honest.
    static func pairSpan(
        joints: [Joint],
        x: [[Double]],
        y: [[Double]],
        valid: [[Bool]],
        _ first: Joint,
        _ second: Joint
    ) -> [Double] {
        guard let a = joints.firstIndex(of: first), let b = joints.firstIndex(of: second) else {
            return [Double](repeating: .nan, count: x.count)
        }
        return x.indices.map { index in
            guard valid[index][a], valid[index][b] else { return .nan }
            return hypot(x[index][a] - x[index][b], y[index][a] - y[index][b])
        }
    }

    /// The midpoint of two joints of each frame, and whether both are valid —
    /// `_midpoint`.
    static func midpoint(
        joints: [Joint],
        x: [[Double]],
        y: [[Double]],
        valid: [[Bool]],
        _ first: Joint,
        _ second: Joint
    ) -> (x: [Double], y: [Double], both: [Bool]) {
        guard let a = joints.firstIndex(of: first), let b = joints.firstIndex(of: second) else {
            let nan = [Double](repeating: .nan, count: x.count)
            return (nan, nan, [Bool](repeating: false, count: x.count))
        }
        var midX: [Double] = []
        var midY: [Double] = []
        var both: [Bool] = []
        midX.reserveCapacity(x.count)
        midY.reserveCapacity(x.count)
        both.reserveCapacity(x.count)
        for index in x.indices {
            midX.append((x[index][a] + x[index][b]) / 2.0)
            midY.append((y[index][a] + y[index][b]) / 2.0)
            both.append(valid[index][a] && valid[index][b])
        }
        return (midX, midY, both)
    }

    // MARK: - Step 3: outliers

    /// The validity mask with the teleports taken out — `remove_speed_outliers`.
    ///
    /// A joint that covers more than `maxSpeed` body lengths in a second
    /// between two consecutive valid samples is invalid in the *later* of the
    /// two: the model lost it, it did not travel that fast. The earlier
    /// sample is kept and stays the reference, so a single spike is dropped
    /// and the joint is not dragged invalid along with it, and a whole run of
    /// spikes collapses to its first sample. Samples that were already
    /// invalid are never used as a reference, and a pair with no positive
    /// `dt` between them is not evidence of a jump (the schema says `t_ms`
    /// increases, so this only guards a duplicate).
    static func removeSpeedOutliers(
        tSeconds: [Double],
        x: [[Double]],
        y: [[Double]],
        valid: [[Bool]],
        bodyLength: Double,
        maxSpeed: Double = PostProcessConstants.maxSpeedLPerS
    ) -> [[Bool]] {
        precondition(
            tSeconds.count == x.count, "t_seconds must have \(x.count) entries, got \(tSeconds.count)"
        )
        precondition(sameShape(x, y) && sameShape(x, valid), "x, y and valid must have the same (frames, joints) shape")
        precondition(
            bodyLength.isFinite && bodyLength > 0.0,
            "body_length must be a positive, finite number, got \(bodyLength)"
        )
        var kept = valid
        guard let columns = x.first?.count else { return kept }
        for column in 0..<columns {
            var last = -1
            for index in x.indices where valid[index][column] {
                if last < 0 {
                    last = index
                    continue
                }
                let dt = tSeconds[index] - tSeconds[last]
                if dt > 0.0 {
                    let jump = hypot(
                        x[index][column] - x[last][column],
                        y[index][column] - y[last][column]
                    ) / (bodyLength * dt)
                    if jump > maxSpeed {
                        kept[index][column] = false
                        continue
                    }
                }
                last = index
            }
        }
        return kept
    }

    // MARK: - Step 4: gap fill

    /// Bridge the short gaps in each joint — `fill_gaps`.
    ///
    /// A run of invalid samples is interpolated linearly **in time** between
    /// the valid samples on either side of it when those two are at most
    /// `maxGapS` apart, and left NaN when they are further apart or when
    /// there is nothing to interpolate between — the head and tail of a
    /// joint's track have no far side to lean on. The returned `x`/`y` carry
    /// the bridges and NaN everywhere the sample is not valid, `valid` gains
    /// the bridges, and `filled` marks exactly them.
    ///
    /// Interpolating in `t_ms` rather than in frame index is the whole point:
    /// a 0.1 s gap is three frames in a 30 fps clip and one frame in a 10 fps
    /// one, and the samples on either side of it are in the middle of their
    /// own `dt` either way.
    static func fillGaps(
        tSeconds: [Double],
        x: [[Double]],
        y: [[Double]],
        valid: [[Bool]],
        maxGapS: Double = PostProcessConstants.maxGapS
    ) -> (x: [[Double]], y: [[Double]], valid: [[Bool]], filled: [[Bool]]) {
        precondition(
            tSeconds.count == x.count, "t_seconds must have \(x.count) entries, got \(tSeconds.count)"
        )
        precondition(sameShape(x, y) && sameShape(x, valid), "x, y and valid must have the same (frames, joints) shape")

        let columns = x.first?.count ?? 0
        var outX = x.indices.map { index in
            x[index].indices.map { column in valid[index][column] ? x[index][column] : .nan }
        }
        var outY = y.indices.map { index in
            y[index].indices.map { column in valid[index][column] ? y[index][column] : .nan }
        }
        var outValid = valid
        var filled = [[Bool]](repeating: [Bool](repeating: false, count: columns), count: x.count)

        for column in 0..<columns {
            let known = x.indices.filter { valid[$0][column] }
            for pair in 0..<Swift.max(known.count - 1, 0) {
                let first = known[pair]
                let second = known[pair + 1]
                guard second > first + 1 else { continue }
                let span = tSeconds[second] - tSeconds[first]
                guard span > 0.0, span <= maxGapS else { continue }
                for index in (first + 1)..<second {
                    let weight = (tSeconds[index] - tSeconds[first]) / span
                    outX[index][column] =
                        x[first][column] + weight * (x[second][column] - x[first][column])
                    outY[index][column] =
                        y[first][column] + weight * (y[second][column] - y[first][column])
                    outValid[index][column] = true
                    filled[index][column] = true
                }
            }
        }
        return (outX, outY, outValid, filled)
    }

    // MARK: - Step 5: One-Euro smoothing

    /// One-Euro smooth every joint coordinate of a clip, in pixels —
    /// `smooth_track`.
    ///
    /// The filter runs on positions in **body lengths** and the results are
    /// scaled back, because its cutoffs are frequencies of a signal whose unit
    /// is the body's own: a cutoff of 1 Hz means the same thing for a 350 px
    /// athlete and a 600 px one, on every clip in the dataset.
    ///
    /// The filter is restarted at every invalid sample, so a joint's track is
    /// smoothed within each run of valid samples and never across a gap it
    /// does not know the shape of. Invalid samples come back as NaN.
    static func smoothTrack(
        tSeconds: [Double],
        x: [[Double]],
        y: [[Double]],
        valid: [[Bool]],
        bodyLength: Double,
        minCutoff: Double = PostProcessConstants.minCutoff,
        beta: Double = PostProcessConstants.beta,
        dCutoff: Double = PostProcessConstants.dCutoff,
        minSampleDt: Double = PostProcessConstants.minSampleDt
    ) -> (x: [[Double]], y: [[Double]]) {
        precondition(
            tSeconds.count == x.count, "t_seconds must have \(x.count) entries, got \(tSeconds.count)"
        )
        precondition(sameShape(x, y) && sameShape(x, valid), "x, y and valid must have the same (frames, joints) shape")
        precondition(
            bodyLength.isFinite && bodyLength > 0.0,
            "body_length must be a positive, finite number, got \(bodyLength)"
        )

        let columns = x.first?.count ?? 0
        var outX = [[Double]](repeating: [Double](repeating: .nan, count: columns), count: x.count)
        var outY = [[Double]](repeating: [Double](repeating: .nan, count: columns), count: x.count)

        for column in 0..<columns {
            var filterX = OneEuroFilter(
                minCutoff: minCutoff, beta: beta, dCutoff: dCutoff, minSampleDt: minSampleDt
            )
            var filterY = OneEuroFilter(
                minCutoff: minCutoff, beta: beta, dCutoff: dCutoff, minSampleDt: minSampleDt
            )
            for index in x.indices {
                if !valid[index][column] {
                    // Restart at every gap: the filter's memory is the
                    // athlete's own motion, and across an invalid run that
                    // memory is a guess about what happened while nobody was
                    // watching.
                    filterX.reset()
                    filterY.reset()
                    continue
                }
                outX[index][column] =
                    bodyLength * filterX(x[index][column] / bodyLength, at: tSeconds[index])
                outY[index][column] =
                    bodyLength * filterY(y[index][column] / bodyLength, at: tSeconds[index])
            }
        }
        return (outX, outY)
    }

    // MARK: - Shared shape checks (Python's ValueError messages, kept)

    private static func sameShape(_ a: [[Double]], _ b: [[Double]]) -> Bool {
        guard a.count == b.count else { return false }
        guard let columns = a.first?.count else { return true }
        return a.indices.allSatisfy { a[$0].count == columns && b[$0].count == columns }
    }

    private static func sameShape(_ a: [[Double]], _ b: [[Bool]]) -> Bool {
        guard a.count == b.count else { return false }
        guard let columns = a.first?.count else { return true }
        return a.indices.allSatisfy { a[$0].count == columns && b[$0].count == columns }
    }
}
