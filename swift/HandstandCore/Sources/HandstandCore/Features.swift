import Foundation

// --------------------------------------------------------------------------- #
// Per-frame features, and one row per hold.
//
// A port of the measurement half of `pipeline/handstand/features.py`
// (chainlink #81): the constants, `BodyTrack` / `bodyFrameTrack`, the geometry
// of `bodyFeatures`, the balance columns, `applyHoldFacing`, `ClipFeatures`
// and the hold summary (`spread`, `holdStability`, `holdRows`) — with the
// centre of mass they are built on in CentreOfMass.swift.
//
// The tables (`feature_table`), the parquet / CSV writing, the `ClipStats`
// summary and its `measure` / `summary` / `ranked`, the TOLERANCES fault list,
// the plots and the CLI are **not** ported: this is the maths, nothing else.
//
// The rules the port lives by, taken straight from Python:
//
// * everything is measured in the body frame of `BodyFrame.toBodyFrame` —
//   origin at the wrist midpoint of *that* frame, `u` to the right, `v` up,
//   both in body lengths `L`, so an offset of `0` is a joint exactly over the
//   hands and a 350 px athlete reads the same numbers as a 600 px one;
// * **sides**: a left/right pair is the mean of both where both are, the one
//   that is where only one is, `NaN` where neither is — except
//   `leg_separation` and `hand_width`, which *are* a measurement between the
//   two sides and are `NaN` with one;
// * **signs**: a feature whose sign only says "which way" is signed in `u`;
//   `banana` is signed towards the nose's side of the body's axis, and
//   `com_forward` by the facing sign the **hold** voted for, not by each
//   frame's own flicker;
// * **no half-measurements**: a feature the frame could not support is `NaN`,
//   never a number made of nothing, and `valid` is what a later stage
//   filters on (a trainer in front of the camera makes a frame invalid, but
//   the numbers are still in the table);
// * the hold summary rounds like Python's `round(x, d)` — half to even on the
//   binary value — with `FeatureSpec.digits` for a feature and the digits of
//   `HOLD_STABILITY` for a balance column, and `NaN` stays `NaN`.
// --------------------------------------------------------------------------- #

/// One feature: what the column is called, the unit it is in, and the value a
/// good line handstand shows — `handstand.features.Feature`.
public struct FeatureSpec: Sendable, Equatable {
    /// The column's name, e.g. `off_shoulder`.
    public var name: String
    /// The unit: `L` (body lengths), `deg` or `ratio`.
    public var unit: String
    /// The value a good line handstand shows, as `Features.all`'s docstring
    /// quotes it — `Feature.target`.
    public var target: Double
    /// How many decimals the hold summary rounds the column to —
    /// `Feature.digits`: five on lengths and ratios, two on degrees.
    public var digits: Int

    public init(name: String, unit: String, target: Double, digits: Int) {
        self.name = name
        self.unit = unit
        self.target = target
        self.digits = digits
    }
}

/// One joint's position in the body frame: `u` rightwards, `v` upwards, both
/// in body lengths — the two components of `BodyTrack.uv`.
///
/// `BodyPoint.nan` is how a joint the source did not report, and a frame the
/// body frame could not be placed on, are written: every measurement below
/// can then be read as "is this finite".
struct BodyPoint: Sendable, Equatable, Hashable {
    /// Horizontal position in body lengths, `u` to the right.
    var u: Double
    /// Vertical position in body lengths, `v` upwards.
    var v: Double

    /// Both components `NaN` — a joint that was not seen.
    static let nan = BodyPoint(u: .nan, v: .nan)

    /// Was this position measured at all.
    var isFinite: Bool { u.isFinite && v.isFinite }

    init(u: Double, v: Double) {
        self.u = u
        self.v = v
    }
}

/// A clip's joints in the body frame, with a name for every column —
/// `handstand.features.BodyTrack`.
///
/// `uv[frame][column]` is one joint of that frame, `NaN` wherever the joint
/// was not visible, and `joints` is the schema's names in column order. A
/// joint a schema does not have simply has no column, and `point(_:)` answers
/// `NaN` for it — which is what lets the same code run over every source.
struct BodyTrack: Sendable, Equatable {
    /// `[frame][joint]` positions in body-frame units.
    var uv: [[BodyPoint]]
    /// The schema's names, in column order.
    var joints: [Joint]

    /// How many frames the clip has — `BodyTrack.frames`.
    var frames: Int { uv.count }

    /// The `(frames)` track of one joint, all `NaN` when this clip's schema
    /// has no such joint — `BodyTrack.point`.
    func point(_ name: Joint) -> [BodyPoint] {
        guard let column = joints.firstIndex(of: name) else {
            return [BodyPoint](repeating: .nan, count: frames)
        }
        return uv.map { row in column < row.count ? row[column] : .nan }
    }
}

/// Every feature of a clip's frames, measured from its joints in the body
/// frame — `handstand.features.BodyFeatures`.
///
/// `values` holds one `(frames,)` array per feature and per CoM column, `NaN`
/// wherever the frame could not support it; `valid` says whether the frame
/// could support them all. The nose is deliberately not part of `valid` — it
/// only gives `banana` its sign and `head` its number, so a frame without one
/// is a frame with two `NaN`s rather than a frame to throw away.
struct BodyFeatures: Sendable, Equatable {
    /// One array per feature / CoM column, keyed by name.
    var values: [String: [Double]]
    /// Which frames every feature could be measured on.
    var valid: [Bool]
    /// Did every segment of the CoM model exist on this frame —
    /// `com_complete`.
    var comComplete: [Bool]
    /// Which side of the base of support each frame's CoM is on —
    /// `balance_zone`, `nil` where there is no CoM.
    var balanceZone: [BalanceZone?]

    /// How many frames the measurements cover — `BodyFeatures.frames`.
    var frames: Int { valid.count }

    /// One feature's values, `NaN` for a name that is not a feature —
    /// `BodyFeatures.value`.
    func value(_ name: String) -> [Double] {
        values[name] ?? [Double](repeating: .nan, count: frames)
    }
}

/// One clip's features, with the frame identity and the phase labels they sit
/// in — `handstand.features.ClipFeatures` minus the clip identity (there is
/// no file to write here, so `clip_id` and `source` are not carried).
///
/// `phase` and `holdId` come straight from `PhaseSegmenter.classify` and are
/// carried here so a later stage can filter on "inside a hold" without asking
/// a second stage, and so the hold summary can be built from one value.
public struct ClipFeatures: Sendable, Equatable {
    /// The clip's frame timestamps, milliseconds — the only clock these clips
    /// have, and variable from frame to frame.
    public var tMs: [Int]
    /// The phase of each frame, from `PhaseSegmenter.classify`.
    public var phase: [Phase]
    /// The number of the hold each frame is inside, `PhaseSegmenter.noHold`
    /// outside one.
    public var holdId: [Int]
    /// One array per feature and CoM column, keyed by name: every
    /// `Features.all` name (14, including `hand_width`) plus `com_u`,
    /// `com_v`, `facing_sign` and `com_forward`. `NaN` wherever the frame
    /// could not support the column.
    ///
    /// `com_complete` and `balance_zone` are not in here — a flag and a
    /// category are not a column of numbers — they are `comComplete` and
    /// `balanceZone`.
    public var values: [String: [Double]]
    /// Which side of the base of support each frame's CoM is on, `nil` where
    /// the frame has no CoM — `balance_zone`.
    public var balanceZone: [BalanceZone?]
    /// Did every segment of the CoM model exist on this frame —
    /// `com_complete`.
    public var comComplete: [Bool]
    /// Which frames every feature could be measured on: a placeable body
    /// frame with a visible midpoint of every part, and no trainer in the
    /// way. The numbers of an invalid frame are still in `values`, as
    /// everywhere else in the pipeline — what is not true is that they are
    /// the athlete's.
    public var valid: [Bool]
    /// Why the clip has no features; `""` when it has some — Python's
    /// `unusable_reason`.
    public var unusableReason: String

    public init(
        tMs: [Int],
        phase: [Phase],
        holdId: [Int],
        values: [String: [Double]],
        balanceZone: [BalanceZone?],
        comComplete: [Bool],
        valid: [Bool],
        unusableReason: String
    ) {
        self.tMs = tMs
        self.phase = phase
        self.holdId = holdId
        self.values = values
        self.balanceZone = balanceZone
        self.comComplete = comComplete
        self.valid = valid
        self.unusableReason = unusableReason
    }

    /// Does this clip have a scale, and so any features at all —
    /// `ClipFeatures.usable`.
    public var usable: Bool { unusableReason.isEmpty }

    /// How many frames the clip has — `ClipFeatures.frames`.
    public var frames: Int { tMs.count }

    /// One feature's values, `NaN` for a name that is not one —
    /// `ClipFeatures.value`.
    public func value(_ name: String) -> [Double] {
        values[name] ?? [Double](repeating: .nan, count: frames)
    }

    /// The clip's hold numbers, in time order — `ClipFeatures.hold_ids`.
    public func holdIds() -> [Int] {
        Array(Set(holdId.filter { $0 >= 0 })).sorted()
    }

    /// The frames of one hold — `ClipFeatures.hold_mask`.
    public func holdMask(_ id: Int) -> [Bool] {
        holdId.map { $0 == id }
    }

    /// The frames of one hold the features could actually be measured on —
    /// `ClipFeatures.measurable_hold_mask`.
    ///
    /// This is the filter every number about a hold is taken over: inside the
    /// hold, and `valid`. A hold with a trainer in front of the camera for a
    /// third of it is summarised over the two thirds a human could see, and
    /// the row says how many frames that was.
    public func measurableHoldMask(_ id: Int) -> [Bool] {
        zip(holdMask(id), valid).map { $0 && $1 }
    }

    /// The clip's longest hold, or `PhaseSegmenter.noHold` when it has none —
    /// `ClipFeatures.longest_hold_id`.
    ///
    /// The longest hold of a clip is the one representative of it: a short one
    /// is a hop on the hands. Ties go to the earlier hold.
    public func longestHoldId() -> Int {
        var best = PhaseSegmenter.noHold
        var bestSpan = -1.0
        for id in holdIds() {
            let frames = holdId.indices.filter { holdId[$0] == id }
            guard let first = frames.first, let last = frames.last else { continue }
            let span = Double(tMs[last] - tMs[first])
            if span > bestSpan {
                best = id
                bestSpan = span
            }
        }
        return best
    }

    /// How long one hold lasted, first frame to last, in seconds —
    /// `ClipFeatures.hold_duration_s`.
    public func holdDurationS(_ id: Int) -> Double {
        let frames = holdId.indices.filter { holdId[$0] == id }
        guard let first = frames.first, let last = frames.last else { return 0.0 }
        return Double(tMs[last] - tMs[first]) / 1000.0
    }

    /// The per-frame table Python's scorer reads — `feature_table`'s rounding
    /// on every column it carries — `ClipFeatures.tableRounded`.
    ///
    /// The scorer in production never sees the raw measurements: it reads the
    /// parquet `features.feature_table` writes, which rounds each feature to
    /// `Feature.digits` (five decimals on lengths and ratios, two on angles),
    /// `com_u`/`com_v`/`com_forward` to five and `facing_sign` to zero —
    /// half to even on the binary value, NaN staying NaN. Scoring the rounded
    /// table is therefore what makes the on-device score equal the pipeline's,
    /// and `Scorer` applies this first, exactly as Python applies it by
    /// reading the file. Columns this copy does not carry are left alone;
    /// they are read as `NaN` either way, as Python reads a column the
    /// table does not have.
    public func tableRounded() -> ClipFeatures {
        var rounded = values
        for spec in Features.all {
            if let column = rounded[spec.name] {
                rounded[spec.name] = column.map {
                    Features.roundedScalar($0, digits: spec.digits)
                }
            }
        }
        // The centre-of-mass columns round like lengths, except the sign,
        // which is a whole number: -1, 0 or 1.
        for name in ["com_u", "com_v", "com_forward"] {
            if let column = rounded[name] {
                rounded[name] = column.map {
                    Features.roundedScalar($0, digits: Features.lengthDigits)
                }
            }
        }
        if let column = rounded["facing_sign"] {
            rounded["facing_sign"] = column.map { Features.roundedScalar($0, digits: 0) }
        }
        var table = self
        table.values = rounded
        return table
    }
}

/// One row of the per-hold table — the numeric half of a `hold_rows` dict in
/// `pipeline/handstand/features.py` (`clip_id` and `source` are not carried:
/// there is no file to write and the test compares everything else).
public struct HoldSummaryRow: Sendable, Equatable {
    /// The hold's number — `hold_id`.
    public var holdId: Int
    /// How many frames the hold covers, including the ones bridged into it —
    /// `hold_frames`.
    public var holdFrames: Int
    /// How many of those frames the summary is measured over — `valid_frames`.
    public var validFrames: Int
    /// The `t_ms` of the hold's first frame — `hold_start_ms`.
    public var holdStartMs: Int
    /// The `t_ms` of the hold's last frame — `hold_end_ms`.
    public var holdEndMs: Int
    /// How long the hold lasted, first frame to last, in seconds, rounded to
    /// three decimals — `hold_duration_s`.
    public var holdDurationS: Double
    /// Every other numeric column of `HOLD_SUMMARY_COLUMNS`: `{feature}_median`
    /// and `{feature}_iqr` for all 14 features, then the eight balance
    /// columns — rounded the way Python writes them, `NaN` where Python has
    /// `NaN`.
    public var stats: [String: Double]

    public init(
        holdId: Int,
        holdFrames: Int,
        validFrames: Int,
        holdStartMs: Int,
        holdEndMs: Int,
        holdDurationS: Double,
        stats: [String: Double]
    ) {
        self.holdId = holdId
        self.holdFrames = holdFrames
        self.validFrames = validFrames
        self.holdStartMs = holdStartMs
        self.holdEndMs = holdEndMs
        self.holdDurationS = holdDurationS
        self.stats = stats
    }
}

/// Port of the measurement half of `pipeline/handstand/features.py`: the
/// constants, the geometry (`bodyFrameTrack`, `bodyFeatures`) and the hold
/// summary (`holdRows`).
///
/// `extract(tMs:processed:phases:trainerContact:)` is the entry point — it
/// mirrors `extract_clip` and is what Python calls like this:
///
/// ```python
/// extract_clip(case, source, t_ms, processed.x, processed.y, processed.valid,
///              joints, processed.body_length.total, phase=labels.phase_codes,
///              hold_id=labels.hold_id, trainer_contact=clip.trainer_contact)
/// ```
///
/// Swift does the same with the #39/#40 outputs and the input frames'
/// `trainerContact`.
public enum Features {
    // MARK: - Constants, one per Python module constant

    /// `MIN_SEGMENT_L` — a span shorter than this has no direction and no
    /// magnitude to measure against: a degenerate frame, not a 0° angle.
    public static let minSegmentL = 1e-9
    /// `LINE_TOLERANCE_L` — how far the worst stacking error may be, in body
    /// lengths, before the line is bent.
    public static let lineToleranceL = 0.1
    /// `BODY_ANGLE_TOL_DEG` — how far the wrist→ankle line may lean from
    /// vertical before the body is not straight any more.
    public static let bodyAngleTolDeg = 10.0
    /// `PIKE_HIP_DEG` — the hip angle under which a hold is a pike.
    public static let pikeHipDeg = 165.0
    /// `OPEN_SHOULDER_DEG` — the shoulder angle under which the shoulders are
    /// closed and the chest arched open.
    public static let openShoulderDeg = 160.0
    /// `STRAIGHT_KNEE_DEG` — the knee angle under which the legs are bent
    /// rather than locked.
    public static let straightKneeDeg = 165.0
    /// `BENT_ELBOW_DEG` — the elbow angle under which the arms are bent.
    public static let bentElbowDeg = 160.0
    /// `BANANA_TOLERANCE_L` — how far the hip midpoint may sit from the
    /// shoulder→ankle line before the shape is an arched back.
    public static let bananaToleranceL = 0.1
    /// `HEAD_FLEXION_DEG` — how far the head may leave the torso's line
    /// before it is dropped rather than merely looking.
    public static let headFlexionDeg = 60.0
    /// `SPLIT_DEG` — the leg separation over which the legs are apart in the
    /// image rather than together.
    public static let splitDeg = 30.0
    /// `_LENGTH_DIGITS` — the rounding of a length or a ratio in the hold
    /// summary.
    public static let lengthDigits = 5
    /// `_ANGLE_DIGITS` — the rounding of an angle or a percentage there.
    public static let angleDigits = 2

    /// `SIDES` — the two sides, in the order the schema names them.
    static let sides = ["left", "right"]
    /// `NOSE` — the only joint that says which way the athlete is facing, and
    /// so the only one that gives `banana` and `head` a sign.
    static let noseName = "nose"
    /// `BODY_PARTS` — the six body parts every feature is read off, in the
    /// order the body is read from the hands up.
    static let bodyParts = ["wrist", "elbow", "shoulder", "hip", "knee", "ankle"]
    /// `WANTED_JOINTS` — every joint the features are read off, as the schema
    /// names it. A source that does not report one leaves it `NaN` and the
    /// features that need it `NaN` with it.
    static let wantedJoints: [Joint] =
        Features.bodyParts.flatMap { part in
            Features.sides.compactMap { Joint(rawValue: "\($0)_\(part)") }
        } + [.nose]

    // MARK: - The features

    /// `FEATURES` — every feature, in the order the table and the hold summary
    /// carry them: where the stack is, how straight the line is, the joints
    /// from the shoulders down, then the shape.
    public static let all: [FeatureSpec] = [
        FeatureSpec(name: "off_shoulder", unit: "L", target: 0.0, digits: Features.lengthDigits),
        FeatureSpec(name: "off_hip", unit: "L", target: 0.0, digits: Features.lengthDigits),
        FeatureSpec(name: "off_knee", unit: "L", target: 0.0, digits: Features.lengthDigits),
        FeatureSpec(name: "off_ankle", unit: "L", target: 0.0, digits: Features.lengthDigits),
        FeatureSpec(
            name: "line_deviation", unit: "L", target: 0.0, digits: Features.lengthDigits),
        FeatureSpec(name: "body_angle", unit: "deg", target: 0.0, digits: Features.angleDigits),
        FeatureSpec(
            name: "shoulder_angle", unit: "deg", target: 180.0, digits: Features.angleDigits),
        FeatureSpec(name: "hip_angle", unit: "deg", target: 180.0, digits: Features.angleDigits),
        FeatureSpec(name: "knee_angle", unit: "deg", target: 180.0, digits: Features.angleDigits),
        FeatureSpec(name: "elbow_angle", unit: "deg", target: 180.0, digits: Features.angleDigits),
        FeatureSpec(name: "banana", unit: "L", target: 0.0, digits: Features.lengthDigits),
        FeatureSpec(name: "head", unit: "deg", target: 0.0, digits: Features.angleDigits),
        FeatureSpec(
            name: "leg_separation", unit: "deg", target: 0.0, digits: Features.angleDigits),
        // No target in a side view: 1.0 is the front-view answer (hands about
        // shoulder-width apart), what a frontal recording would read.
        FeatureSpec(name: "hand_width", unit: "ratio", target: 1.0, digits: Features.lengthDigits),
    ]

    /// `FEATURE_NAMES` — the feature names alone, in `Features.all` order.
    public static let featureNames: [String] = Features.all.map(\.name)

    /// The centre-of-mass columns that live in `ClipFeatures.values`, in
    /// `COM_COLUMNS` order. `com_complete` and `balance_zone` are the other
    /// two of Python's six, and they are `ClipFeatures.comComplete` and
    /// `ClipFeatures.balanceZone` instead — a flag and a category are not a
    /// column of numbers.
    public static let comColumns: [String] = [
        "com_u", "com_v", "facing_sign", "com_forward",
    ]

    /// One balance column of the hold table, with the number of decimals
    /// Python writes it with — one row of `HOLD_STABILITY`.
    public struct BalanceStat: Sendable, Equatable {
        /// The column's name, e.g. `com_sway_sd`.
        public var name: String
        /// The rounding: five decimals for lengths and speeds, two for angles
        /// and percentages.
        public var digits: Int

        public init(name: String, digits: Int) {
            self.name = name
            self.digits = digits
        }
    }

    /// `HOLD_STABILITY` — the per-hold balance columns, each with the
    /// rounding it is written with, computed by `holdStability(_:holdId:)`
    /// over the hold's measurable frames. (The name is `balanceStats` rather
    /// than `holdStability` because that is the function that computes them.)
    public static let balanceStats: [BalanceStat] = [
        BalanceStat(name: "com_forward_median", digits: Features.lengthDigits),
        BalanceStat(name: "com_sway_sd", digits: Features.lengthDigits),
        BalanceStat(name: "com_sway_range", digits: Features.lengthDigits),
        BalanceStat(name: "com_speed_rms", digits: Features.lengthDigits),
        BalanceStat(name: "hip_angle_sd", digits: Features.angleDigits),
        BalanceStat(name: "shoulder_angle_sd", digits: Features.angleDigits),
        BalanceStat(name: "pct_over", digits: Features.angleDigits),
        BalanceStat(name: "pct_under", digits: Features.angleDigits),
    ]

    /// `HOLD_STABILITY_COLUMNS` — those names alone, in write order.
    public static let holdStabilityNames: [String] = Features.balanceStats.map(\.name)

    /// `HOLD_SUMMARY_COLUMNS` — the per-hold table's columns, in write order,
    /// minus the two identity columns Swift does not carry (`clip_id` and
    /// `source`): the hold's own fields, then a median and an IQR for every
    /// feature, then the balance columns.
    public static let holdSummaryColumns: [String] = {
        let fields = [
            "hold_id", "hold_frames", "valid_frames", "hold_start_ms", "hold_end_ms",
            "hold_duration_s",
        ]
        let spreads = Features.featureNames.flatMap { ["\($0)_median", "\($0)_iqr"] }
        return fields + spreads + Features.holdStabilityNames
    }()

    // MARK: - One clip

    /// Measure one clip's features, from its processed trajectory to its
    /// per-frame answer — `extract_clip`.
    ///
    /// The input is exactly what Python passes: the frame clock, the
    /// post-processed trajectory of `PostProcess.process`, the phase labels of
    /// `PhaseSegmenter.classify`, and the **raw** `trainerContact` flags — the
    /// numbers of a flagged frame are measured as everywhere else, but the
    /// frame is `valid == false`, so no measurement takes it for the athlete's.
    ///
    /// A clip with no body length comes back with every feature `NaN`, every
    /// frame invalid and `unusableReason` set: without `L` there is no unit to
    /// measure an offset or an angle in, and a feature in pixels is not a
    /// feature.
    ///
    /// The `phase` and `holdId` lengths are checked the way Python checks
    /// them; a malformed *track* is a programming error rather than data, and
    /// stops the program as Python's `ValueError` would.
    public static func extract(
        tMs: [Int],
        processed: ProcessedClip,
        phases: ClipPhases,
        trainerContact: [Bool]
    ) -> ClipFeatures {
        let frames = tMs.count
        precondition(
            phases.phase.count == frames,
            "phase has \(phases.phase.count) entries, t_ms has \(frames)")
        precondition(
            phases.holdId.count == frames,
            "hold_id has \(phases.holdId.count) entries, t_ms has \(frames)")
        let bodyLength = processed.bodyLength.totalPx ?? .nan
        var reason = ""
        if !bodyLength.isFinite || bodyLength <= 0.0 {
            // Python's `f"no body length ({body_length!r} px)"`: Swift's
            // description of a `Double` is Python's `repr` for every value
            // this path can produce (`nan`, `inf`, `0.0`, …).
            reason = "no body length (\(bodyLength) px)"
        }

        let measured: BodyFeatures
        if !reason.isEmpty {
            measured = unusableFeatures(frames: frames)
        } else {
            precondition(
                trainerContact.count == frames,
                "trainer_contact must have one entry per frame (\(frames)), "
                    + "got \(trainerContact.count)")
            // A track built from a `ProcessedClip` always has one point per
            // joint name per frame, so the shape Python refuses here cannot
            // happen; `try!` is that guarantee, not a swallowed error.
            var features = try! bodyFeatures(
                bodyFrameTrack(processed, bodyLength: bodyLength))
            for index in trainerContact.indices where trainerContact[index] {
                features.valid[index] = false
            }
            measured = features
        }
        // The facing sign is decided per hold, so it is applied here rather
        // than in `bodyFeatures`: this is where the hold labels exist.
        let facing = applyHoldFacing(measured, holdId: phases.holdId)
        return ClipFeatures(
            tMs: tMs,
            phase: phases.phase,
            holdId: phases.holdId,
            values: facing.values,
            balanceZone: facing.balanceZone,
            comComplete: facing.comComplete,
            valid: facing.valid,
            unusableReason: reason
        )
    }

    /// A processed clip as the pixel arrays `extract_clip` is handed — the
    /// `x`, `y` and `valid` Python passes in, one column per joint of
    /// `Joint.allCases`.
    static func bodyFrameTrack(_ clip: ProcessedClip, bodyLength: Double) -> BodyTrack {
        let order = Joint.allCases
        let rows = clip.frames.count
        var x = [[Double]](repeating: [Double](repeating: .nan, count: order.count), count: rows)
        var y = [[Double]](repeating: [Double](repeating: .nan, count: order.count), count: rows)
        var valid = [[Bool]](repeating: [Bool](repeating: false, count: order.count), count: rows)
        for (index, frame) in clip.frames.enumerated() {
            for (column, joint) in order.enumerated() {
                x[index][column] = frame.joints[joint]?.x ?? .nan
                y[index][column] = frame.joints[joint]?.y ?? .nan
                valid[index][column] = frame.valid.contains(joint)
            }
        }
        return bodyFrameTrack(
            x: x, y: y, valid: valid, joints: order, bodyLength: bodyLength)
    }

    /// A processed clip as a `BodyTrack`: the joints in the body frame —
    /// `body_frame_track`.
    ///
    /// This is `BodyFrame.toBodyFrame` over a whole clip, one frame at a
    /// time, with the origin the wrist midpoint of *each* frame (one visible
    /// wrist is enough, none is a frame whose body frame was never placed:
    /// every joint comes back `NaN` rather than a position measured against
    /// the image's own origin).
    static func bodyFrameTrack(
        x: [[Double]],
        y: [[Double]],
        valid: [[Bool]],
        joints: [Joint],
        bodyLength: Double
    ) -> BodyTrack {
        precondition(
            sameShape(x, y) && sameShape(x, valid),
            "x, y and valid must have the same (frames, joints) shape")
        if let columns = x.first {
            precondition(
                columns.count == joints.count,
                "x has \(columns.count) joint columns but \(joints.count) joint names")
        }
        precondition(
            bodyLength.isFinite && bodyLength > 0.0,
            "body_length must be a positive, finite number, got \(bodyLength)")

        let wristColumns = joints.indices.filter {
            joints[$0] == .leftWrist || joints[$0] == .rightWrist
        }
        let origin = pixelMidpoint(x: x, y: y, valid: valid, columns: wristColumns)

        var uv: [[BodyPoint]] = []
        uv.reserveCapacity(x.count)
        for index in x.indices {
            var row: [BodyPoint] = []
            row.reserveCapacity(joints.count)
            for column in joints.indices {
                guard valid[index][column] else {
                    row.append(.nan)
                    continue
                }
                row.append(
                    BodyPoint(
                        u: (x[index][column] - origin.x[index]) / bodyLength,
                        v: (origin.y[index] - y[index][column]) / bodyLength
                    ))
            }
            uv.append(row)
        }
        return BodyTrack(uv: uv, joints: joints)
    }

    /// A joint group's per-frame midpoint over its visible members, in
    /// pixels — `_pixel_midpoint`.
    ///
    /// One wrist is enough to place the origin, so a side-on clip with one
    /// wrist in shot is still measured; a frame where no member of the group
    /// is visible is `NaN` rather than averaged over zeros, which for the
    /// wrist group would be a body frame placed at `(0, 0)` of the image.
    static func pixelMidpoint(
        x: [[Double]], y: [[Double]], valid: [[Bool]], columns: [Int]
    ) -> (x: [Double], y: [Double]) {
        guard !columns.isEmpty else {
            return (
                [Double](repeating: .nan, count: x.count),
                [Double](repeating: .nan, count: x.count)
            )
        }
        var midX = [Double](repeating: .nan, count: x.count)
        var midY = [Double](repeating: .nan, count: x.count)
        for index in x.indices {
            var sumX = 0.0
            var sumY = 0.0
            var count = 0
            for column in columns where valid[index][column] {
                sumX += x[index][column]
                sumY += y[index][column]
                count += 1
            }
            if count > 0 {
                midX[index] = sumX / Double(count)
                midY[index] = sumY / Double(count)
            }
        }
        return (midX, midY)
    }

    /// Every feature of `FEATURES` for a whole clip, from its body-frame
    /// joints — `body_features`.
    ///
    /// This is the whole geometry of the module and it knows nothing about
    /// files, phases or holds, which is why the tests can drive it from a
    /// pose written by hand. It throws only the shape refusal Python raises
    /// (see `CentreOfMass.centreOfMass`).
    static func bodyFeatures(_ track: BodyTrack) throws -> BodyFeatures {
        let frames = track.frames
        var points: [String: [BodyPoint]] = [:]
        for joint in Features.wantedJoints {
            points[joint.rawValue] = track.point(joint)
        }
        func point(_ name: String) -> [BodyPoint] {
            points[name] ?? [BodyPoint](repeating: .nan, count: frames)
        }

        let wristMid = groupMidpoint(points, part: "wrist", frames: frames)
        let elbowMid = groupMidpoint(points, part: "elbow", frames: frames)
        let shoulderMid = groupMidpoint(points, part: "shoulder", frames: frames)
        let hipMid = groupMidpoint(points, part: "hip", frames: frames)
        let kneeMid = groupMidpoint(points, part: "knee", frames: frames)
        let ankleMid = groupMidpoint(points, part: "ankle", frames: frames)
        let nose = point(Features.noseName)

        // The stacking offsets are the u of the midpoints: the body frame's
        // origin is the wrist midpoint, so u = 0 *is* the vertical line
        // through the hands, and a feature of 0 needs no reference line.
        let offsets = [shoulderMid, hipMid, kneeMid, ankleMid].map { $0.map(\.u) }

        var values: [String: [Double]] = [:]
        values["off_shoulder"] = offsets[0]
        values["off_hip"] = offsets[1]
        values["off_knee"] = offsets[2]
        values["off_ankle"] = offsets[3]
        // The worst of the four as a magnitude: a stack that has drifted to
        // the left is as wrong as one that has drifted to the right, and this
        // is the one number both are judged by. `rowMax` takes rows of
        // frames, so the station-major offsets are read frame by frame.
        values["line_deviation"] = rowMax(
            (0..<frames).map { index in offsets.map { abs($0[index]) } })
        // atan2 of the horizontal against the vertical component: the lean of
        // the wrist->ankle vector away from straight up, signed by which way
        // it leans.
        values["body_angle"] = ankleMid.indices.map { index in
            atan2(offsets[3][index], ankleMid[index].v) * (180.0 / Double.pi)
        }
        values["shoulder_angle"] = bothSides(
            points, chain: ["hip", "shoulder", "wrist"], frames: frames)
        values["hip_angle"] = bothSides(
            points, chain: ["shoulder", "hip", "knee"], frames: frames)
        values["knee_angle"] = bothSides(
            points, chain: ["hip", "knee", "ankle"], frames: frames)
        values["elbow_angle"] = bothSides(
            points, chain: ["shoulder", "elbow", "wrist"], frames: frames)
        values["banana"] = banana(
            shoulderMid: shoulderMid, hipMid: hipMid, ankleMid: ankleMid, nose: nose)
        values["head"] = head(hipMid: hipMid, shoulderMid: shoulderMid, nose: nose)
        // The last two are the only features that *are* a measurement between
        // the two sides, so they need both of them and are `NaN` with one — a
        // leg separation of 0 because the far leg was never seen is a leg
        // separation nobody measured.
        values["leg_separation"] = angleBetween(
            minus(point("left_ankle"), point("left_hip")),
            minus(point("right_ankle"), point("right_hip")))
        values["hand_width"] = ratio(
            distance(point("left_wrist"), point("right_wrist")),
            distance(point("left_shoulder"), point("right_shoulder")))

        // The centre of mass, from the same joints in the same frame: the
        // segment model of handstand.com, then which way the athlete faces
        // (the nose's side of the shoulder→hip torso line, so +1 is towards
        // +u), then the CoM in that direction. The facing sign is per *frame*
        // here because the hold has not been seen yet; `applyHoldFacing`
        // replaces it with the hold's majority once it has.
        let mass = try CentreOfMass.centreOfMass(uv: track.uv, joints: track.joints)
        let facing = CentreOfMass.facingSign(
            shoulderMid: shoulderMid, hipMid: hipMid, nose: nose)
        let forward = CentreOfMass.comForward(mass.u, facing)
        values["com_u"] = mass.u
        values["com_v"] = mass.v
        values["facing_sign"] = facing
        values["com_forward"] = forward

        // Every part the features are read off has a midpoint, and the wrist
        // group is where the origin comes from, so a frame whose body frame
        // could not be placed at all is not measurable however good the rest
        // of the pose looks.
        var valid = [Bool](repeating: true, count: frames)
        for mid in [wristMid, elbowMid, shoulderMid, hipMid, kneeMid, ankleMid] {
            for index in 0..<frames {
                valid[index] = valid[index] && mid[index].isFinite
            }
        }
        return BodyFeatures(
            values: values,
            valid: valid,
            comComplete: mass.complete,
            balanceZone: CentreOfMass.balanceZone(forward)
        )
    }

    /// The features of a clip with no scale: nothing measured, every frame
    /// invalid — `_unusable_features`.
    ///
    /// No body length is no body to weigh and no direction to weigh it in:
    /// the CoM columns are `NaN` like every other measurement, the
    /// completeness flag is `false` like every other frame's, and there is no
    /// zone without a position.
    static func unusableFeatures(frames: Int) -> BodyFeatures {
        var values: [String: [Double]] = [:]
        for name in Features.featureNames + Features.comColumns {
            values[name] = [Double](repeating: .nan, count: frames)
        }
        return BodyFeatures(
            values: values,
            valid: [Bool](repeating: false, count: frames),
            comComplete: [Bool](repeating: false, count: frames),
            balanceZone: [BalanceZone?](repeating: nil, count: frames)
        )
    }

    /// One facing sign per hold, and the two columns that follow from it —
    /// `apply_hold_facing`.
    ///
    /// The hold votes, every frame of the hold takes the sign most of its
    /// frames had (frames outside a hold keep their own vote, and a hold with
    /// no majority keeps its frames' `NaN`), and `com_forward` and
    /// `balance_zone` are recomputed from the result. The CoM itself is not
    /// touched: it is where it is whichever way the athlete faces.
    static func applyHoldFacing(_ features: BodyFeatures, holdId: [Int]) -> BodyFeatures {
        let facing = CentreOfMass.majorityPerHold(features.value("facing_sign"), holdId: holdId)
        let forward = CentreOfMass.comForward(features.value("com_u"), facing)
        var values = features.values
        values["facing_sign"] = facing
        values["com_forward"] = forward
        var out = features
        out.values = values
        out.balanceZone = CentreOfMass.balanceZone(forward)
        return out
    }

    // MARK: - The hold summary

    /// One row per hold: every feature's median and IQR, its length and its
    /// balance — `hold_rows`.
    ///
    /// Measured over the hold's *measurable* frames, so a hold nobody could
    /// see for part of its length is summarised over the rest, and the row
    /// carries both the number of frames in the hold and the number of frames
    /// the median is over.
    public static func holdRows(_ features: ClipFeatures) -> [HoldSummaryRow] {
        var rows: [HoldSummaryRow] = []
        for holdId in features.holdIds() {
            let mask = features.measurableHoldMask(holdId)
            let frames = features.holdId.indices.filter { features.holdId[$0] == holdId }
            var stats: [String: Double] = [:]
            for spec in Features.all {
                let measured = zip(features.value(spec.name), mask)
                    .filter { $0.1 }.map { $0.0 }
                let spread = Features.spread(measured)
                stats["\(spec.name)_median"] = roundedScalar(spread.median, digits: spec.digits)
                stats["\(spec.name)_iqr"] = roundedScalar(spread.iqr, digits: spec.digits)
            }
            let stability = holdStability(features, holdId: holdId)
            for spec in Features.balanceStats {
                stats[spec.name] = roundedScalar(
                    stability[spec.name] ?? .nan, digits: spec.digits)
            }
            rows.append(
                HoldSummaryRow(
                    holdId: holdId,
                    holdFrames: frames.count,
                    validFrames: mask.reduce(0) { $0 + ($1 ? 1 : 0) },
                    holdStartMs: features.tMs[frames.first ?? 0],
                    holdEndMs: features.tMs[frames.last ?? 0],
                    holdDurationS: roundedScalar(
                        features.holdDurationS(holdId), digits: 3),
                    stats: stats
                ))
        }
        return rows
    }

    /// How steady one hold was: the CoM's sway and speed, the two strategy
    /// angles and the zones — `hold_stability`.
    ///
    /// Everything is measured over the hold's *measurable* frames, and the
    /// CoM numbers over the subset of those with a finite `com_forward` — a
    /// frame nobody could see is not a frame the balance was measured on.
    /// The eight keys are exactly `holdStabilityNames`, `NaN` where the hold
    /// cannot support one:
    ///
    /// * `com_forward_median` — where the CoM sat, towards the fingers (+) or
    ///   the heel of the hand (−);
    /// * `com_sway_sd` — how far it wandered, the **population** SD: the
    ///   hold's frames are the whole population a judge saw, not a sample of
    ///   a bigger one, so there is no `N−1` correction;
    /// * `com_sway_range` — the same wander as p95−p5, which one spike cannot
    ///   inflate the way a max−min can;
    /// * `com_speed_rms` — how fast it moved, the RMS of `d com_forward/dt`
    ///   in `L/s`, the difference taken between consecutive measured frames
    ///   over the **real** `t_ms` gap between them, because these clips are
    ///   variable frame rate and a fixed step would make the same sway faster
    ///   on the clips that were sampled faster;
    /// * `hip_angle_sd`, `shoulder_angle_sd` — how much each joint was
    ///   correcting: the hip strategy against the shoulder strategy;
    /// * `pct_over`, `pct_under` — the percentage of those frames outside the
    ///   base of support.
    ///
    /// A spread needs two frames to be a spread, so the SDs, the range and
    /// the speed are `NaN` on a one-frame hold; the median and the
    /// percentages are defined on one.
    static func holdStability(_ features: ClipFeatures, holdId: Int) -> [String: Double] {
        var result: [String: Double] = [:]
        for name in Features.holdStabilityNames {
            result[name] = .nan
        }
        let mask = features.measurableHoldMask(holdId)
        let values = features.value("com_forward")
        let kept = mask.indices.filter { mask[$0] && values[$0].isFinite }
        let forward = kept.map { values[$0] }
        if !forward.isEmpty {
            result["com_forward_median"] = median(forward)
            let over = forward.filter { $0 > CentreOfMass.baseFront }.count
            let under = forward.filter { $0 < -CentreOfMass.baseBack }.count
            result["pct_over"] = 100.0 * Double(over) / Double(forward.count)
            result["pct_under"] = 100.0 * Double(under) / Double(forward.count)
        }
        if forward.count >= 2 {
            result["com_sway_sd"] = populationSD(forward)
            let low = PostProcess.percentileOf(forward, p: 5).value
            let high = PostProcess.percentileOf(forward, p: 95).value
            result["com_sway_range"] = high - low
            let times = kept.map { Double(features.tMs[$0]) / 1000.0 }
            var speeds: [Double] = []
            for index in 1..<times.count {
                let step = times[index] - times[index - 1]
                if step > 0.0 {
                    speeds.append((forward[index] - forward[index - 1]) / step)
                }
            }
            if !speeds.isEmpty {
                let sumSquares = speeds.reduce(0.0) { $0 + $1 * $1 }
                result["com_speed_rms"] = sqrt(sumSquares / Double(speeds.count))
            }
        }
        for name in ["hip_angle", "shoulder_angle"] {
            let angles = zip(features.value(name), mask)
                .filter { $0.1 && $0.0.isFinite }.map { $0.0 }
            if angles.count >= 2 {
                result["\(name)_sd"] = populationSD(angles)
            }
        }
        return result
    }

    /// The median and the inter-quartile range of the finite values —
    /// `_spread`.
    ///
    /// The median because a hold is judged on its shape rather than on one
    /// frame of it, and the IQR because a handstand that is perfect for 90 %
    /// of a hold and bent for the rest is a different hold from one that is
    /// the same all the way through.
    static func spread(_ values: [Double]) -> (median: Double, iqr: Double) {
        let finite = values.filter { $0.isFinite }
        if finite.isEmpty {
            return (.nan, .nan)
        }
        let low = PostProcess.percentileOf(finite, p: 25).value
        let high = PostProcess.percentileOf(finite, p: 75).value
        return (median(finite), high - low)
    }

    /// `np.median` of a list: the middle value, or the mean of the two middle
    /// ones — `NaN` for an empty list.
    static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        guard !sorted.isEmpty else { return .nan }
        if sorted.count % 2 == 0 {
            return (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2.0
        }
        return sorted[sorted.count / 2]
    }

    /// The **population** standard deviation (`ddof = 0`) — `np.std`: the
    /// hold's frames are the whole population a judge saw.
    static func populationSD(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return .nan }
        let mean = values.reduce(0.0, +) / Double(values.count)
        let squares = values.reduce(0.0) { total, value in
            let delta = value - mean
            return total + delta * delta
        }
        return sqrt(squares / Double(values.count))
    }

    /// One number rounded for the table, `NaN` left as `NaN` —
    /// `_rounded_scalar`.
    ///
    /// Python's `round(x, d)` rounds half to even on the **binary** value, so
    /// this is `(x * 10^d).rounded(.toNearestOrEven) / 10^d` rather than a
    /// decimal formatter: a tie is decided by the last bit of the float both
    /// languages store.
    static func roundedScalar(_ value: Double, digits: Int) -> Double {
        guard value.isFinite else { return .nan }
        var factor = 1.0
        for _ in 0..<Swift.max(digits, 0) {
            factor *= 10.0
        }
        return (value * factor).rounded(.toNearestOrEven) / factor
    }

    // MARK: - Geometry

    /// The `(frames, 2)` midpoint of both sides of one body part —
    /// `_group_midpoint`.
    static func groupMidpoint(
        _ points: [String: [BodyPoint]], part: String, frames: Int
    ) -> [BodyPoint] {
        sideMean(
            points["left_\(part)"] ?? [BodyPoint](repeating: .nan, count: frames),
            points["right_\(part)"] ?? [BodyPoint](repeating: .nan, count: frames)
        )
    }

    /// The two sides of a measurement as one: the mean of both, one where
    /// there is one, `NaN` where there is none — `_side_mean`, componentwise
    /// exactly as numpy broadcasts it.
    static func sideMean(_ left: Double, _ right: Double) -> Double {
        if left.isFinite && right.isFinite {
            return (left + right) / 2.0
        }
        return left.isFinite ? left : right
    }

    /// `sideMean` over two points.
    static func sideMean(_ left: BodyPoint, _ right: BodyPoint) -> BodyPoint {
        BodyPoint(u: sideMean(left.u, right.u), v: sideMean(left.v, right.v))
    }

    /// `sideMean` over two point tracks.
    static func sideMean(_ left: [BodyPoint], _ right: [BodyPoint]) -> [BodyPoint] {
        precondition(left.count == right.count, "the two sides must have the same shape")
        return left.indices.map { sideMean(left[$0], right[$0]) }
    }

    /// `sideMean` over two measurement tracks.
    static func sideMean(_ left: [Double], _ right: [Double]) -> [Double] {
        precondition(left.count == right.count, "the two sides must have the same shape")
        return left.indices.map { sideMean(left[$0], right[$0]) }
    }

    /// One side's joint angle: the angle at the middle of the three named
    /// parts — `_side_angle`. The chain is named from the first joint through
    /// the middle one to the last, so `("hip", "shoulder", "wrist")` is the
    /// angle at the shoulder.
    static func sideAngle(
        _ points: [String: [BodyPoint]], chain: [String], side: String, frames: Int
    ) -> [Double] {
        func at(_ part: String) -> [BodyPoint] {
            points["\(side)_\(part)"] ?? [BodyPoint](repeating: .nan, count: frames)
        }
        return angleAt(at(chain[0]), at(chain[1]), at(chain[2]))
    }

    /// A joint angle averaged over the sides, or taken from whichever side is
    /// there — `_both_sides`.
    static func bothSides(
        _ points: [String: [BodyPoint]], chain: [String], frames: Int
    ) -> [Double] {
        sideMean(
            sideAngle(points, chain: chain, side: "left", frames: frames),
            sideAngle(points, chain: chain, side: "right", frames: frames)
        )
    }

    /// The unsigned angle in degrees between two vectors, in `[0, 180]` —
    /// `_angle_between`.
    ///
    /// `atan2` of the cross against the dot rather than `arccos` of the
    /// normalised dot, so a nearly straight angle stays accurate and a `NaN`
    /// in either vector comes out `NaN` rather than being clipped into the
    /// range. A vector shorter than `minSegmentL` has no direction, so the
    /// angle is not a measurement.
    static func angleBetween(_ first: [BodyPoint], _ second: [BodyPoint]) -> [Double] {
        precondition(first.count == second.count, "the two vectors must have the same shape")
        return first.indices.map { index in
            let a = first[index]
            let b = second[index]
            let cross = a.u * b.v - a.v * b.u
            let dot = a.u * b.u + a.v * b.v
            let usable = cross.isFinite && dot.isFinite
                && hypot(a.u, a.v) > Features.minSegmentL
                && hypot(b.u, b.v) > Features.minSegmentL
            guard usable else { return .nan }
            return atan2(abs(cross), dot) * (180.0 / Double.pi)
        }
    }

    /// The interior angle in degrees at `middle` of the chain
    /// first→middle→last — `_angle_at`.
    ///
    /// A joint angle is the angle between the two limbs *at* the joint, which
    /// is why a straight arm and a straight leg both read 180° rather than
    /// 0°: the athlete is upside down, so the two segments point in opposite
    /// directions from it.
    static func angleAt(_ first: [BodyPoint], _ middle: [BodyPoint], _ last: [BodyPoint]) -> [Double]
    {
        angleBetween(minus(first, middle), minus(last, middle))
    }

    /// The elementwise difference of two point tracks — what every vector of
    /// the geometry is built from.
    static func minus(_ first: [BodyPoint], _ second: [BodyPoint]) -> [BodyPoint] {
        precondition(first.count == second.count, "the two tracks must have the same shape")
        return first.indices.map { index in
            BodyPoint(
                u: first[index].u - second[index].u,
                v: first[index].v - second[index].v
            )
        }
    }

    /// The largest finite value of each row, `NaN` where a row is all `NaN` —
    /// `_row_max`.
    ///
    /// A frame whose four stacking stations are all invisible is exactly
    /// that, so the rows without a finite value are taken out before the
    /// maximum is taken.
    static func rowMax(_ values: [[Double]]) -> [Double] {
        guard let firstRow = values.first, !firstRow.isEmpty else {
            return [Double](repeating: .nan, count: values.count)
        }
        return values.map { row in
            var best = -Double.infinity
            var finite = false
            for value in row where value.isFinite {
                finite = true
                best = Swift.max(best, value)
            }
            return finite ? best : .nan
        }
    }

    /// Which way the athlete faces: `+1`, `-1`, or `NaN` when the nose is on
    /// the line — features' `_facing_sign`.
    ///
    /// The body's own axis — shoulder midpoint to ankle midpoint, which in a
    /// handstand runs up the image — has a normal to it, and the nose is on
    /// one side of that line or the other. This is what makes `banana`
    /// anatomical rather than directional; the sign `com_forward` is built
    /// with is the hold's majority vote (`applyHoldFacing`).
    static func facingSign(
        shoulderMid: [BodyPoint], ankleMid: [BodyPoint], nose: [BodyPoint]
    ) -> [Double] {
        precondition(
            shoulderMid.count == ankleMid.count && shoulderMid.count == nose.count,
            "the torso line and the nose must have the same shape")
        return shoulderMid.indices.map { index in
            let axisU = ankleMid[index].u - shoulderMid[index].u
            let axisV = ankleMid[index].v - shoulderMid[index].v
            let span = hypot(axisU, axisV)
            // `-axis_v / span` and `axis_u / span`: `span` is `NaN` or zero
            // exactly when the axis has no direction, and the division is
            // numpy's, `NaN` and all.
            let acrossU = -axisV / span
            let acrossV = axisU / span
            let front = (nose[index].u - shoulderMid[index].u) * acrossU
                + (nose[index].v - shoulderMid[index].v) * acrossV
            if front > 0.0 {
                return 1.0
            }
            if front < 0.0 {
                return -1.0
            }
            return .nan
        }
    }

    /// The signed distance of the hip midpoint from the shoulder→ankle line,
    /// in L — `_banana`.
    ///
    /// A line handstand's three midpoints are collinear, so this is 0 for one
    /// and grows with how far the body curves. The sign is towards the
    /// direction the athlete faces, so positive is the belly side of the
    /// curve and negative the back side, whichever way round the camera was.
    static func banana(
        shoulderMid: [BodyPoint], hipMid: [BodyPoint], ankleMid: [BodyPoint], nose: [BodyPoint]
    ) -> [Double] {
        let facing = facingSign(
            shoulderMid: shoulderMid, ankleMid: ankleMid, nose: nose)
        return shoulderMid.indices.map { index in
            let axisU = ankleMid[index].u - shoulderMid[index].u
            let axisV = ankleMid[index].v - shoulderMid[index].v
            let span = hypot(axisU, axisV)
            let acrossU = -axisV / span
            let acrossV = axisU / span
            let offset = (hipMid[index].u - shoulderMid[index].u) * acrossU
                + (hipMid[index].v - shoulderMid[index].v) * acrossV
            return offset * facing[index]
        }
    }

    /// The angle of the shoulder→nose vector against the spine, in degrees,
    /// unsigned — `_head`.
    ///
    /// The reference is the torso direction as a vector from the hip
    /// midpoint to the shoulder midpoint. In an inverted body that points
    /// *down* the image, towards the head, which is where a head in line with
    /// the body points too. Unsigned on purpose: which side of the spine the
    /// nose is on is the only facing cue a side view has, so a head thrown
    /// back and a head dropped forward are the same fault and the same
    /// number.
    static func head(hipMid: [BodyPoint], shoulderMid: [BodyPoint], nose: [BodyPoint]) -> [Double] {
        angleBetween(minus(shoulderMid, hipMid), minus(nose, shoulderMid))
    }

    /// The distance between two points, `NaN` where either is missing —
    /// `_distance`.
    static func distance(_ first: [BodyPoint], _ second: [BodyPoint]) -> [Double] {
        precondition(first.count == second.count, "the two tracks must have the same shape")
        return first.indices.map { index in
            hypot(first[index].u - second[index].u, first[index].v - second[index].v)
        }
    }

    /// `numerator / denominator`, `NaN` where the denominator is missing or
    /// zero — `_ratio`.
    static func ratio(_ numerator: [Double], _ denominator: [Double]) -> [Double] {
        precondition(
            numerator.count == denominator.count, "the two tracks must have the same shape")
        return numerator.indices.map { index in
            let divisor = denominator[index]
            guard divisor.isFinite, abs(divisor) > Features.minSegmentL else { return .nan }
            return numerator[index] / divisor
        }
    }

    // MARK: - Shared shape checks (Python's ValueError messages, kept)

    private static func sameShape(_ a: [[Double]], _ b: [[Double]]) -> Bool {
        guard a.count == b.count else { return false }
        guard let columns = a.first else { return true }
        return a.indices.allSatisfy { a[$0].count == columns.count && b[$0].count == columns.count }
    }

    private static func sameShape(_ a: [[Double]], _ b: [[Bool]]) -> Bool {
        guard a.count == b.count else { return false }
        guard let columns = a.first else { return true }
        return a.indices.allSatisfy { a[$0].count == columns.count && b[$0].count == columns.count }
    }
}
