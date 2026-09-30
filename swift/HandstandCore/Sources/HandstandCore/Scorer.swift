import Foundation

// --------------------------------------------------------------------------- #
// The weighted z-score scorer (#29), on device — the end of the on-device
// chain: PostProcess (#39) -> PhaseSegmenter (#40) -> Features (#81) -> this.
//
// A port of `pipeline/handstand/score.py`: the constants, the reference
// validation, `holdValues`, `scoreHold`, `_top_faults`, `scoreClip` and
// `clipScore`. The CSV table, the summary and the CLI are **not** ported:
// this is the maths the app needs, nothing else. `docs/scoring.md` documents
// the format of both the score and the reference file, and this file follows
// it line for line.
//
// The rules the port lives by, taken straight from Python:
//
// * the scorer reads the **rounded** per-frame table: Python's stage reads
//   the parquet `features.feature_table` writes, so `Scorer` applies
//   `ClipFeatures.tableRounded()` first — that is what makes the on-device
//   score equal the pipeline's;
// * `SIGNED_BY_FACING`'s five features are multiplied by each frame's
//   `facing_sign` before anything is summarised (a NaN sign making that frame
//   NaN, which drops it out of the median like any frame nobody measured);
//   the hold values are medians over the hold's *measurable* frames and the
//   two spreads are **population** SDs of the same frames' finite values;
// * per feature `z = (value - mean) / max(sd, SD_FLOOR)` capped at `Z_CAP`,
//   each group takes the mean capped `|z|` of the features it could compare,
//   the deviation `D` is the weight average over the groups that are there
//   (so a missing group renormalises the weights), `P` is the one-sided hip
//   penalty, and the score is Python's `round(…, 1)` — half to even;
// * `NaN` propagates exactly where Python lets it: a value nobody measured is
//   `NaN` and drops out of its median, a group with nothing to compare is
//   `nil`, and a hold with every group missing has no score at all —
//   "nothing to compare".
//
// Who owns what: #28 builds the real reference from the labelled good clips
// (nothing is bundled here), #30 tunes `groups` and the constants.
// --------------------------------------------------------------------------- #

/// A problem with a reference file, as `ScoreReference.decode` throws it —
/// Python's `ValueError` from `load_reference`, with the same message text
/// minus the file path (a `Data` knows no path).
public enum ScoreReferenceError: Error, Equatable, Sendable {
    /// The bytes are not JSON at all — Python's
    /// `"{file}: not valid JSON ({error})"`.
    case invalidJSON(String)
    /// The document is JSON but not a usable reference; the associated value
    /// names the problem the way Python's `ValueError` does.
    case invalid(String)
}

extension ScoreReferenceError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .invalidJSON(let message), .invalid(let message):
            return message
        }
    }
}

/// What a good hold looks like: per feature the mean, the SD and how many
/// holds they came from — `handstand.score.Reference`, ready to score with.
///
/// `features` is keyed by `Scorer.scoreFeatures` names — a feature left out is
/// one simply not scored against, never a zero, which is what lets a partial
/// reference exist — and `nHolds`/`builtFrom` say where the numbers came from.
public struct ScoreReference: Sendable, Equatable {
    /// One feature's entry: `(mean, sd, n)` — `features[name]` of the file.
    public struct Stat: Sendable, Equatable {
        /// The mean of the good holds' values.
        public var mean: Double
        /// Their **population** SD, `>= 0`.
        public var sd: Double
        /// How many finite values the two were taken over.
        public var n: Int

        public init(mean: Double, sd: Double, n: Int) {
            self.mean = mean
            self.sd = sd
            self.n = n
        }
    }

    /// One entry per scored feature the file carried, by name.
    public var features: [String: Stat]
    /// How many holds the numbers came from — `n_holds`.
    public var nHolds: Int
    /// Which holds, in free text — `built_from`.
    public var builtFrom: String

    public init(features: [String: Stat], nHolds: Int, builtFrom: String) {
        self.features = features
        self.nHolds = nHolds
        self.builtFrom = builtFrom
    }

    /// Read and validate a reference file's bytes — `load_reference` +
    /// `_build_reference`.
    ///
    /// Throws `ScoreReferenceError` naming the problem the way Python's
    /// `ValueError` does: the schema, version or signs marker, a mean or sd
    /// that is not a finite number, a negative sd, a count that is not a
    /// non-negative integer, or a feature name outside
    /// `Scorer.scoreFeatures`. Features the document does not carry are fine —
    /// they just are not scored.
    public static func decode(_ data: Data) throws -> ScoreReference {
        let object: Any
        do {
            object = try JSONSerialization.jsonObject(with: data, options: [.allowFragments])
        } catch {
            throw ScoreReferenceError.invalidJSON("not valid JSON (\(error))")
        }
        return try build(object)
    }

    /// The parsed reference behind `decode`, refusing to guess — `_build_reference`.
    private static func build(_ object: Any) throws -> ScoreReference {
        guard let data = object as? [String: Any] else {
            throw ScoreReferenceError.invalid(
                "a reference must be a JSON object, got \(describe(object))")
        }

        let schema = data["schema"]
        guard let schemaName = schema as? String, schemaName == Scorer.referenceSchema else {
            throw ScoreReferenceError.invalid(
                "schema is \(describe(schema)), expected '\(Scorer.referenceSchema)'")
        }

        // Python compares with `!=`, so any number equal to the version passes
        // (1, 1.0 — even `true`, which Python's `==` reads as 1); anything
        // else, a string included, is refused with its own spelling.
        let version = data["version"]
        guard let versionNumber = version as? NSNumber,
            versionNumber.doubleValue == Double(Scorer.referenceVersion)
        else {
            throw ScoreReferenceError.invalid(
                "version is \(describe(version)), expected \(Scorer.referenceVersion)")
        }

        let signs = data["signs"]
        guard let signText = signs as? String, signText == Scorer.referenceSigns else {
            throw ScoreReferenceError.invalid(
                "signs are \(describe(signs)), expected '\(Scorer.referenceSigns)' — the hold "
                    + "values this module scores are athlete-signed")
        }

        let builtFrom = data["built_from"]
        guard let provenance = builtFrom as? String else {
            throw ScoreReferenceError.invalid(
                "built_from must be free text, got \(describe(builtFrom))")
        }

        let nHolds = try count(data["n_holds"], origin: "n_holds")

        let entries = data["features"]
        guard let table = entries as? [String: Any] else {
            throw ScoreReferenceError.invalid(
                "features must be an object, got \(describe(entries))")
        }
        var parsed: [String: Stat] = [:]
        for (name, entry) in table {
            guard Scorer.scoreFeatures.contains(name) else {
                throw ScoreReferenceError.invalid(
                    "'\(name)' is not a scored feature (the keys are SCORE_FEATURES)")
            }
            parsed[name] = try featureStats(entry, origin: "features['\(name)']")
        }
        return ScoreReference(features: parsed, nHolds: nHolds, builtFrom: provenance)
    }

    /// One feature's `(mean, sd, n)`, with every part checked by name —
    /// `_feature_stats`.
    private static func featureStats(_ entry: Any, origin: String) throws -> Stat {
        guard let fields = entry as? [String: Any] else {
            throw ScoreReferenceError.invalid(
                "\(origin) must be an object with mean, sd and n, got \(describe(entry))")
        }
        let mean = try finite(fields["mean"], origin: "\(origin).mean")
        let sd = try finite(fields["sd"], origin: "\(origin).sd")
        if sd < 0.0 {
            throw ScoreReferenceError.invalid("\(origin).sd must be >= 0, got \(sd)")
        }
        let n = try count(fields["n"], origin: "\(origin).n")
        return Stat(mean: mean, sd: sd, n: n)
    }

    /// `value` as a finite float, or an error naming `origin` — `_finite`.
    /// Python refuses a boolean first, so `true` is no mean of anything.
    private static func finite(_ value: Any?, origin: String) throws -> Double {
        guard let number = value as? NSNumber, !isBool(value), number.doubleValue.isFinite else {
            throw ScoreReferenceError.invalid(
                "\(origin) must be a finite number, got \(describe(value))")
        }
        return number.doubleValue
    }

    /// `value` as a non-negative whole number, or an error naming `origin` —
    /// `_count`. A boolean is refused first, and so is a fractional number:
    /// Python's `isinstance(4.0, int)` is False, so `4.0` is no count either.
    private static func count(_ value: Any?, origin: String) throws -> Int {
        guard let number = value as? NSNumber, !isBool(value), isInteger(number), number.int64Value >= 0
        else {
            throw ScoreReferenceError.invalid(
                "\(origin) must be a non-negative integer, got \(describe(value))")
        }
        return Int(number.int64Value)
    }

    // MARK: - JSON values as Python spells them

    /// Is this JSON value a boolean? The identity check against `kCFBoolean`
    /// is the only reliable one: `NSNumber(1)` is *not* the boolean singleton,
    /// so `1` and `true` stay the different things Python keeps them.
    static func isBool(_ value: Any?) -> Bool {
        guard let value else { return false }
        let object = value as AnyObject
        return object === kCFBooleanTrue || object === kCFBooleanFalse
    }

    /// A JSON number Python would have read as an `int` rather than a float.
    /// `JSONSerialization` types every integral literal without `.` or an
    /// exponent as an integer and every other number as a double, which is
    /// exactly Python's `isinstance(value, int)` split.
    static func isInteger(_ number: NSNumber) -> Bool {
        String(cString: number.objCType) != "d"
    }

    /// One JSON value in Python's `repr` spelling — `None`, `True`, `'text'`,
    /// `42`, `1.5`, `[a, b]`, `{'k': v}` — so an error message reads the same
    /// on both sides. Dictionary keys are sorted: Swift's hashing has no
    /// insertion order to fall back on, and a message must be reproducible.
    static func describe(_ value: Any?) -> String {
        guard let value, !(value is NSNull) else { return "None" }
        if isBool(value) {
            return (value as! NSNumber).boolValue ? "True" : "False"
        }
        if let number = value as? NSNumber {
            if isInteger(number) {
                return String(number.int64Value)
            }
            return String(number.doubleValue)
        }
        if let text = value as? String {
            return "'\(text)'"
        }
        if let array = value as? [Any] {
            return "[" + array.map { describe($0) }.joined(separator: ", ") + "]"
        }
        if let table = value as? [String: Any] {
            let items = table.keys.sorted().map { "'\(String($0))': \(describe(table[$0]!))" }
            return "{" + items.joined(separator: ", ") + "}"
        }
        return String(describing: value)
    }
}

/// One hold's score, and everything the score was made of —
/// `handstand.score.HoldScore`.
///
/// `holdFrames` counts every frame the hold spans while `validFrames` counts
/// the ones the values could be measured on, and the two times and the
/// duration are first frame to last — the same row `Features.holdRows`
/// writes, so a score and a hold summary line up. `score` is `nil` with a
/// `reason` when the hold could not be scored; `deviation` (D) and `penalty`
/// (P) are what it would be scored from, `groups` the per-group deviations
/// (`nil` where the group is missing), `values` and `z` the hold's values and
/// their capped z-scores, `topFaults` the worst features, and
/// `missingGroups` which groups had nothing to compare.
public struct HoldScore: Sendable, Equatable {
    /// One of `topFaults`: a feature, the hold's value, the reference's mean
    /// and the capped z it was ranked by — Python's
    /// `(feature, value, mean, z)` tuple, as a struct because tuples of four
    /// do not make a type `Equatable`.
    public struct TopFault: Sendable, Equatable {
        /// The feature's name, a `Scorer.scoreFeatures` entry.
        public var feature: String
        /// The hold's value, athlete-signed.
        public var value: Double
        /// The reference's mean for it.
        public var mean: Double
        /// The capped z-score the fault was ranked by.
        public var z: Double

        public init(feature: String, value: Double, mean: Double, z: Double) {
            self.feature = feature
            self.value = value
            self.mean = mean
            self.z = z
        }
    }

    /// The hold's number — `hold_id`.
    public var holdId: Int
    /// How many frames the hold covers, including invalid ones — `hold_frames`.
    public var holdFrames: Int
    /// How many of those the values were measured on — `valid_frames`.
    public var validFrames: Int
    /// The `t_ms` of the hold's first frame — `hold_start_ms`.
    public var holdStartMs: Int
    /// The `t_ms` of the hold's last frame — `hold_end_ms`.
    public var holdEndMs: Int
    /// First frame to last in seconds, rounded to three decimals —
    /// `hold_duration_s`.
    public var holdDurationS: Double
    /// The score out of 100; `nil` exactly when `reason` says why there is
    /// no score — `score`.
    public var score: Double?
    /// `""` for a scored hold, else why it is not ("too few valid frames",
    /// "nothing to compare") — `reason`.
    public var reason: String
    /// The weight-averaged group deviation D — `deviation`.
    public var deviation: Double
    /// The one-sided hip penalty P — `penalty`.
    public var penalty: Double
    /// Per-group deviation, `nil` where the group is missing — `groups`.
    public var groups: [String: Double?]
    /// The hold's values, `NaN` where it could not measure one — `values`.
    public var values: [String: Double]
    /// The capped z-scores, only for features that could be compared — `z`.
    public var z: [String: Double]
    /// The worst faults of the hold, worst first — `top_faults`.
    public var topFaults: [TopFault]
    /// The groups with nothing to compare — `missing_groups`.
    public var missingGroups: [String]

    public init(
        holdId: Int,
        holdFrames: Int,
        validFrames: Int,
        holdStartMs: Int,
        holdEndMs: Int,
        holdDurationS: Double,
        score: Double?,
        reason: String,
        deviation: Double,
        penalty: Double,
        groups: [String: Double?],
        values: [String: Double],
        z: [String: Double],
        topFaults: [TopFault],
        missingGroups: [String]
    ) {
        self.holdId = holdId
        self.holdFrames = holdFrames
        self.validFrames = validFrames
        self.holdStartMs = holdStartMs
        self.holdEndMs = holdEndMs
        self.holdDurationS = holdDurationS
        self.score = score
        self.reason = reason
        self.deviation = deviation
        self.penalty = penalty
        self.groups = groups
        self.values = values
        self.z = z
        self.topFaults = topFaults
        self.missingGroups = missingGroups
    }
}

/// Port of `pipeline/handstand/score.py`: score every hold of a clip against
/// a reference of good holds.
///
/// The entry points are `scoreClip(_:reference:)` → `[HoldScore]` (one per
/// hold, in `hold_id` order) and `clipScore(_:)` → the hold a clip is
/// represented by; `scoreHold(_:holdId:reference:)` and
/// `holdValues(_:holdId:)` are the single-hold pieces Python names the same
/// way. Every one of them rounds the features first, so the numbers are the
/// pipeline's.
public enum Scorer {
    // MARK: - Constants, one per Python module constant

    /// The file's markers, validated by `ScoreReference.decode`: the name
    /// identifies the file, `version` says which reader it was written for,
    /// and `signs` says the numbers are already athlete-signed —
    /// `REFERENCE_SCHEMA`, `REFERENCE_VERSION`, `REFERENCE_SIGNS`.
    public static let referenceSchema = "handstand-reference"
    public static let referenceVersion = 1
    public static let referenceSigns = "athlete"

    /// `SIGNED_BY_FACING` — the five features #22/#23 write in **image**
    /// coordinates (+ = screen right), converted to athlete-signed values by
    /// multiplying with that frame's `facing_sign` before anything is
    /// summarised or scored. Without the flip a mirrored recording would look
    /// like an opposite fault; `banana` and `com_forward` are already signed
    /// and the rest are unsigned magnitudes.
    public static let signedByFacing: [String] = [
        "off_shoulder", "off_hip", "off_knee", "off_ankle", "body_angle",
    ]

    /// `SCORE_FEATURES` — the values a hold is scored on: the fourteen
    /// per-frame features (medians over the hold) plus the two spreads.
    /// Not scored: `hand_width`, which the side view cannot trust.
    public static let scoreFeatures: [String] = [
        "off_shoulder", "off_hip", "off_knee", "off_ankle", "body_angle",
        "line_deviation", "shoulder_angle", "hip_angle", "knee_angle", "elbow_angle",
        "banana", "head", "leg_separation", "com_forward",
        "com_sway_sd", "hip_angle_sd",
    ]

    /// `_DERIVED` — the two scored values that are not medians of a column:
    /// spreads computed by `holdValues` from the frames it already holds.
    static let derived: [String] = ["com_sway_sd", "hip_angle_sd"]

    /// `_MEDIAN_FEATURES` — every scored feature but the two spreads, in
    /// `scoreFeatures` order.
    static let medianFeatures: [String] = scoreFeatures.filter { !derived.contains($0) }

    /// One group a hold is judged in: its name, its weight and the features
    /// it spreads that weight over — one row of Python's `GROUPS`.
    public struct Group: Sendable, Equatable {
        /// The group's name, e.g. `stack`.
        public var name: String
        /// Its share of the deviation; the weights sum to 1.0 (#30 tunes them).
        public var weight: Double
        /// The features the group is the mean capped |z| over.
        public var features: [String]

        public init(name: String, weight: Double, features: [String]) {
            self.name = name
            self.weight = weight
            self.features = features
        }
    }

    /// `GROUPS` — what a judge would say out loud ("the stack is off"), with
    /// the starting weights #30 will tune. A group's deviation is the mean
    /// capped |z| over the features of it that could be compared, so one wild
    /// feature cannot carry a group of six.
    public static let groups: [Group] = [
        Group(
            name: "stack", weight: 0.25,
            features: [
                "off_shoulder", "off_hip", "off_knee", "off_ankle", "line_deviation",
                "body_angle",
            ]),
        // Open shoulders: the one piece of style advice every judge gives.
        Group(name: "shoulder", weight: 0.20, features: ["shoulder_angle"]),
        // The line-vs-pike difference, and the curve that comes with it.
        Group(name: "hip", weight: 0.20, features: ["hip_angle", "banana"]),
        // Where the centre of mass sat and how far it wandered: balance, not shape.
        Group(name: "com", weight: 0.15, features: ["com_forward", "com_sway_sd"]),
        // A bent arm is a severe fault, which is why it speaks louder than
        // its share of the body.
        Group(name: "elbows", weight: 0.10, features: ["elbow_angle"]),
        Group(name: "knees_toes", weight: 0.05, features: ["knee_angle", "leg_separation"]),
        Group(name: "head", weight: 0.05, features: ["head"]),
    ]

    /// `_GROUP_FOR` — which group each feature is scored in, with the group's
    /// weight and size: exactly what `topFaults` ranks by, a feature's share
    /// of the deviation being `weight * |z| / size`.
    struct GroupRole {
        var name: String
        var weight: Double
        var size: Int
    }

    static let groupFor: [String: GroupRole] = {
        var roles: [String: GroupRole] = [:]
        for group in Scorer.groups {
            for feature in group.features {
                roles[feature] = GroupRole(
                    name: group.name, weight: group.weight, size: group.features.count)
            }
        }
        return roles
    }()

    /// `MIN_SCORE_FRAMES` — a hold with fewer valid frames is not scored: a
    /// median of four frames of a body mostly out of view is a number, not a
    /// measurement.
    public static let minScoreFrames = 5

    /// `Z_CAP` — the cap on |z|, and the yardstick the score is cut from:
    /// `D + P` of one `zCap` scores 0, so one absurd feature cannot drag an
    /// otherwise good hold to zero by itself.
    public static let zCap = 4.0

    /// `SD_FLOOR_L` — the smallest reference SD used for a length feature, in
    /// body lengths: below a hundredth of a body the spread between good holds
    /// is measurement noise, and dividing by it would make the z infinite.
    public static let sdFloorL = 0.01

    /// `SD_FLOOR_DEG` — the same floor for angles, in degrees: a reference
    /// whose good holds agree within a degree cannot promise a tenth of one.
    public static let sdFloorDeg = 1.0

    /// `HIP_PENALTY_WEIGHT` — how much of a z of the hip's own sway the
    /// one-sided hip penalty takes: a hip busier than the reference's is
    /// paying for the balance out of the score, at most `HIP_PENALTY_WEIGHT *
    /// Z_CAP` (0.4) points of deviation.
    public static let hipPenaltyWeight = 0.10

    /// `TOP_FAULTS` — how many faults `topFaults` reports: three is what a
    /// person can act on.
    public static let topFaults = 3

    /// `_SD_FLOORS` — which SD floor each scored feature is divided by:
    /// lengths in body lengths, angles in degrees, taken from the feature's
    /// own unit the way Python reads it off `features.FEATURES`, and spelled
    /// out for the three columns that are not in it (both CoM numbers are in
    /// body lengths like every other length here).
    static let sdFloors: [String: Double] = {
        var floors: [String: Double] = [:]
        for spec in Features.all where Scorer.scoreFeatures.contains(spec.name) {
            floors[spec.name] = spec.unit == "L" ? Scorer.sdFloorL : Scorer.sdFloorDeg
        }
        floors["com_forward"] = Scorer.sdFloorL
        floors["com_sway_sd"] = Scorer.sdFloorL
        floors["hip_angle_sd"] = Scorer.sdFloorDeg
        return floors
    }()

    // MARK: - The values a hold is scored on

    /// One hold's athlete-signed values, keyed by `scoreFeatures` —
    /// `hold_values`.
    ///
    /// The medians are taken over the hold's measurable frames
    /// (`holdId == h && valid`) and the two spreads over the same frames'
    /// finite values, all in the units the reference speaks: the five
    /// `signedByFacing` features are multiplied by each frame's `facing_sign`
    /// first (a NaN sign making that frame's value NaN), and a feature nobody
    /// could measure comes back `NaN` rather than as a number out of nothing.
    ///
    /// The table is rounded first, because that is the table Python reads —
    /// see `ClipFeatures.tableRounded`.
    public static func holdValues(_ features: ClipFeatures, holdId: Int) -> [String: Double] {
        holdValues(of: features.tableRounded(), holdId: holdId)
    }

    /// `holdValues` over a table that is already rounded — what `scoreHold`
    /// calls so the table is rounded once rather than twice.
    static func holdValues(of table: ClipFeatures, holdId: Int) -> [String: Double] {
        let mask = zip(table.holdId, table.valid).map { pair in pair.0 == holdId && pair.1 }
        let facing = table.value("facing_sign")
        var values: [String: Double] = [:]
        for name in medianFeatures {
            var column = table.value(name)
            if signedByFacing.contains(name) {
                column = zip(column, facing).map { pair in pair.0 * pair.1 }
            }
            values[name] = medianFinite(selected(column, mask: mask))
        }
        values["com_sway_sd"] = sdFinite(selected(table.value("com_forward"), mask: mask))
        values["hip_angle_sd"] = sdFinite(selected(table.value("hip_angle"), mask: mask))
        return values
    }

    /// The elements of `values` whose mask entry is true, in order.
    private static func selected(_ values: [Double], mask: [Bool]) -> [Double] {
        zip(values, mask).filter { $0.1 }.map { $0.0 }
    }

    /// The median of the finite values, `NaN` when there are none —
    /// `_median_finite` (numpy's `nanmedian` semantics).
    private static func medianFinite(_ values: [Double]) -> Double {
        let finite = values.filter { $0.isFinite }
        if finite.isEmpty { return .nan }
        return Features.median(finite)
    }

    /// The **population** SD of the finite values, `NaN` under two of them —
    /// `_sd_finite`. The hold's frames are the whole population a judge saw,
    /// not a sample of a bigger one, so there is no N-1 correction — the same
    /// answer `Features.holdStability` gives `com_sway_sd`.
    private static func sdFinite(_ values: [Double]) -> Double {
        let finite = values.filter { $0.isFinite }
        if finite.count < 2 { return .nan }
        return Features.populationSD(finite)
    }

    // MARK: - Scoring one hold

    /// Score one hold of a clip against `reference`, or say why it is not
    /// scored — `score_hold`.
    ///
    /// Three answers are possible: a score out of 100; `"nothing to compare"`,
    /// when the reference carries no feature of any group this hold could
    /// offer a value for; and `"too few valid frames"`, below
    /// `minScoreFrames`, which is decided before anything is measured.
    public static func scoreHold(
        _ features: ClipFeatures, holdId: Int, reference: ScoreReference
    ) -> HoldScore {
        scoreHold(on: features.tableRounded(), holdId: holdId, reference: reference)
    }

    /// `scoreHold` over a table that is already rounded.
    static func scoreHold(
        on table: ClipFeatures, holdId: Int, reference: ScoreReference
    ) -> HoldScore {
        let frames = table.holdId.indices.filter { table.holdId[$0] == holdId }
        let holdFrames = frames.count
        let validFrames = frames.reduce(0) { total, index in total + (table.valid[index] ? 1 : 0) }
        let holdStartMs = frames.first.map { table.tMs[$0] } ?? 0
        let holdEndMs = frames.last.map { table.tMs[$0] } ?? 0
        let holdDurationS = frames.isEmpty
            ? 0.0
            : Features.roundedScalar(Double(holdEndMs - holdStartMs) / 1000.0, digits: 3)

        var groupDeviations: [String: Double?] = [:]
        for group in Scorer.groups {
            groupDeviations.updateValue(nil, forKey: group.name)
        }

        guard validFrames >= minScoreFrames else {
            return HoldScore(
                holdId: holdId,
                holdFrames: holdFrames,
                validFrames: validFrames,
                holdStartMs: holdStartMs,
                holdEndMs: holdEndMs,
                holdDurationS: holdDurationS,
                score: nil,
                reason: "too few valid frames",
                deviation: 0.0,
                penalty: 0.0,
                groups: groupDeviations,
                values: [:],
                z: [:],
                topFaults: [],
                missingGroups: Scorer.groups.map(\.name)
            )
        }

        let values = holdValues(of: table, holdId: holdId)
        var z: [String: Double] = [:]
        for name in scoreFeatures {
            guard let stat = reference.features[name], let value = values[name], value.isFinite
            else {
                continue
            }
            let floor = sdFloors[name] ?? sdFloorL
            let raw = (value - stat.mean) / Swift.max(stat.sd, floor)
            // The z the score is made of: signed, but never beyond the cap, so
            // one broken feature is worth at most Z_CAP like any other. Only a
            // NaN input can pass a NaN through the cap, as numpy's clip does.
            z[name] = raw.isNaN ? .nan : Swift.min(Swift.max(raw, -zCap), zCap)
        }

        var missing: [String] = []
        var weighted = 0.0
        var weights = 0.0
        for group in Scorer.groups {
            let comparable = group.features.compactMap { z[$0] }
            if comparable.isEmpty {
                groupDeviations.updateValue(nil, forKey: group.name)
                missing.append(group.name)
                continue
            }
            let deviation =
                comparable.reduce(0.0) { $0 + Swift.abs($1) } / Double(comparable.count)
            groupDeviations.updateValue(deviation, forKey: group.name)
            weighted += group.weight * deviation
            weights += group.weight
        }

        let hipSway = z["hip_angle_sd"]
        let penalty = hipSway.map { hipPenaltyWeight * Swift.max(0.0, $0) } ?? 0.0

        let deviation: Double
        let score: Double?
        let reason: String
        if weights > 0.0 {
            deviation = weighted / weights
            score = Features.roundedScalar(
                100.0 * Swift.max(0.0, 1.0 - (deviation + penalty) / zCap), digits: 1)
            reason = ""
        } else {
            deviation = 0.0
            score = nil
            reason = "nothing to compare"
        }

        return HoldScore(
            holdId: holdId,
            holdFrames: holdFrames,
            validFrames: validFrames,
            holdStartMs: holdStartMs,
            holdEndMs: holdEndMs,
            holdDurationS: holdDurationS,
            score: score,
            reason: reason,
            deviation: deviation,
            penalty: penalty,
            groups: groupDeviations,
            values: values,
            z: z,
            topFaults: rankedTopFaults(values: values, z: z, reference: reference),
            missingGroups: missing
        )
    }

    /// The `topFaults` features carrying the largest share of the deviation —
    /// `_top_faults`.
    ///
    /// A feature's share of the score is its group's weight times its capped
    /// |z|, over the number of features the group spreads that weight across —
    /// so a two-SD hip speaks louder than a two-SD knee. Ties keep
    /// `scoreFeatures` order (Python's sort is stable over `z`'s insertion
    /// order, so the candidates are taken in that order and the sort below is
    /// made stable by hand), and `hip_angle_sd` is never one: no group takes
    /// it, it is the penalty's own input.
    static func rankedTopFaults(
        values: [String: Double], z: [String: Double], reference: ScoreReference
    ) -> [HoldScore.TopFault] {
        let candidates = scoreFeatures.filter { z[$0] != nil && groupFor[$0] != nil }
        let ranked = candidates.enumerated()
            .map { offset, name -> (offset: Int, share: Double, name: String) in
                let role = groupFor[name]!
                let share = role.weight * Swift.abs(z[name]!) / Double(role.size)
                return (offset, share, name)
            }
            .sorted { first, second in
                if first.share != second.share {
                    return first.share > second.share
                }
                return first.offset < second.offset
            }
        return ranked.prefix(Scorer.topFaults).map { entry in
            HoldScore.TopFault(
                feature: entry.name,
                value: values[entry.name] ?? .nan,
                mean: reference.features[entry.name]?.mean ?? .nan,
                z: z[entry.name] ?? .nan
            )
        }
    }

    // MARK: - A whole clip

    /// Score every hold of a clip, in `hold_id` order — `score_clip`.
    ///
    /// Frames outside a hold (`holdId == -1`) are not holds and are skipped:
    /// a warm-up, a bail and a hand step are all one hold each or none.
    public static func scoreClip(
        _ features: ClipFeatures, reference: ScoreReference
    ) -> [HoldScore] {
        let table = features.tableRounded()
        return features.holdIds().map { scoreHold(on: table, holdId: $0, reference: reference) }
    }

    /// The hold a clip is represented by: its longest **scored** hold —
    /// `clip_score`.
    ///
    /// A short hold is a hop on the hands and its median says more about
    /// balance than about form, so the clip's score is the longest hold
    /// anyone could actually score, ties going to the earlier one — the same
    /// rule `ClipFeatures.longestHoldId` picks a clip's hold by. `nil` when no
    /// hold of the clip was scored.
    public static func clipScore(_ scores: [HoldScore]) -> HoldScore? {
        var best: HoldScore?
        for hold in scores where hold.score != nil {
            if let current = best, current.holdDurationS >= hold.holdDurationS {
                continue  // ties go to the earlier hold, as Python's max does
            }
            best = hold
        }
        return best
    }
}
