import XCTest

@testable import HandstandCore

/// The scorer (#41): golden parity first, then the unit cases of
/// `pipeline/tests/test_score.py` one for one.
///
/// The parity test runs every committed fixture through the whole on-device
/// chain — `PostProcess.process` → `PhaseSegmenter.classify` →
/// `Features.extract` → `Scorer.scoreClip` — against `parity_reference.json`
/// and compares the answer with `expected.score` inside `meta.tolerances`
/// (integers, strings, `missing_groups` and the top-fault *names* exact;
/// `null` ⇔ `nil`/`NaN`). A mismatch names the case, the hold, the key and
/// both values.
///
/// The unit tests build their tables column by column the way
/// `test_score.py` does — every reference is the inline `GOOD` table, never a
/// committed number: the real reference is chainlink #28's, and nothing here
/// may look like one. Where Python's tests run an *unrounded* table and
/// assert a deviation of 0 to twelve decimals, these run the **rounded**
/// table the scorer really reads (`tableRounded`), so a value Python pins to
/// `approx(0)` is asserted within 1e-4 here — the same tolerance the golden
/// fixtures carry for `expected.score.deviation` and `.groups`.
final class ScorerTests: XCTestCase {
    // MARK: - The table Python's tests build, column by column

    /// What the tests' inline reference says a good hold shows, as
    /// `(mean, sd)` per scored feature — `GOOD` in `test_score.py`. Every SD
    /// is above the floor of its unit (lengths in body lengths, angles in
    /// degrees), so a test can say "two SDs off" and mean it.
    static let good: [String: (mean: Double, sd: Double)] = [
        "off_shoulder": (0.0, 0.03),
        "off_hip": (0.0, 0.03),
        "off_knee": (0.0, 0.04),
        "off_ankle": (0.0, 0.05),
        "line_deviation": (0.05, 0.03),
        "body_angle": (0.0, 3.0),
        "shoulder_angle": (178.0, 4.0),
        "hip_angle": (176.0, 3.0),
        "knee_angle": (179.0, 2.0),
        "elbow_angle": (175.0, 4.0),
        "banana": (0.02, 0.03),
        "head": (18.0, 6.0),
        "leg_separation": (3.0, 4.0),
        "com_forward": (0.01, 0.02),
        "com_sway_sd": (0.02, 0.01),
        "hip_angle_sd": (1.0, 0.5),
    ]

    /// `frames` values whose median is `mean` and whose population SD is
    /// `sd` — Python's `spread`: symmetric around the mean and scaled to unit
    /// spread first, which is how a test holds a hold's median and its sway
    /// apart.
    static func spread(_ mean: Double, _ sd: Double, frames: Int = 10) -> [Double] {
        guard frames >= 2 else { return [Double](repeating: mean, count: frames) }
        let center = Double(frames - 1) / 2.0
        let steps = (0..<frames).map { Double($0) - center }
        let scale = Features.populationSD(steps)
        return steps.map { mean + sd * ($0 / scale) }
    }

    /// `count` copies of `value` — Python broadcasts scalars.
    private func column(_ value: Double, _ count: Int) -> [Double] {
        [Double](repeating: value, count: count)
    }

    /// `count` copies of `value` as a `hold_id` column.
    private func ints(_ value: Int, _ count: Int) -> [Int] {
        [Int](repeating: value, count: count)
    }

    /// `count` copies of `value` as a `valid` column.
    private func flags(_ value: Bool, _ count: Int) -> [Bool] {
        [Bool](repeating: value, count: count)
    }

    /// `groups` with `nil` — a group with nothing to compare — as `NaN`,
    /// because `groups[name]` on a `[String: Double?]` is a `Double??` no
    /// assertion wants to unwrap.
    private func flatGroups(_ groups: [String: Double?]) -> [String: Double] {
        groups.mapValues { $0 ?? .nan }
    }

    /// A minimal per-frame features table for the scoring functions —
    /// `frame_table`. Every column `Scorer.holdValues` reads defaults to a
    /// value a good hold would show (see `good`), so a test overrides only
    /// what it is about; `t_ms` is `0, 33, 66, …` like Python's `FRAME_MS`.
    private func frameTable(
        frames: Int = 10,
        holdId: [Int]? = nil,
        valid: [Bool]? = nil,
        overrides: [String: [Double]] = [:]
    ) -> ClipFeatures {
        var values: [String: [Double]] = [:]
        for name in Scorer.medianFeatures {
            values[name] = [Double](repeating: Self.good[name]!.mean, count: frames)
        }
        values["facing_sign"] = column(1.0, frames)
        for (name, column) in overrides {
            precondition(
                column.count == frames, "expected \(frames) values for \(name), got \(column.count)")
            values[name] = column
        }
        return ClipFeatures(
            tMs: (0..<frames).map { $0 * 33 },
            phase: [Phase](repeating: .hold, count: frames),
            holdId: holdId ?? ints(0, frames),
            values: values,
            balanceZone: [BalanceZone?](repeating: nil, count: frames),
            comComplete: flags(true, frames),
            valid: valid ?? flags(true, frames),
            unusableReason: ""
        )
    }

    /// A hold whose values all sit on the reference's means, spreads included
    /// — `at_means_table`: `com_forward` gets an array of the right median and
    /// the right population SD, `hip_angle` likewise, so a hold scored from
    /// this against `good` is exactly 100 — and a test can move one feature
    /// off it and know the rest did not budge.
    private func atMeansTable(
        frames: Int = 10,
        holdId: [Int]? = nil,
        valid: [Bool]? = nil,
        overrides: [String: [Double]] = [:]
    ) -> ClipFeatures {
        var columns = overrides
        if columns["com_forward"] == nil {
            columns["com_forward"] = Self.spread(
                Self.good["com_forward"]!.mean, Self.good["com_sway_sd"]!.mean, frames: frames)
        }
        if columns["hip_angle"] == nil {
            columns["hip_angle"] = Self.spread(
                Self.good["hip_angle"]!.mean, Self.good["hip_angle_sd"]!.mean, frames: frames)
        }
        return frameTable(frames: frames, holdId: holdId, valid: valid, overrides: columns)
    }

    /// A `ScoreReference` from `(mean, sd)` pairs; by default, all of `good`
    /// — Python's `reference()`.
    private func reference(
        _ entries: [String: (mean: Double, sd: Double)]? = nil,
        nHolds: Int = 42,
        builtFrom: String = "the tests' inline reference"
    ) -> ScoreReference {
        let table = entries ?? Self.good
        var stats: [String: ScoreReference.Stat] = [:]
        for (name, entry) in table {
            stats[name] = ScoreReference.Stat(mean: entry.mean, sd: entry.sd, n: nHolds)
        }
        return ScoreReference(features: stats, nHolds: nHolds, builtFrom: builtFrom)
    }

    /// The JSON a reference file holds — `payload()` in `test_score.py`.
    /// `featuresObject` replaces `features` verbatim (a malformed one), and
    /// `markers` replaces whole top-level keys (a wrong schema, version or
    /// signs marker).
    private func payloadData(
        features featuresObject: [String: Any]? = nil,
        markers: [String: Any] = [:]
    ) throws -> Data {
        var features: [String: Any] = [:]
        for (name, entry) in Self.good {
            features[name] = ["mean": entry.mean, "sd": entry.sd, "n": 42] as [String: Any]
        }
        var payload: [String: Any] = [
            "schema": Scorer.referenceSchema,
            "version": Scorer.referenceVersion,
            "signs": Scorer.referenceSigns,
            "built_from": "the tests' inline reference",
            "n_holds": 42,
            "features": featuresObject ?? features,
        ]
        for (key, value) in markers {
            payload[key] = value
        }
        return try JSONSerialization.data(withJSONObject: payload)
    }

    /// The fixture's `input` as the post-process sees it — the same helper the
    /// other parity tests use.
    private func inputFrames(from fixture: GoldenFixtures.Fixture) -> [PostProcessInputFrame] {
        fixture.input.map { frame in
            var joints: [Joint: Keypoint] = [:]
            for (name, values) in frame.joints {
                guard let joint = Joint(rawValue: name) else { continue }
                guard values.count >= 3, let x = values[0], let y = values[1],
                    let visibility = values[2]
                else { continue }
                joints[joint] = Keypoint(x: x, y: y, visibility: visibility)
            }
            return PostProcessInputFrame(
                tMs: frame.tMs,
                detected: frame.detected,
                trainerContact: frame.trainerContact,
                joints: joints
            )
        }
    }

    /// The whole on-device chain over one fixture, as the parity test runs it.
    private func scores(
        of fixture: GoldenFixtures.Fixture, against reference: ScoreReference
    ) -> [HoldScore] {
        let tMs = fixture.input.map(\.tMs)
        let trainerContact = fixture.input.map(\.trainerContact)
        let processed = PostProcess.process(inputFrames(from: fixture))
        let phases = PhaseSegmenter.classify(
            tMs: tMs, processed: processed, trainerContact: trainerContact)
        let features = Features.extract(
            tMs: tMs, processed: processed, phases: phases, trainerContact: trainerContact)
        return Scorer.scoreClip(features, reference: reference)
    }

    // MARK: - Parity against the golden fixtures of chainlink #25

    /// Every committed fixture, run through the Swift post-process, the phase
    /// segmenter, the features and the scorer, then compared with
    /// `expected.score` at `meta.tolerances` — the whole document, walked
    /// leaf by leaf: a number must be within the tolerance, a `null` must be
    /// `nil`/`NaN`, and integers, the `reason`, `missing_groups` and the
    /// top-fault names are exact because they are not quantities. A mismatch
    /// names the case, the hold, the key and both values.
    func testEveryGoldenFixtureMatchesThePythonScores() throws {
        let reference = try ScoreReference.decode(GoldenFixtures.parityReferenceData())
        let urls = GoldenFixtures.urls()
        XCTAssertGreaterThanOrEqual(urls.count, 5, "the five committed cases")

        for url in urls {
            let fixture = try GoldenFixtures.load(url)
            let name = fixture.meta.caseName
            let scores = self.scores(of: fixture, against: reference)
            let mine = scoreJSON(scores, clipHoldId: Scorer.clipScore(scores)?.holdId)
            compare(
                stored: fixture.expected.score, mine: mine,
                path: "expected.score", tolerances: fixture.meta.tolerances, label: name)
        }
    }

    /// The reference beside the fixtures: it decodes as schema v1 (or the
    /// test above would have thrown first), it is built from the synthetic
    /// holds of the five cases, and it says — in the file itself — that it is
    /// not a real one.
    func testTheParityReferenceDecodesAndIsNotARealReference() throws {
        let reference = try ScoreReference.decode(GoldenFixtures.parityReferenceData())
        XCTAssertEqual(reference.nHolds, 6, "the five cases hold six scorable holds")
        XCTAssertTrue(
            reference.builtFrom.hasPrefix("synthetic golden holds:"),
            reference.builtFrom)
        XCTAssertTrue(
            reference.builtFrom.contains("NOT a real reference (chainlink #80, #28)"),
            reference.builtFrom)

        XCTAssertEqual(Set(reference.features.keys), Set(Scorer.scoreFeatures))
        for name in Scorer.scoreFeatures {
            let stat = try XCTUnwrap(reference.features[name], "the reference has no \(name)")
            XCTAssertTrue(stat.mean.isFinite, name)
            XCTAssertGreaterThanOrEqual(stat.sd, 0.0, name)
            XCTAssertEqual(stat.n, reference.nHolds, "\(name): every hold measured it")
        }
    }

    // MARK: - The comparison the parity test makes

    /// One scorer answer as the fixture's JSON shape:
    /// `{"holds": [...], "clip_hold_id": ...}`, with `NaN` written the way
    /// Python writes it — `null`.
    private func scoreJSON(_ scores: [HoldScore], clipHoldId: Int?) -> GoldenFixtures.JSONValue {
        let holds: [GoldenFixtures.JSONValue] = scores.map { hold -> GoldenFixtures.JSONValue in
            var groups: [String: GoldenFixtures.JSONValue] = [:]
            for (name, value) in hold.groups {
                groups[name] = numberJSON(value ?? .nan)
            }
            let faults: [GoldenFixtures.JSONValue] = hold.topFaults.map { fault in
                GoldenFixtures.JSONValue.array([
                    .text(fault.feature),
                    numberJSON(fault.value),
                    numberJSON(fault.mean),
                    numberJSON(fault.z),
                ])
            }
            return .object([
                "hold_id": .number(Double(hold.holdId)),
                "hold_frames": .number(Double(hold.holdFrames)),
                "valid_frames": .number(Double(hold.validFrames)),
                "hold_start_ms": .number(Double(hold.holdStartMs)),
                "hold_end_ms": .number(Double(hold.holdEndMs)),
                "hold_duration_s": numberJSON(hold.holdDurationS),
                "score": numberJSON(hold.score ?? .nan),
                "reason": .text(hold.reason),
                "deviation": numberJSON(hold.deviation),
                "penalty": numberJSON(hold.penalty),
                "groups": .object(groups),
                "values": .object(hold.values.mapValues { numberJSON($0) }),
                "z": .object(hold.z.mapValues { numberJSON($0) }),
                "top_faults": .array(faults),
                "missing_groups": .array(
                    hold.missingGroups.map { GoldenFixtures.JSONValue.text($0) }),
            ])
        }
        return .object([
            "holds": .array(holds),
            "clip_hold_id": numberJSON(clipHoldId.map { Double($0) }),
        ])
    }

    /// One number as JSON: a non-finite one — or no number at all — is
    /// Python's `null`.
    private func numberJSON(_ value: Double?) -> GoldenFixtures.JSONValue {
        guard let value, value.isFinite else { return .null }
        return .number(value)
    }

    /// Every place `mine` differs from `stored` past the field's tolerance —
    /// the recursive walk `tests/test_golden.py` does, reporting through
    /// `XCTFail` so one mismatch does not hide the rest.
    private func compare(
        stored: GoldenFixtures.JSONValue,
        mine: GoldenFixtures.JSONValue,
        path: String,
        tolerances: [String: Double],
        label: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        switch (stored, mine) {
        case (.null, .null):
            return
        case (.null, _):
            XCTFail(
                "\(label): \(path): Python null, Swift \(describe(mine))",
                file: file, line: line)
        case (_, .null):
            XCTFail(
                "\(label): \(path): Python \(describe(stored)), Swift null",
                file: file, line: line)
        case (.text(let want), .text(let got)):
            if want != got {
                XCTFail(
                    "\(label): \(path): Python '\(want)', Swift '\(got)'",
                    file: file, line: line)
            }
        case (.bool(let want), .bool(let got)):
            if want != got {
                XCTFail(
                    "\(label): \(path): Python \(want), Swift \(got)", file: file, line: line)
            }
        case (.number(let want), .number(let got)):
            let tolerance = tolerance(for: path, in: tolerances)
            if abs(want - got) > tolerance {
                XCTFail(
                    "\(label): \(path): Python \(want), Swift \(got) "
                        + "(tolerance \(tolerance))",
                    file: file, line: line)
            }
        case (.object(let want), .object(let got)):
            let missing = want.keys.filter { got[$0] == nil }.sorted()
            let extra = got.keys.filter { want[$0] == nil }.sorted()
            if !missing.isEmpty || !extra.isEmpty {
                XCTFail(
                    "\(label): \(path): keys differ (missing=\(missing), extra=\(extra))",
                    file: file, line: line)
            }
            for (key, value) in want {
                if let other = got[key] {
                    compare(
                        stored: value, mine: other, path: "\(path).\(key)",
                        tolerances: tolerances, label: label, file: file, line: line)
                }
            }
        case (.array(let want), .array(let got)):
            if want.count != got.count {
                XCTFail(
                    "\(label): \(path): \(want.count) entries stored, \(got.count) recomputed",
                    file: file, line: line)
                return
            }
            for (index, item) in want.enumerated() {
                compare(
                    stored: item, mine: got[index], path: "\(path).\(index)",
                    tolerances: tolerances, label: label, file: file, line: line)
            }
        default:
            XCTFail(
                "\(label): \(path): Python \(describe(stored)), Swift \(describe(mine))",
                file: file, line: line)
        }
    }

    /// The tolerance that applies to `path`: the longest key of
    /// `meta.tolerances` that prefixes it, with list indices dropped first —
    /// `tolerance_for` in `tests/test_golden.py`.
    private func tolerance(for path: String, in tolerances: [String: Double]) -> Double {
        let parts = path.split(separator: ".")
            .filter { segment in !segment.allSatisfy { $0.isNumber } }
        var best: [Substring] = []
        var value = tolerances["default"] ?? 1e-6
        for (key, tolerance) in tolerances {
            let keyParts = key.split(separator: ".")
            guard keyParts.count >= best.count, parts.count >= keyParts.count else { continue }
            if Array(parts.prefix(keyParts.count)) == keyParts {
                best = keyParts
                value = tolerance
            }
        }
        return value
    }

    /// One JSON value in `null`/number/string form, for a failure message.
    private func describe(_ value: GoldenFixtures.JSONValue) -> String {
        switch value {
        case .null: return "null"
        case .bool(let flag): return "\(flag)"
        case .number(let number): return "\(number)"
        case .text(let text): return "'\(text)'"
        case .array: return "[…]"
        case .object: return "{…}"
        }
    }

    // MARK: - The constants

    func testTheGroupWeightsSumToOne() {
        let sum = Scorer.groups.reduce(0.0) { $0 + $1.weight }
        XCTAssertEqual(sum, 1.0, accuracy: 1e-12)
    }

    func testEveryScoredFeatureIsInExactlyOneGroupButTheHipSpread() {
        let covered = Scorer.groups.flatMap(\.features)
        XCTAssertEqual(Set(covered), Set(Scorer.scoreFeatures).subtracting(["hip_angle_sd"]))
        XCTAssertEqual(covered.count, Set(covered).count, "no feature counted twice")
        XCTAssertEqual(Scorer.scoreFeatures.count, 16)
        // The hip spread is scored where it is discussed: the one-sided penalty.
        XCTAssertFalse(covered.contains("hip_angle_sd"))
    }

    func testTheConstantsTheIssueFixesAreThoseValues() {
        XCTAssertEqual(Scorer.zCap, 4.0)  // Z_CAP
        XCTAssertEqual(Scorer.sdFloorL, 0.01)  // SD_FLOOR_L
        XCTAssertEqual(Scorer.sdFloorDeg, 1.0)  // SD_FLOOR_DEG
        XCTAssertEqual(Scorer.hipPenaltyWeight, 0.10)  // HIP_PENALTY_WEIGHT
        XCTAssertEqual(Scorer.minScoreFrames, 5)  // MIN_SCORE_FRAMES
        XCTAssertEqual(Scorer.topFaults, 3)  // TOP_FAULTS
        XCTAssertEqual(
            Scorer.signedByFacing,
            ["off_shoulder", "off_hip", "off_knee", "off_ankle", "body_angle"])
        XCTAssertEqual(Scorer.derived, ["com_sway_sd", "hip_angle_sd"])  // _DERIVED
        XCTAssertEqual(
            Scorer.scoreFeatures,
            [
                "off_shoulder", "off_hip", "off_knee", "off_ankle", "body_angle",
                "line_deviation", "shoulder_angle", "hip_angle", "knee_angle", "elbow_angle",
                "banana", "head", "leg_separation", "com_forward",
                "com_sway_sd", "hip_angle_sd",
            ])
        XCTAssertEqual(Scorer.referenceSchema, "handstand-reference")  // REFERENCE_SCHEMA
        XCTAssertEqual(Scorer.referenceVersion, 1)  // REFERENCE_VERSION
        XCTAssertEqual(Scorer.referenceSigns, "athlete")  // REFERENCE_SIGNS
        XCTAssertEqual(
            Scorer.groups.map(\.name),
            ["stack", "shoulder", "hip", "com", "elbows", "knees_toes", "head"])
    }

    func testTheSDFloorEachFeatureIsDividedByFollowsItsUnit() {
        for name in ["off_hip", "line_deviation", "banana", "com_forward", "com_sway_sd"] {
            XCTAssertEqual(Scorer.sdFloors[name] ?? .nan, Scorer.sdFloorL, name)
        }
        for name in ["body_angle", "shoulder_angle", "hip_angle", "knee_angle", "hip_angle_sd"] {
            XCTAssertEqual(Scorer.sdFloors[name] ?? .nan, Scorer.sdFloorDeg, name)
        }
        XCTAssertEqual(Scorer.sdFloors.count, Scorer.scoreFeatures.count)
    }

    // MARK: - Scoring one hold

    func testAHoldSittingOnEveryReferenceMeanScores100() {
        let result = Scorer.scoreHold(atMeansTable(), holdId: 0, reference: reference())

        XCTAssertEqual(result.score, 100.0)
        XCTAssertEqual(result.reason, "")
        // The spreads are built to their SDs in floating point and the table
        // is rounded like production, so these are a rounding error from zero
        // rather than bit-exactly it — within the fixture's own 1e-4.
        XCTAssertEqual(result.deviation, 0.0, accuracy: 1e-4)
        XCTAssertEqual(result.penalty, 0.0, accuracy: 1e-4)
        XCTAssertTrue(result.missingGroups.isEmpty)
        XCTAssertEqual(Set(result.groups.keys), Set(Scorer.groups.map(\.name)))
        for (_, deviation) in result.groups {
            XCTAssertEqual(deviation ?? .nan, 0.0, accuracy: 1e-4)
        }
    }

    func testOneFeatureTwoSDsOffMovesOnlyItsGroupByTheRightAmount() {
        let result = Scorer.scoreHold(
            atMeansTable(overrides: ["shoulder_angle": column(186.0, 10)]),
            holdId: 0, reference: reference())

        // shoulder_angle 186 against 178 +/- 4 is z = 2, and shoulder is the
        // whole of its group, so d_shoulder = 2 and D = 0.20 * 2 / 1.00 = 0.4.
        XCTAssertEqual(result.z["shoulder_angle"] ?? .nan, 2.0, accuracy: 1e-9)
        XCTAssertEqual(flatGroups(result.groups)["shoulder"] ?? .nan, 2.0, accuracy: 1e-9)
        for group in Scorer.groups where group.name != "shoulder" {
            XCTAssertEqual(
                flatGroups(result.groups)[group.name] ?? .nan, 0.0, accuracy: 1e-4,
                group.name)
        }
        XCTAssertEqual(result.deviation, 0.4, accuracy: 1e-4)
        XCTAssertEqual(result.penalty, 0.0, accuracy: 1e-4)
        // score = 100 * (1 - 0.4 / 4) = 90.
        XCTAssertEqual(result.score, 90.0)
        XCTAssertEqual(result.topFaults.count, Scorer.topFaults)
        let fault = result.topFaults[0]
        XCTAssertEqual(fault.feature, "shoulder_angle")
        XCTAssertEqual(fault.value, 186.0, accuracy: 1e-9)
        XCTAssertEqual(fault.mean, 178.0, accuracy: 1e-9)
        XCTAssertEqual(fault.z, 2.0, accuracy: 1e-9)
    }

    func testAZBeyondTheCapIsCappedAndTheScoreCannotGoNegative() {
        let result = Scorer.scoreHold(
            atMeansTable(overrides: ["head": column(618.0, 10)]),
            holdId: 0, reference: reference())  // 100 SDs off

        XCTAssertEqual(result.z["head"] ?? .nan, Scorer.zCap)
        XCTAssertEqual(flatGroups(result.groups)["head"] ?? .nan, Scorer.zCap, accuracy: 1e-9)
        // D = 0.05 * 4 = 0.2, so score = 100 * (1 - 0.2 / 4) = 95, not 0.
        XCTAssertEqual(result.deviation, 0.2, accuracy: 1e-4)
        XCTAssertEqual(result.score, 95.0)

        // And a hold far past the cap on *every* group stops at 0 rather than
        // going under: D of Z_CAP plus the hip penalty is past the whole scale.
        var columns: [String: [Double]] = [:]
        for name in Scorer.medianFeatures {
            columns[name] = column(1e9, 10)
        }
        columns["com_forward"] = Self.spread(1e9, 100.0)
        columns["hip_angle"] = Self.spread(1e9, 100.0)
        let hopeless = Scorer.scoreHold(
            atMeansTable(overrides: columns), holdId: 0, reference: reference())
        XCTAssertEqual(hopeless.score, 0.0)
    }

    func testTheSDFloorDividesANearZeroReferenceSd() {
        let reference = reference([
            "off_hip": (0.0, 0.001),
            "knee_angle": (179.0, 0.0),
        ])
        let result = Scorer.scoreHold(
            atMeansTable(overrides: [
                "off_hip": column(0.02, 10),
                "knee_angle": column(179.5, 10),
            ]),
            holdId: 0, reference: reference)

        // Without the floors these would be z = 20 (0.02 / 0.001) and infinite.
        XCTAssertEqual(result.z["off_hip"] ?? .nan, 0.02 / Scorer.sdFloorL, accuracy: 1e-9)
        XCTAssertEqual(
            result.z["knee_angle"] ?? .nan, 0.5 / Scorer.sdFloorDeg, accuracy: 1e-9)
        XCTAssertTrue(result.deviation.isFinite)
    }

    func testTheFacingSignFlipsTheImageSignedFeaturesBeforeScoring() {
        let filmedLeft = atMeansTable(overrides: [
            "facing_sign": column(1.0, 10),
            "off_hip": column(0.06, 10),
            "body_angle": column(6.0, 10),
        ])
        let filmedRight = atMeansTable(overrides: [
            "facing_sign": column(-1.0, 10),
            "off_hip": column(-0.06, 10),
            "body_angle": column(-6.0, 10),
        ])

        let left = Scorer.holdValues(filmedLeft, holdId: 0)
        let right = Scorer.holdValues(filmedRight, holdId: 0)
        XCTAssertEqual(left["off_hip"] ?? .nan, 0.06, accuracy: 1e-9)
        XCTAssertEqual(right["off_hip"] ?? .nan, 0.06, accuracy: 1e-9)  // -0.06 * -1: same pose
        XCTAssertEqual(left["body_angle"] ?? .nan, right["body_angle"] ?? .nan, accuracy: 1e-9)

        // The unsigned features are not touched by the flip at all.
        XCTAssertEqual(
            right["line_deviation"] ?? .nan, Self.good["line_deviation"]!.mean, accuracy: 1e-9)

        let leftResult = Scorer.scoreHold(filmedLeft, holdId: 0, reference: reference())
        let rightResult = Scorer.scoreHold(filmedRight, holdId: 0, reference: reference())
        XCTAssertEqual(leftResult.score ?? .nan, rightResult.score ?? .nan, accuracy: 1e-9)
        XCTAssertEqual(leftResult.score ?? .nan, 95.8, accuracy: 1e-9)
        XCTAssertLessThan(leftResult.score ?? 100.0, 100.0)
    }

    func testANaNFacingSignMakesTheFiveValuesNaN() {
        let table = atMeansTable(overrides: ["facing_sign": column(.nan, 10)])
        let values = Scorer.holdValues(table, holdId: 0)

        for name in Scorer.signedByFacing {
            XCTAssertTrue(values[name]?.isNaN ?? false, "\(name) is not NaN")
        }
        // The features that were already athlete-signed or unsigned are untouched.
        XCTAssertEqual(
            values["line_deviation"] ?? .nan, Self.good["line_deviation"]!.mean, accuracy: 1e-9)
        XCTAssertEqual(values["banana"] ?? .nan, Self.good["banana"]!.mean, accuracy: 1e-9)
        XCTAssertEqual(
            values["com_forward"] ?? .nan, Self.good["com_forward"]!.mean, accuracy: 1e-9)
    }

    func testTheStackGroupFallsBackToItsOtherFeatureWithNoFacingSign() {
        let result = Scorer.scoreHold(
            atMeansTable(overrides: ["facing_sign": column(.nan, 10)]),
            holdId: 0, reference: reference())

        // Every signed feature of the stack is NaN, so the group is
        // line_deviation alone.
        XCTAssertNil(result.z["off_hip"])
        XCTAssertEqual(flatGroups(result.groups)["stack"] ?? .nan, 0.0, accuracy: 1e-9)
        XCTAssertTrue(result.missingGroups.isEmpty)
        XCTAssertEqual(result.score, 100.0)
    }

    func testAGroupWhoseFeaturesAreAllUnusableGoesMissing() {
        let unusable = Set(Scorer.signedByFacing).union(["line_deviation"])
        let withoutStack = reference(Self.good.filter { !unusable.contains($0.key) })
        let result = Scorer.scoreHold(
            atMeansTable(overrides: ["facing_sign": column(.nan, 10)]),
            holdId: 0, reference: withoutStack)

        XCTAssertTrue(flatGroups(result.groups)["stack"]!.isNaN, "the stack group is missing")
        XCTAssertEqual(result.missingGroups, ["stack"])
        // The other six groups still score: the weights renormalise over them.
        XCTAssertEqual(result.score, 100.0)
    }

    func testAQuieterHipThanTheReferencePaysNoPenalty() {
        // The hip angle never moves: hip_angle_sd is 0 against a reference of 1.
        let result = Scorer.scoreHold(
            atMeansTable(overrides: ["hip_angle": column(176.0, 10)]),
            holdId: 0, reference: reference())

        XCTAssertEqual(result.z["hip_angle_sd"] ?? .nan, -1.0, accuracy: 1e-9)  // one-sided
        XCTAssertEqual(result.penalty, 0.0, accuracy: 1e-9)
        XCTAssertEqual(result.deviation, 0.0, accuracy: 1e-4)
        XCTAssertEqual(result.score, 100.0)
    }

    func testABusierHipThanTheReferencePaysThePenalty() {
        // hip_angle_sd of 2 against a reference of 1 +/- 0.5 is z = 1 (the
        // angle floor is 1 degree), so P = 0.10 * 1 and the score is
        // 100 * (1 - 0.1/4).
        let result = Scorer.scoreHold(
            atMeansTable(overrides: ["hip_angle": Self.spread(176.0, 2.0)]),
            holdId: 0, reference: reference())

        XCTAssertEqual(result.penalty, Scorer.hipPenaltyWeight, accuracy: 1e-4)
        XCTAssertEqual(result.deviation, 0.0, accuracy: 1e-4)
        XCTAssertEqual(result.score, 97.5)
    }

    func testMissingGroupsRenormaliseTheWeights() {
        let reference = reference([
            "shoulder_angle": Self.good["shoulder_angle"]!,
            "head": Self.good["head"]!,
        ])
        let result = Scorer.scoreHold(
            atMeansTable(overrides: ["shoulder_angle": column(186.0, 10)]),
            holdId: 0, reference: reference)

        // Only shoulder and head can be compared: D = (0.20 * 2 + 0.05 * 0) / 0.25.
        XCTAssertEqual(
            result.missingGroups, ["stack", "hip", "com", "elbows", "knees_toes"])
        XCTAssertEqual(result.deviation, 1.6, accuracy: 1e-9)
        XCTAssertEqual(result.score, 60.0)
    }

    func testAHoldWithNothingToCompareHasNoScore() {
        let result = Scorer.scoreHold(atMeansTable(), holdId: 0, reference: reference([:]))

        XCTAssertNil(result.score)
        XCTAssertEqual(result.reason, "nothing to compare")
        XCTAssertEqual(result.deviation, 0.0)
        for group in Scorer.groups {
            XCTAssertTrue(
                flatGroups(result.groups)[group.name]!.isNaN, "\(group.name) is missing")
        }
        XCTAssertEqual(result.missingGroups, Scorer.groups.map(\.name))
        XCTAssertTrue(result.topFaults.isEmpty)
    }

    func testAHoldWithTooFewValidFramesIsNotScored() {
        let result = Scorer.scoreHold(
            atMeansTable(frames: 4), holdId: 0, reference: reference())

        XCTAssertNil(result.score)
        XCTAssertEqual(result.reason, "too few valid frames")
        XCTAssertEqual(result.holdFrames, 4)
        XCTAssertEqual(result.validFrames, 4)
        XCTAssertTrue(result.values.isEmpty)
        XCTAssertTrue(result.z.isEmpty)

        // Invalid frames count for the hold's length but not for its score.
        let sparsely = Scorer.scoreHold(
            atMeansTable(frames: 10, valid: flags(true, 4) + flags(false, 6)),
            holdId: 0, reference: reference())
        XCTAssertNil(sparsely.score)
        XCTAssertEqual(sparsely.reason, "too few valid frames")
        XCTAssertEqual(sparsely.holdFrames, 10)
        XCTAssertEqual(sparsely.validFrames, 4)
    }

    func testInvalidFramesAndFramesOutsideTheHoldAreIgnored() {
        let table = frameTable(
            frames: 16,
            holdId: ints(0, 12) + ints(-1, 4),
            valid: flags(true, 10) + flags(false, 2) + flags(true, 4),
            overrides: [
                // Junk on exactly the frames that must not be read.
                "off_hip": column(0.0, 10) + column(999.0, 6),
                "com_forward": Self.spread(
                    Self.good["com_forward"]!.mean, Self.good["com_sway_sd"]!.mean)
                    + column(50.0, 6),
                "hip_angle": Self.spread(
                    Self.good["hip_angle"]!.mean, Self.good["hip_angle_sd"]!.mean)
                    + column(176.0, 6),
            ])
        let values = Scorer.holdValues(table, holdId: 0)

        // The invalid frames' 999 never enters.
        XCTAssertEqual(values["off_hip"] ?? .nan, 0.0, accuracy: 1e-9)
        XCTAssertEqual(
            values["com_sway_sd"] ?? .nan, Self.good["com_sway_sd"]!.mean, accuracy: 1e-4)
        XCTAssertEqual(
            values["hip_angle_sd"] ?? .nan, Self.good["hip_angle_sd"]!.mean, accuracy: 1e-3)

        let result = Scorer.scoreHold(table, holdId: 0, reference: reference())
        XCTAssertEqual(result.holdFrames, 12)  // both invalid frames belong to the hold
        XCTAssertEqual(result.validFrames, 10)
        XCTAssertEqual(result.holdStartMs, 0)
        XCTAssertEqual(result.holdEndMs, 11 * 33)
        XCTAssertEqual(result.holdDurationS, Double(11 * 33) / 1000.0, accuracy: 1e-9)
        XCTAssertEqual(result.score, 100.0)

        // The hold_id == -1 frames are not a hold: they are not scored at all.
        XCTAssertEqual(Scorer.scoreClip(table, reference: reference()).map(\.holdId), [0])
    }

    func testTheSpreadOfAHoldIsThePopulationSDOfItsFiniteValues() {
        let two = frameTable(
            frames: 2, overrides: ["com_forward": [0.0, 0.1], "hip_angle": [170.0, 180.0]])
        let values = Scorer.holdValues(two, holdId: 0)

        // ddof = 0: (0.05^2 + 0.05^2) / 2 = 0.05^2, not the sample's 0.0707.
        XCTAssertEqual(values["com_sway_sd"] ?? .nan, 0.05, accuracy: 1e-9)
        XCTAssertEqual(values["hip_angle_sd"] ?? .nan, 5.0, accuracy: 1e-9)

        // One finite value is not a spread, and NaN is never counted.
        let one = frameTable(frames: 3, overrides: ["com_forward": [0.0, 0.1, .nan]])
        XCTAssertEqual(
            Scorer.holdValues(one, holdId: 0)["com_sway_sd"] ?? .nan, 0.05, accuracy: 1e-9)
        let lonely = frameTable(frames: 2, overrides: ["com_forward": [0.0, .nan]])
        XCTAssertTrue(Scorer.holdValues(lonely, holdId: 0)["com_sway_sd"]!.isNaN)
    }

    func testTheHoldFieldsAreTheOnesTheHoldSummaryWrites() {
        let table = atMeansTable(
            frames: 8, holdId: ints(0, 8), valid: flags(true, 6) + flags(false, 2))
        let result = Scorer.scoreHold(table, holdId: 0, reference: reference())

        XCTAssertEqual(result.holdFrames, 8)
        XCTAssertEqual(result.validFrames, 6)
        XCTAssertEqual(result.holdStartMs, 0)
        XCTAssertEqual(result.holdEndMs, 7 * 33)
        XCTAssertEqual(result.holdDurationS, Double(7 * 33) / 1000.0, accuracy: 1e-9)
    }

    // MARK: - clip_score

    func testTheClipsScoreIsItsLongestScoredHold() {
        let reference = reference()
        let longest = Scorer.scoreClip(
            atMeansTable(frames: 16, holdId: ints(0, 6) + ints(1, 10)),
            reference: reference)
        XCTAssertEqual(longest.map(\.holdId), [0, 1])
        XCTAssertEqual(Scorer.clipScore(longest)?.holdId, 1, "9 frames span beats 5")

        // A longer hold nobody could score does not represent the clip.
        let sparse = Scorer.scoreClip(
            atMeansTable(
                frames: 16, holdId: ints(0, 6) + ints(1, 10),
                valid: flags(true, 6) + flags(false, 10)),
            reference: reference)
        XCTAssertEqual(Scorer.clipScore(sparse)?.holdId, 0)

        // Ties go to the earlier hold, like features.longestHoldId.
        let tied = Scorer.scoreClip(
            atMeansTable(frames: 20, holdId: ints(0, 10) + ints(1, 10)),
            reference: reference)
        XCTAssertEqual(Scorer.clipScore(tied)?.holdId, 0)

        // Nothing scored is no answer, not a zero.
        let none = Scorer.scoreClip(
            atMeansTable(
                frames: 8, holdId: ints(0, 4) + ints(1, 4), valid: flags(false, 8)),
            reference: reference)
        XCTAssertNil(Scorer.clipScore(none))
        XCTAssertNil(Scorer.clipScore([]))
    }

    // MARK: - The reference file

    func testAReferenceFileRoundTrips() throws {
        let loaded = try ScoreReference.decode(payloadData())

        XCTAssertEqual(loaded.nHolds, 42)
        XCTAssertEqual(loaded.builtFrom, "the tests' inline reference")
        XCTAssertEqual(
            loaded.features["hip_angle"],
            ScoreReference.Stat(mean: 176.0, sd: 3.0, n: 42))
        XCTAssertEqual(Set(loaded.features.keys), Set(Self.good.keys))
    }

    func testAReferenceWithoutEveryFeatureIsFine() throws {
        let loaded = try ScoreReference.decode(
            payloadData(features: ["head": ["mean": 18.0, "sd": 6.0, "n": 42] as [String: Any]]))

        XCTAssertEqual(Set(loaded.features.keys), ["head"])
        // ... and the hold is then only scored where the reference has
        // something to say.
        let result = Scorer.scoreHold(atMeansTable(), holdId: 0, reference: loaded)
        XCTAssertEqual(flatGroups(result.groups)["head"] ?? .nan, 0.0, accuracy: 1e-9)
        XCTAssertTrue(flatGroups(result.groups)["stack"]!.isNaN)
    }

    func testTheReferenceMarkersAreRefusedWhenTheyAreWrong() throws {
        func message(_ markers: [String: Any]) throws -> String {
            do {
                _ = try ScoreReference.decode(payloadData(markers: markers))
            } catch let error as ScoreReferenceError {
                return "\(error)"
            }
            throw FixtureFailure("a wrong marker was accepted: \(markers)")
        }

        XCTAssertTrue(try message(["schema": "something-else"]).contains("schema"), "schema")
        XCTAssertTrue(try message(["version": 2]).contains("version"), "version")
        XCTAssertTrue(try message(["signs": "image"]).contains("signs"), "signs")
        XCTAssertTrue(try message(["built_from": 42]).contains("free text"), "built_from")
    }

    func testAReferenceWithABadNumberIsRefused() throws {
        func message(_ features: [String: Any]) throws -> String {
            do {
                _ = try ScoreReference.decode(payloadData(features: features))
            } catch let error as ScoreReferenceError {
                return "\(error)"
            }
            throw FixtureFailure("a bad number was accepted: \(features)")
        }

        let negative: [String: Any] = [
            "hip_angle": ["mean": 176.2, "sd": -1.0, "n": 42] as [String: Any],
        ]
        XCTAssertTrue(try message(negative).contains("sd must be >= 0"), "negative sd")

        let notANumber: [String: Any] = [
            "hip_angle": ["mean": "176.2", "sd": 3.1, "n": 42] as [String: Any],
        ]
        XCTAssertTrue(try message(notANumber).contains("finite"), "mean is not a number")

        let noEntry: [String: Any] = [
            "hip_angle": ["mean": 176.2, "sd": 3.1] as [String: Any],
        ]
        XCTAssertTrue(try message(noEntry).contains("features['hip_angle'].n"), "n is missing")

        let fractional: [String: Any] = [
            "hip_angle": ["mean": 176.2, "sd": 3.1, "n": 4.5] as [String: Any],
        ]
        XCTAssertTrue(try message(fractional).contains("non-negative integer"), "n is 4.5")
    }

    func testAReferenceWithAFeatureWeDoNotScoreIsRefused() throws {
        let features: [String: Any] = [
            "hand_width": ["mean": 1.0, "sd": 0.1, "n": 42] as [String: Any],
        ]
        do {
            _ = try ScoreReference.decode(payloadData(features: features))
            XCTFail("hand_width is not a scored feature")
        } catch let error as ScoreReferenceError {
            XCTAssertTrue("\(error)".contains("not a scored feature"), "\(error)")
        }
    }

    func testAReferenceThatIsNotAnObjectOrNotJsonIsRefused() throws {
        let array = try JSONSerialization.data(withJSONObject: [1, 2])
        XCTAssertThrowsError(try ScoreReference.decode(array)) { error in
            guard let error = error as? ScoreReferenceError else {
                return XCTFail("expected ScoreReferenceError, got \(error)")
            }
            XCTAssertTrue("\(error)".contains("JSON object"), "\(error)")
        }

        do {
            _ = try ScoreReference.decode(Data("{not json".utf8))
            XCTFail("bad JSON was accepted")
        } catch let error as ScoreReferenceError {
            guard case .invalidJSON(let message) = error else {
                return XCTFail("expected .invalidJSON, got \(error)")
            }
            XCTAssertTrue(message.contains("not valid JSON"), message)
        }
    }

    func testAReferenceWithANaNIsRefused() throws {
        // JSON has no NaN, so this one is written the way Python's json.dumps
        // writes it: a bare NaN token. Whatever the parser makes of it, the
        // reference must not come back from it.
        let data = Data("""
            {"schema": "handstand-reference", "version": 1, "signs": "athlete",
             "built_from": "x", "n_holds": 1,
             "features": {"hip_angle": {"mean": NaN, "sd": 3.1, "n": 1}}}
            """.utf8)
        XCTAssertThrowsError(try ScoreReference.decode(data)) { error in
            XCTAssertTrue(error is ScoreReferenceError, "\(error)")
        }
    }

    // MARK: - tableRounded

    /// The rounding Python's `feature_table` applies before the scorer reads
    /// the table, on the half-way values where the rule shows: **half to
    /// even** on the binary value (`0.125` → `0.12`, `0.375` → `0.38`), five
    /// decimals on lengths, two on angles, none on the sign, `NaN` staying
    /// `NaN`. These are numpy's answers (`np.round`), which is what
    /// `feature_table` calls.
    func testTableRoundedMatchesPythonsFeatureTableOnHalfWayValues() {
        let frames = 6
        var values: [String: [Double]] = [:]
        // Degrees round to two digits: 12.5, 37.5, 62.5, 87.5, 6.25, 106.25.
        values["hip_angle"] = [0.125, 0.375, 0.625, 0.875, 0.0625, 1.0625]
        // Lengths round to five: 0.5 and 100000.5 are half-way, 1.5625 and
        // 15625.0 are not, and NaN is not a number to round.
        values["com_forward"] = [5e-6, 1.000005, 1.5625e-05, 0.15625, .nan, 0.1]
        values["com_u"] = [5e-6, 1.000005, 1.5625e-05, 0.15625, .nan, 0.1]
        // The facing sign rounds to a whole number: 0.5 → 0, -0.5 → -0,
        // 1.5 → 2, 2.5 → 2.
        values["facing_sign"] = [0.5, -0.5, 1.5, 2.5, 1.0, -1.0]

        let table = ClipFeatures(
            tMs: (0..<frames).map { $0 * 33 },
            phase: [Phase](repeating: .hold, count: frames),
            holdId: [Int](repeating: 0, count: frames),
            values: values,
            balanceZone: [BalanceZone?](repeating: nil, count: frames),
            comComplete: flags(true, frames),
            valid: flags(true, frames),
            unusableReason: ""
        )
        let rounded = table.tableRounded()

        XCTAssertEqual(rounded.value("hip_angle"), [0.12, 0.38, 0.62, 0.88, 0.06, 1.06])
        // NaN does not equal NaN, so the length columns are compared around it.
        let comForward = rounded.value("com_forward")
        XCTAssertEqual(
            Array(comForward.prefix(4)), [0.0, 1.0, 0.00002, 0.15625], "com_forward")
        XCTAssertTrue(comForward[4].isNaN, "com_forward: NaN must stay NaN")
        XCTAssertEqual(comForward[5], 0.1, "com_forward")
        let comU = rounded.value("com_u")
        XCTAssertEqual(Array(comU.prefix(4)), [0.0, 1.0, 0.00002, 0.15625], "com_u")
        XCTAssertTrue(comU[4].isNaN, "com_u: NaN must stay NaN")
        XCTAssertEqual(comU[5], 0.1, "com_u")
        XCTAssertEqual(rounded.value("facing_sign"), [0.0, -0.0, 2.0, 2.0, 1.0, -1.0])
        // Nothing else about the clip changed.
        XCTAssertEqual(rounded.valid, table.valid)
        XCTAssertEqual(rounded.holdId, table.holdId)
        XCTAssertEqual(rounded.tMs, table.tMs)
    }

    /// A fixture-shaped error, for "this must not have decoded" paths.
    private struct FixtureFailure: Error, CustomStringConvertible {
        let message: String
        var description: String { message }
        init(_ message: String) { self.message = message }
    }
}
