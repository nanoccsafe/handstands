import XCTest

@testable import HandstandCore

/// The Swift mirror of the feature port (chainlink #81): the golden fixtures
/// of chainlink #25 run through `PostProcess.process`,
/// `PhaseSegmenter.classify` and `Features.extract`, compared with
/// `expected.features` and `expected.hold_summary` at `meta.tolerances`, plus
/// the core of `pipeline/tests/test_features.py` on poses written by hand in
/// the body frame — the same cases, the same expectations.
///
/// Nothing here reads a real video: the fixtures are the synthetic cases
/// `pipeline/handstand/golden.py` writes, and every other input is a handful
/// of numbers built in the test itself, in the **body frame** the way the
/// Python tests write it — the wrist midpoint is the origin, `v` is up, both
/// in body lengths, and `length` is the known pixel scale that turns a pose
/// back into one.
final class FeaturesTests: XCTestCase {
    // MARK: - A synthetic body, as `pipeline/tests/test_features.py` builds it

    /// The clip's body length in pixels — Python's `LENGTH`. 300 keeps the
    /// coordinates in a comfortable range and makes 0.1 L exactly 30 px.
    private static let length = 300.0
    /// The frame interval of the synthetic clips, in milliseconds — Python's
    /// `FRAME_MS`.
    private static let frameMs = 33
    /// Where the wrist midpoint sits in the display frame — `to_pixels`'s
    /// default origin.
    private static let wristX = 200.0
    private static let wristY = 500.0
    /// How far above the ankle the toes sit in the synthetic pose — Python's
    /// `FOOT_BEYOND`, which is what makes a frame built from this pose a
    /// *complete* one for the centre of mass to weigh.
    private static let footBeyond = 0.1

    /// One frame of a synthetic athlete, in the body frame — Python's `Pose`.
    ///
    /// Every number is a body length with the wrist midpoint at `(0, 0)` and
    /// `v` up, so a held handstand is a column of stations up the image and a
    /// failing assertion says which station moved. The two sides of a part are
    /// the station plus and minus that part's `spread`, which is what a
    /// straddle widens and a side view leaves at zero.
    private struct Pose: Sendable {
        /// The height of each station up the body: ankles highest, knees
        /// between the hips and them.
        var vShoulder = 0.39
        var vHip = 0.75
        var vKnee = 0.95
        var vAnkle = 1.15
        /// The horizontal offset of each station from the vertical through the
        /// hands: the stacking errors, 0 for a line.
        var uShoulder = 0.0
        var uHip = 0.0
        var uKnee = 0.0
        var uAnkle = 0.0
        /// The nose: a handstand's head hangs *past* the shoulders towards the
        /// floor, so it is below the shoulder line and a little in front of
        /// the chest.
        var uNose = 0.04
        var vNose = 0.30
        /// The elbow of a straight arm, halfway between the shoulder and the
        /// hands, which are the origin.
        var uElbow = 0.0
        var vElbow = 0.195
        /// How far apart the two sides of a part are, in u, per part. Empty is
        /// a side view; `["ankle": 0.5]` is a straddle.
        var spread: [String: Double] = [:]
        /// How wide the hands and the shoulders are. Both 0 in a side view.
        var handSpan = 0.0
        var shoulderSpan = 0.0
        /// Joint names to leave unseen, for the tests about one side missing.
        var hidden: Set<String> = []
        /// False for a frame the model saw nobody in.
        var visible = true

        /// Every joint's position by name, both sides, the nose and the toes —
        /// Python's `Pose.stations`.
        func stations() -> [String: (u: Double, v: Double)] {
            var points: [String: (u: Double, v: Double)] = [:]
            points["nose"] = (uNose, vNose)
            let centres: [(part: String, u: Double, v: Double)] = [
                ("shoulder", uShoulder, vShoulder),
                ("elbow", uElbow, vElbow),
                ("hip", uHip, vHip),
                ("knee", uKnee, vKnee),
                ("ankle", uAnkle, vAnkle),
            ]
            for centre in centres {
                let half = (spread[centre.part] ?? 0.0) / 2.0
                points["left_\(centre.part)"] = (centre.u - half, centre.v)
                points["right_\(centre.part)"] = (centre.u + half, centre.v)
            }
            // The wrists are the origin by definition, so their spread is the
            // only thing about them, and only hand_width reads it.
            points["left_wrist"] = (-handSpan / 2.0, 0.0)
            points["right_wrist"] = (handSpan / 2.0, 0.0)
            let halfShoulders = shoulderSpan / 2.0
            points["left_shoulder"] = (uShoulder - halfShoulders, vShoulder)
            points["right_shoulder"] = (uShoulder + halfShoulders, vShoulder)
            // The toes carry the leg on past the ankle, taking the ankle's own
            // place — spread and all — with them.
            for side in ["left", "right"] {
                let ankle = points["\(side)_ankle"] ?? (0.0, 0.0)
                points["\(side)_foot_index"] = (ankle.u, ankle.v + FeaturesTests.footBeyond)
            }
            return points
        }

        /// This pose seen from the other side: every `u` negated. The athlete
        /// still faces the same way *anatomically*, so a shape feature must
        /// read the same number on both and a directional one must flip.
        func mirrored() -> Pose {
            var out = self
            out.uShoulder = -uShoulder
            out.uHip = -uHip
            out.uKnee = -uKnee
            out.uAnkle = -uAnkle
            out.uNose = -uNose
            out.uElbow = -uElbow
            out.spread = spread.mapValues { -$0 }
            out.handSpan = -handSpan
            return out
        }
    }

    /// A held line handstand — Python's `LINE`.
    private static let line = Pose()
    /// A pike: the hips above the shoulders and the legs folded back down
    /// towards the hands — Python's `PIKE`.
    private static let pike = Pose(
        vHip: 0.95, vKnee: 0.55, vAnkle: 0.40, vNose: 0.26, vElbow: 0.195)
    /// A banana: the hips off the shoulder→ankle line — Python's `BANANA`.
    private static let banana = Pose(uHip: 0.08, uKnee: 0.03)
    /// A bent arm: the elbow forward of the shoulder→wrist line — Python's
    /// `BENT_ARMS`.
    private static let bentArms = Pose(uElbow: 0.15, vElbow: 0.30)
    /// A straddle: the ankles a leg's width apart, the hips where they were —
    /// Python's `STRADDLE`.
    private static let straddle = Pose(spread: ["ankle": 0.5])
    /// A front-view handstand, the only geometry in which the two widths
    /// `hand_width` calls a ratio can be read — Python's `FRONTAL`.
    private static let frontal = Pose(handSpan: 0.24, shoulderSpan: 0.24)

    /// A pose list as the body-frame track the geometry reads — Python's
    /// `track_of`.
    private func trackOf(_ poses: [Pose]) -> BodyTrack {
        let joints = Features.wantedJoints
        var uv: [[BodyPoint]] = []
        for pose in poses {
            let stations = pose.stations()
            var row: [BodyPoint] = []
            for joint in joints {
                if let point = stations[joint.rawValue], pose.visible {
                    row.append(BodyPoint(u: point.u, v: point.v))
                } else {
                    row.append(.nan)
                }
            }
            for name in pose.hidden {
                if let column = joints.firstIndex(where: { $0.rawValue == name }) {
                    row[column] = .nan
                }
            }
            uv.append(row)
        }
        return BodyTrack(uv: uv, joints: joints)
    }

    /// The features of a pose list, with no file in the way — Python's
    /// `measure`.
    private func measure(_ poses: [Pose]) throws -> BodyFeatures {
        try Features.bodyFeatures(trackOf(poses))
    }

    /// One feature of a one-frame pose list — Python's `value`.
    private func value(_ poses: [Pose], _ name: String) throws -> Double {
        try measure(poses).value(name)[0]
    }

    /// The processed frames a pose list reads back as — Python's `_arrays`
    /// over `pm.JOINT_NAMES`: every station carries its pixel position, and
    /// only a visible, unhidden joint is `valid`.
    private func processedFrames(_ poses: [Pose]) -> [ProcessedFrame] {
        poses.map { pose in
            var joints: [Joint: Point2] = [:]
            var valid: Set<Joint> = []
            for (name, point) in pose.stations() {
                guard let joint = Joint(rawValue: name) else { continue }
                joints[joint] = Point2(
                    x: FeaturesTests.wristX + point.u * FeaturesTests.length,
                    y: FeaturesTests.wristY - point.v * FeaturesTests.length
                )
                if pose.visible && !pose.hidden.contains(name) {
                    valid.insert(joint)
                }
            }
            return ProcessedFrame(joints: joints, valid: valid, filled: [])
        }
    }

    /// The body-frame track a processed clip of these poses reads back as —
    /// Python's `track_from_pixels`.
    private func trackFromPixels(
        _ poses: [Pose],
        wristX: Double = FeaturesTests.wristX,
        wristY: Double = FeaturesTests.wristY,
        length: Double = FeaturesTests.length
    ) -> BodyTrack {
        let order = Joint.allCases
        let rows = poses.count
        var x = [[Double]](repeating: [Double](repeating: .nan, count: order.count), count: rows)
        var y = [[Double]](repeating: [Double](repeating: .nan, count: order.count), count: rows)
        var valid = [[Bool]](repeating: [Bool](repeating: false, count: order.count), count: rows)
        for (index, pose) in poses.enumerated() {
            for (name, point) in pose.stations() {
                guard let joint = Joint(rawValue: name),
                    let column = order.firstIndex(of: joint)
                else { continue }
                x[index][column] = wristX + point.u * length
                y[index][column] = wristY - point.v * length
                valid[index][column] = pose.visible && !pose.hidden.contains(name)
            }
        }
        return Features.bodyFrameTrack(
            x: x, y: y, valid: valid, joints: order, bodyLength: length)
    }

    /// The `(phase, hold_id)` labels for a label per frame — Python's
    /// `phases_for`. Holds are numbered from 0 in time order, so a stretch of
    /// hold frames is a different hold from the one before it.
    private func phasesFor(_ labels: [String]) -> (phase: [Phase], holdId: [Int]) {
        var numbers: [Int] = []
        var current = PhaseSegmenter.noHold
        for label in labels {
            guard label == "hold" else {
                numbers.append(PhaseSegmenter.noHold)
                continue
            }
            if numbers.isEmpty || numbers[numbers.count - 1] == PhaseSegmenter.noHold {
                current += 1
            }
            numbers.append(current)
        }
        return (labels.map { Phase(rawValue: $0) ?? .unknown }, numbers)
    }

    /// The sidecar `ProcessedClip` carries: a usable one whose parts add up to
    /// `length`, or — for `NaN` — a reason Python would refuse.
    private func sidecar(_ length: Double) -> BodyLength {
        if length.isNaN {
            return BodyLength(
                reason: "torso was measurable on 3 frame(s), need 10",
                torsoPx: nil, thighPx: nil, shinPx: nil,
                frames: ["torso": 3, "thigh": 3, "shin": 3])
        }
        // The parts only have to add up to `length` exactly — the features
        // never read them, and `totalPx` is the scale Python is handed. A
        // scale of 0.0 is a *usable* sidecar with a zero total, which is the
        // other way `extract_clip` refuses a clip.
        return BodyLength(
            reason: nil, torsoPx: length, thighPx: 0, shinPx: 0, frames: [:])
    }

    /// Measure a pose list straight, with no file in the way — Python's
    /// `extract`.
    private func extract(
        _ poses: [Pose],
        labels: [String]? = nil,
        bodyLength: Double = FeaturesTests.length,
        trainerContact: [Bool]? = nil
    ) throws -> ClipFeatures {
        let frames = poses.count
        let tMs = (0..<frames).map { $0 * FeaturesTests.frameMs }
        let names = labels ?? [String](repeating: "unknown", count: frames)
        let (phase, holdId) = phasesFor(names)
        let clipPhases = ClipPhases(
            tMs: tMs,
            phase: phase,
            holdId: holdId,
            signals: PhaseSegmenter.unusableSignals(frames: frames, reason: ""),
            runs: [],
            usable: true,
            unusableReason: ""
        )
        return Features.extract(
            tMs: tMs,
            processed: ProcessedClip(
                frames: processedFrames(poses), bodyLength: sidecar(bodyLength)),
            phases: clipPhases,
            trainerContact: trainerContact
                ?? [Bool](repeating: false, count: frames)
        )
    }

    /// A one-hold clip whose `com_forward` is exactly `forward`, with real
    /// timestamps — Python's `sway_clip`. Every other measurement is `NaN`,
    /// as it is on a frame nobody could see, and every frame is in the hold.
    private func swayClip(tMs: [Int], forward: [Double]) -> ClipFeatures {
        let frames = tMs.count
        var values: [String: [Double]] = [:]
        for name in Features.featureNames + Features.comColumns {
            values[name] = [Double](repeating: .nan, count: frames)
        }
        values["com_forward"] = forward
        return ClipFeatures(
            tMs: tMs,
            phase: [Phase](repeating: .hold, count: frames),
            holdId: [Int](repeating: 0, count: frames),
            values: values,
            balanceZone: [BalanceZone?](repeating: nil, count: frames),
            comComplete: [Bool](repeating: true, count: frames),
            valid: [Bool](repeating: true, count: frames),
            unusableReason: ""
        )
    }

    /// A pose whose knees and ankles are `amount` away from the hands —
    /// Python's `shifted_forward`. The legs swing and the torso stays over
    /// the wrists, so the facing sign's reference stays put while the CoM goes
    /// as far as the leg mass share can carry it.
    private func shiftedForward(_ amount: Double) -> Pose {
        var pose = FeaturesTests.line
        pose.uKnee = amount
        pose.uAnkle = amount
        return pose
    }

    /// `count` copies of one pose — Python's `[pose] * count`.
    private func repeated(_ pose: Pose, _ count: Int) -> [Pose] {
        [Pose](repeating: pose, count: count)
    }

    // MARK: - Parity against the golden fixtures of chainlink #25

    /// Every committed fixture, decoded, run through the Swift post-process,
    /// the phase segmenter and `Features.extract`, then compared with
    /// `expected.features` (per frame, per column) and `expected.hold_summary`
    /// (per hold) at `meta.tolerances` — by the shared
    /// `GoldenComparison.features` and `GoldenComparison.holdSummary`: a
    /// number must be within the tolerance and a `null` must be a `NaN` — and
    /// `balance_zone`, the hold's integer fields and the two identity columns
    /// are compared exactly, because a category is not a quantity. The
    /// comparison itself lives in `GoldenComparison.swift`, shared with
    /// `AnalyzerTests` and `RealParityTests` (chainlink #42).
    func testEveryGoldenFixtureMatchesThePythonFeatures() throws {
        let urls = GoldenFixtures.urls()
        XCTAssertGreaterThanOrEqual(urls.count, 5, "the five committed cases")

        for url in urls {
            let fixture = try GoldenFixtures.load(url)
            let tMs = fixture.input.map(\.tMs)
            let trainerContact = fixture.input.map(\.trainerContact)
            let processed = PostProcess.process(GoldenFixtures.inputFrames(from: fixture))
            let phases = PhaseSegmenter.classify(
                tMs: tMs, processed: processed, trainerContact: trainerContact)
            let features = Features.extract(
                tMs: tMs, processed: processed, phases: phases, trainerContact: trainerContact)
            for mismatch in GoldenComparison.features(features, fixture: fixture) {
                XCTFail(mismatch)
            }
            for mismatch in GoldenComparison.holdSummary(
                Features.holdRows(features), fixture: fixture)
            {
                XCTFail(mismatch)
            }
        }
    }

    // MARK: - The constants

    /// Every constant of `pipeline/handstand/features.py` at its Python value.
    /// The centre-of-mass constants are pinned in `CentreOfMassTests`.
    func testTheConstantsAreThePythonOnes() {
        XCTAssertEqual(Features.minSegmentL, 1e-9)  // MIN_SEGMENT_L
        XCTAssertEqual(Features.lineToleranceL, 0.1)  // LINE_TOLERANCE_L
        XCTAssertEqual(Features.bodyAngleTolDeg, 10.0)  // BODY_ANGLE_TOL_DEG
        XCTAssertEqual(Features.pikeHipDeg, 165.0)  // PIKE_HIP_DEG
        XCTAssertEqual(Features.openShoulderDeg, 160.0)  // OPEN_SHOULDER_DEG
        XCTAssertEqual(Features.straightKneeDeg, 165.0)  // STRAIGHT_KNEE_DEG
        XCTAssertEqual(Features.bentElbowDeg, 160.0)  // BENT_ELBOW_DEG
        XCTAssertEqual(Features.bananaToleranceL, 0.1)  // BANANA_TOLERANCE_L
        XCTAssertEqual(Features.headFlexionDeg, 60.0)  // HEAD_FLEXION_DEG
        XCTAssertEqual(Features.splitDeg, 30.0)  // SPLIT_DEG
        XCTAssertEqual(Features.lengthDigits, 5)  // _LENGTH_DIGITS
        XCTAssertEqual(Features.angleDigits, 2)  // _ANGLE_DIGITS

        XCTAssertEqual(Features.sides, ["left", "right"])  // SIDES
        XCTAssertEqual(
            Features.bodyParts, ["wrist", "elbow", "shoulder", "hip", "knee", "ankle"])
        XCTAssertEqual(
            Features.wantedJoints.map(\.rawValue),
            [
                "left_wrist", "right_wrist", "left_elbow", "right_elbow",
                "left_shoulder", "right_shoulder", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle", "nose",
            ])

        // FEATURES: the same 14 names, in the same order, with the units,
        // targets and rounding Python writes them with.
        XCTAssertEqual(
            Features.featureNames,
            [
                "off_shoulder", "off_hip", "off_knee", "off_ankle", "line_deviation",
                "body_angle", "shoulder_angle", "hip_angle", "knee_angle", "elbow_angle",
                "banana", "head", "leg_separation", "hand_width",
            ])
        XCTAssertEqual(Features.all.map(\.name), Features.featureNames)
        XCTAssertEqual(
            Features.all.map(\.unit),
            ["L", "L", "L", "L", "L", "deg", "deg", "deg", "deg", "deg", "L", "deg", "deg",
                "ratio"])
        XCTAssertEqual(
            Features.all.map(\.target),
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 180.0, 180.0, 180.0, 180.0, 0.0, 0.0, 0.0,
                1.0])
        XCTAssertEqual(
            Features.all.map(\.digits),
            [5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 5, 2, 2, 5])

        XCTAssertEqual(Features.comColumns, ["com_u", "com_v", "facing_sign", "com_forward"])
        XCTAssertEqual(
            Features.holdStabilityNames,
            [
                "com_forward_median", "com_sway_sd", "com_sway_range", "com_speed_rms",
                "hip_angle_sd", "shoulder_angle_sd", "pct_over", "pct_under",
            ])
        XCTAssertEqual(
            Features.balanceStats.map(\.digits), [5, 5, 5, 5, 2, 2, 2, 2])

        // HOLD_SUMMARY_COLUMNS, minus the two columns Swift does not carry.
        let fields: Set<String> = [
            "hold_id", "hold_frames", "valid_frames", "hold_start_ms", "hold_end_ms",
            "hold_duration_s",
        ]
        XCTAssertEqual(
            Set(Features.holdSummaryColumns),
            fields.union(Features.featureNames.flatMap { ["\($0)_median", "\($0)_iqr"] })
                .union(Features.holdStabilityNames))
        XCTAssertEqual(Features.holdSummaryColumns.count, 6 + 28 + 8)
    }

    // MARK: - The features of a line handstand

    /// `test_a_line_handstand_reads_as_one`: a straight vertical body is a
    /// zero line and four straight angles.
    func testALineHandstandReadsAsOne() throws {
        let features = try measure([FeaturesTests.line])

        XCTAssertTrue(features.valid[0])
        for name in ["off_shoulder", "off_hip", "off_knee", "off_ankle"] {
            XCTAssertEqual(features.value(name)[0], 0.0, accuracy: 1e-9, name)
        }
        XCTAssertEqual(features.value("line_deviation")[0], 0.0, accuracy: 1e-9)
        XCTAssertEqual(features.value("body_angle")[0], 0.0, accuracy: 1e-6)
        for name in ["shoulder_angle", "hip_angle", "knee_angle", "elbow_angle"] {
            XCTAssertEqual(features.value(name)[0], 180.0, accuracy: 0.5, name)
        }
        XCTAssertEqual(features.value("banana")[0], 0.0, accuracy: 1e-9)
        // The head hangs 24 degrees off the spine, the relaxed handstand band.
        XCTAssertEqual(features.value("head")[0], 24.0, accuracy: 0.5)
        XCTAssertGreaterThan(features.value("head")[0], 15.0)
        XCTAssertLessThan(features.value("head")[0], 30.0)
        // The legs overlap, so the separation is zero.
        XCTAssertEqual(features.value("leg_separation")[0], 0.0, accuracy: 1e-6)
        // A side view has no shoulder width to divide the hand width by, and
        // the feature says so rather than inventing a ratio.
        XCTAssertTrue(features.value("hand_width")[0].isNaN)
    }

    /// `test_every_feature_is_measured_on_every_frame_of_a_clip`.
    func testEveryFeatureIsMeasuredOnEveryFrameOfAClip() throws {
        let features = try measure([FeaturesTests.line, FeaturesTests.pike, FeaturesTests.banana])

        XCTAssertEqual(features.frames, 3)
        XCTAssertEqual(
            Set(features.values.keys), Set(Features.featureNames + Features.comColumns))
        for name in Features.featureNames {
            XCTAssertEqual(features.value(name).count, 3, name)
        }
        // The pike's hip is folded; the line's is not.
        XCTAssertLessThan(features.value("hip_angle")[1], features.value("hip_angle")[0])
    }

    /// `test_a_stack_that_has_drifted_over_one_hand_is_read_as_one`.
    func testAStackThatHasDriftedOverOneHandIsReadAsOne() throws {
        var pose = FeaturesTests.line
        pose.uShoulder = -0.06
        pose.uHip = -0.05
        pose.uKnee = -0.03
        pose.uAnkle = -0.01

        XCTAssertEqual(try value([pose], "off_shoulder"), -0.06, accuracy: 1e-9)
        XCTAssertEqual(try value([pose], "off_hip"), -0.05, accuracy: 1e-9)
        XCTAssertEqual(try value([pose], "off_knee"), -0.03, accuracy: 1e-9)
        XCTAssertEqual(try value([pose], "off_ankle"), -0.01, accuracy: 1e-9)
        // line_deviation is the worst of the four, and the body angle is the
        // same error read as a lean of the whole line.
        XCTAssertEqual(try value([pose], "line_deviation"), 0.06, accuracy: 1e-9)
        XCTAssertEqual(
            try value([pose], "body_angle"),
            atan2(-0.01, 1.15) * (180.0 / Double.pi), accuracy: 1e-6)
    }

    /// `test_line_deviation_uses_the_stations_the_frame_can_see`.
    func testLineDeviationUsesTheStationsTheFrameCanSee() throws {
        // A frame with no ankles is judged on the three stations it has.
        var pose = FeaturesTests.line
        pose.uHip = 0.07
        pose.hidden = ["left_ankle", "right_ankle"]

        XCTAssertEqual(try value([pose], "line_deviation"), 0.07, accuracy: 1e-9)
        XCTAssertTrue(try value([pose], "body_angle").isNaN)
        XCTAssertTrue(try value([pose], "banana").isNaN)
        // It is still not a valid frame: the body angle and the banana are
        // both measured against the feet.
        XCTAssertFalse(try measure([pose]).valid[0])

        // With none of the four stations there is no line at all.
        var nothing = FeaturesTests.line
        nothing.uHip = 0.07
        nothing.hidden = [
            "left_ankle", "right_ankle", "left_knee", "right_knee",
            "left_shoulder", "right_shoulder", "left_hip", "right_hip",
        ]
        XCTAssertTrue(try value([nothing], "line_deviation").isNaN)
        XCTAssertFalse(try measure([nothing]).valid[0])
    }

    // MARK: - The shapes that are not a line

    /// `test_a_pike_is_a_pike`.
    func testAPikeIsAPike() throws {
        let features = try measure([FeaturesTests.pike])

        XCTAssertLessThan(features.value("hip_angle")[0], Features.pikeHipDeg)
        XCTAssertLessThan(features.value("hip_angle")[0], Features.openShoulderDeg)
        // The legs are straight even in a pike, and the arms are straight
        // whatever the hips do.
        XCTAssertEqual(features.value("knee_angle")[0], 180.0, accuracy: 0.5)
        XCTAssertEqual(features.value("elbow_angle")[0], 180.0, accuracy: 0.5)
        // A pike is a body folded about the hip, not a body off the vertical.
        XCTAssertEqual(features.value("line_deviation")[0], 0.0, accuracy: 1e-9)
        XCTAssertEqual(features.value("banana")[0], 0.0, accuracy: 1e-9)
    }

    /// `test_a_banana_moves_the_hip_and_is_signed_towards_the_face`.
    func testABananaMovesTheHipAndIsSignedTowardsTheFace() throws {
        let features = try measure([FeaturesTests.banana])

        XCTAssertEqual(features.value("off_hip")[0], 0.08, accuracy: 1e-9)
        XCTAssertEqual(features.value("banana")[0], 0.08, accuracy: 0.01)
        // The nose of this pose is at +u, so the hips displaced to +u are
        // towards the way the athlete faces: positive.
        XCTAssertGreaterThan(features.value("banana")[0], 0.0)
        // A curve is also a bent hip — the same fault seen from two sides.
        XCTAssertLessThan(features.value("hip_angle")[0], 180.0)
        XCTAssertGreaterThan(features.value("hip_angle")[0], 130.0)
        XCTAssertGreaterThan(features.value("knee_angle")[0], 170.0)
        // A bigger curve is what the tolerance is for.
        var worse = FeaturesTests.banana
        worse.uHip = 0.14
        worse.uKnee = 0.05
        XCTAssertGreaterThan(
            abs(try value([worse], "banana")), Features.bananaToleranceL)
        // A body that curves the other way is the same fault with the other
        // sign.
        let other = try measure([FeaturesTests.banana.mirrored()])
        XCTAssertEqual(other.value("off_hip")[0], -0.08, accuracy: 1e-9)
        XCTAssertEqual(other.value("banana")[0], 0.08, accuracy: 0.01)
    }

    /// `test_a_straight_body_leaning_off_the_vertical_is_not_a_banana`.
    func testAStraightBodyLeaningOffTheVerticalIsNotABanana() throws {
        // Every joint, the hands included, is on one straight line leaning off
        // the vertical: the curve is zero, and what sees it is the stacking
        // offsets and the body angle.
        func onTheLine(_ v: Double) -> Double { 0.18 * v / 1.15 }
        var lean = FeaturesTests.line
        lean.uShoulder = onTheLine(0.39)
        lean.uHip = onTheLine(0.75)
        lean.uKnee = onTheLine(0.95)
        lean.uAnkle = 0.18
        lean.uNose = onTheLine(0.95) + 0.05
        lean.uElbow = onTheLine(0.195)
        let features = try measure([lean])

        XCTAssertEqual(features.value("banana")[0], 0.0, accuracy: 1e-9)
        XCTAssertEqual(features.value("line_deviation")[0], 0.18, accuracy: 1e-9)
        XCTAssertEqual(
            features.value("body_angle")[0],
            atan2(0.18, 1.15) * (180.0 / Double.pi), accuracy: 1e-6)
        // The stack has come out with it: the shoulders are off the hands too.
        XCTAssertEqual(features.value("off_shoulder")[0], 0.061, accuracy: 1e-3)
        for name in ["shoulder_angle", "hip_angle", "knee_angle", "elbow_angle"] {
            XCTAssertEqual(features.value(name)[0], 180.0, accuracy: 0.5, name)
        }
    }

    /// `test_a_shape_feature_does_not_depend_on_which_way_the_camera_was`.
    func testAShapeFeatureDoesNotDependOnWhichWayTheCameraWas() throws {
        let line = try measure([FeaturesTests.line])
        let banana = try measure([FeaturesTests.banana])
        let mirrored = try measure([FeaturesTests.banana.mirrored()])

        // The offsets are a direction in the image, so a mirrored recording
        // flips them.
        XCTAssertEqual(mirrored.value("off_hip")[0], -0.08, accuracy: 1e-9)
        // The shape features are anatomical: same athlete, same number, in
        // either case.
        for name in ["banana", "head"] {
            XCTAssertEqual(
                banana.value(name)[0], mirrored.value(name)[0], accuracy: 1e-6, name)
        }
        XCTAssertEqual(line.value("banana")[0], 0.0, accuracy: 1e-9)
    }

    /// `test_a_dropped_head_reads_as_a_head_off_the_spine`.
    func testADroppedHeadReadsAsAHeadOffTheSpine() throws {
        let neutral = try value([FeaturesTests.line], "head")
        var droppedPose = FeaturesTests.line
        droppedPose.uNose = 0.12
        droppedPose.vNose = 0.38
        let dropped = try value([droppedPose], "head")

        XCTAssertLessThan(neutral, Features.headFlexionDeg)
        XCTAssertEqual(dropped, 85.0, accuracy: 1.0)
        XCTAssertGreaterThan(dropped, Features.headFlexionDeg)
        // It is a magnitude, not a signed angle: a head thrown back is the
        // same fault and the same number.
        var thrownPose = droppedPose
        thrownPose.uNose = -0.12
        XCTAssertEqual(try value([thrownPose], "head"), dropped, accuracy: 1e-6)
        // No nose, no head and no signed shape — and the nose is the only
        // joint whose absence does not make the frame invalid.
        var blindPose = FeaturesTests.line
        blindPose.hidden = ["nose"]
        XCTAssertTrue(try value([blindPose], "head").isNaN)
        XCTAssertTrue(try value([blindPose], "banana").isNaN)
        XCTAssertTrue(try measure([blindPose]).valid[0])
    }

    /// `test_bent_arms_are_a_severe_fault`.
    func testBentArmsAreASevereFault() throws {
        let features = try measure([FeaturesTests.bentArms])

        XCTAssertLessThan(features.value("elbow_angle")[0], Features.bentElbowDeg)
        XCTAssertLessThan(features.value("elbow_angle")[0], 120.0)
        // The rest of the body is still straight.
        XCTAssertEqual(features.value("hip_angle")[0], 180.0, accuracy: 0.5)
        XCTAssertEqual(features.value("line_deviation")[0], 0.0, accuracy: 1e-9)
    }

    /// `test_a_straddle_is_a_leg_separation_and_not_a_stack_error`.
    func testAStraddleIsALegSeparationAndNotAStackError() throws {
        let features = try measure([FeaturesTests.straddle])

        XCTAssertGreaterThan(features.value("leg_separation")[0], Features.splitDeg)
        XCTAssertEqual(features.value("leg_separation")[0], 64.0, accuracy: 1.0)
        // The ankles are apart but their midpoint is where a line puts it, so
        // this is not a stacking fault.
        XCTAssertEqual(features.value("off_ankle")[0], 0.0, accuracy: 1e-9)
        XCTAssertGreaterThan(
            features.value("leg_separation")[0],
            try measure([FeaturesTests.line]).value("leg_separation")[0])
    }

    /// `test_a_split_needs_both_legs`: one side is enough for every midpoint
    /// feature, and the separation alone is `NaN` — a zero would be a claim
    /// nobody made.
    func testASplitNeedsBothLegs() throws {
        var oneLeg = FeaturesTests.line
        oneLeg.hidden = ["left_ankle"]
        let features = try measure([oneLeg])

        XCTAssertTrue(features.value("leg_separation")[0].isNaN)
        XCTAssertTrue(features.valid[0])
        XCTAssertEqual(features.value("off_ankle")[0], 0.0, accuracy: 1e-9)
    }

    /// `test_hand_width_is_a_ratio_of_two_widths`.
    func testHandWidthIsARatioOfTwoWidths() throws {
        XCTAssertEqual(
            try value([FeaturesTests.frontal], "hand_width"), 1.0, accuracy: 1e-9)
        var narrow = FeaturesTests.frontal
        narrow.handSpan = 0.12
        XCTAssertEqual(try value([narrow], "hand_width"), 0.5, accuracy: 1e-9)
        // It is the one feature that needs both sides, so hiding one is enough
        // to lose it while the angles it shares its joints with survive.
        var oneSide = FeaturesTests.frontal
        oneSide.hidden = ["right_shoulder"]
        XCTAssertTrue(try value([oneSide], "hand_width").isNaN)
        XCTAssertTrue(try measure([oneSide]).valid[0])
    }

    // MARK: - Sides, and frames the model cannot see into

    /// `test_one_visible_side_is_used_when_the_other_is_not`.
    func testOneVisibleSideIsUsedWhenTheOtherIsNot() throws {
        var leftOnly = FeaturesTests.line
        leftOnly.hidden = ["right_hip", "right_knee", "right_ankle", "right_elbow"]
        var rightOnly = FeaturesTests.line
        rightOnly.hidden = ["left_hip", "left_knee", "left_ankle", "left_elbow"]
        let both = try measure([FeaturesTests.line])

        for name in ["off_hip", "off_knee", "off_ankle", "line_deviation", "body_angle"] {
            XCTAssertEqual(
                try measure([leftOnly]).value(name)[0], both.value(name)[0], accuracy: 1e-9,
                "left only: \(name)")
            XCTAssertEqual(
                try measure([rightOnly]).value(name)[0], both.value(name)[0], accuracy: 1e-9,
                "right only: \(name)")
        }
        for name in ["shoulder_angle", "hip_angle", "knee_angle", "elbow_angle"] {
            XCTAssertEqual(
                try measure([leftOnly]).value(name)[0], 180.0, accuracy: 0.5, name)
            XCTAssertEqual(
                try measure([rightOnly]).value(name)[0], 180.0, accuracy: 0.5, name)
        }
        XCTAssertTrue(try measure([leftOnly]).valid[0])
    }

    /// `test_the_two_sides_are_averaged_where_both_are_there`.
    func testTheTwoSidesAreAveragedWhereBothAreThere() throws {
        var both = FeaturesTests.line
        both.uHip = 0.10
        both.spread = ["hip": 0.20, "knee": 0.20, "ankle": 0.20]
        var left = both
        left.hidden = ["right_hip", "right_knee", "right_ankle"]

        XCTAssertEqual(try value([both], "off_hip"), 0.10, accuracy: 1e-9)
        // With the right hip hidden the left one is the midpoint: 0.10 - 0.10.
        XCTAssertEqual(try value([left], "off_hip"), 0.0, accuracy: 1e-9)
    }

    /// `test_a_frame_the_model_cannot_see_into_is_all_nan`.
    func testAFrameTheModelCannotSeeIntoIsAllNaN() throws {
        var invisible = FeaturesTests.line
        invisible.visible = false
        let features = try measure([FeaturesTests.line, invisible, FeaturesTests.line])

        XCTAssertEqual(features.valid, [true, false, true])
        for name in Features.featureNames {
            XCTAssertTrue(features.value(name)[1].isNaN, name)
        }
        XCTAssertFalse(features.value("line_deviation")[0].isNaN)
    }

    /// `test_a_frame_without_a_wrist_has_no_body_frame`: the origin is the
    /// wrist midpoint, so a frame with no wrist is a frame whose body frame
    /// was never placed, and every feature comes out `NaN` rather than a
    /// number measured against an origin of zeros.
    func testAFrameWithoutAWristHasNoBodyFrame() throws {
        var pose = FeaturesTests.line
        pose.hidden = ["left_wrist", "right_wrist"]
        let features = try Features.bodyFeatures(trackFromPixels([pose]))

        XCTAssertFalse(features.valid[0])
        for name in Features.featureNames {
            XCTAssertTrue(features.value(name)[0].isNaN, name)
        }
    }

    /// The body frame is the wrist midpoint in body lengths, whatever the
    /// pixels do — the core of `test_the_body_frame_is_the_wrist_mpoint…`.
    func testTheBodyFrameIsTheWristMidpointInBodyLengths() throws {
        let here = trackFromPixels([FeaturesTests.line])
        let there = trackFromPixels([FeaturesTests.line], wristX: 900.0, wristY: 120.0)
        let twice = trackFromPixels([FeaturesTests.line], length: 2.0 * FeaturesTests.length)

        // The origin is where the hands are, so moving the hands moves nothing
        // else; the unit is the body length, so a twice-as-tall athlete
        // measures the same.
        for joint in Features.wantedJoints {
            let first = here.point(joint)[0]
            let moved = there.point(joint)[0]
            let scaled = twice.point(joint)[0]
            if first.isFinite {
                XCTAssertEqual(first.u, moved.u, accuracy: 1e-9, "\(joint) moved u")
                XCTAssertEqual(first.v, moved.v, accuracy: 1e-9, "\(joint) moved v")
                XCTAssertEqual(first.u, scaled.u, accuracy: 1e-9, "\(joint) scaled u")
                XCTAssertEqual(first.v, scaled.v, accuracy: 1e-9, "\(joint) scaled v")
            } else {
                XCTAssertTrue(moved.u.isNaN, "\(joint)")
                XCTAssertTrue(scaled.u.isNaN, "\(joint)")
            }
        }
        XCTAssertEqual(here.point(.leftWrist)[0].u, 0.0, accuracy: 1e-9)
        XCTAssertEqual(here.point(.leftWrist)[0].v, 0.0, accuracy: 1e-9)
        XCTAssertEqual(here.point(.rightWrist)[0].u, 0.0, accuracy: 1e-9)
        XCTAssertEqual(here.point(.leftAnkle)[0].u, 0.0, accuracy: 1e-9)
        XCTAssertEqual(here.point(.leftAnkle)[0].v, 1.15, accuracy: 1e-9)
    }

    // MARK: - One clip: labels, holds, the table

    /// `test_the_per_frame_table_carries_the_labels_and_every_feature`, minus
    /// the table itself (the CSV / parquet writing is not ported).
    func testExtractCarriesTheLabelsAndEveryFeature() throws {
        let features = try extract(
            [FeaturesTests.line, FeaturesTests.pike, FeaturesTests.banana],
            labels: ["hold", "hold", "unknown"])

        XCTAssertEqual(
            features.tMs,
            [0, FeaturesTests.frameMs, 2 * FeaturesTests.frameMs])
        XCTAssertEqual(features.phase, [.hold, .hold, .unknown])
        XCTAssertEqual(features.holdId, [0, 0, -1])
        XCTAssertEqual(features.valid, [true, true, true])
        XCTAssertTrue(features.usable)
        XCTAssertTrue(features.unusableReason.isEmpty)
        XCTAssertEqual(
            Set(features.values.keys), Set(Features.featureNames + Features.comColumns))
        // The table is the answer, not a summary of it: the pike's hip is
        // folded, the banana's hip is bent by the curve rather than folded.
        XCTAssertEqual(features.value("hip_angle")[1], 0.0, accuracy: 0.5)
        XCTAssertEqual(features.value("hip_angle")[2], 153.4, accuracy: 0.5)
        for (index, expected) in [0.0, 0.0, 0.08].enumerated() {
            XCTAssertEqual(
                features.value("line_deviation")[index], expected, accuracy: 1e-6,
                "frame \(index) line_deviation")
        }
    }

    /// `test_the_hold_table_summarises_each_hold`.
    func testTheHoldTableSummarisesEachHold() throws {
        let labels = [String](repeating: "hold", count: 4) + ["unknown"]
            + [String](repeating: "hold", count: 3)
        let features = try extract(
            repeated(FeaturesTests.line, 8), labels: labels)
        let rows = Features.holdRows(features)

        XCTAssertEqual(rows.count, 2)
        XCTAssertEqual(rows.map(\.holdId), [0, 1])
        XCTAssertEqual(rows.map(\.holdFrames), [4, 3])
        XCTAssertEqual(rows.map(\.validFrames), [4, 3])
        XCTAssertEqual(rows.map(\.holdStartMs), [0, 5 * FeaturesTests.frameMs])
        // (end - start) of the timestamps of the hold's own frames.
        XCTAssertEqual(
            rows[0].holdDurationS, Double(3 * FeaturesTests.frameMs) / 1000.0, accuracy: 1e-3)
        XCTAssertEqual(
            rows[1].holdDurationS, Double(2 * FeaturesTests.frameMs) / 1000.0, accuracy: 1e-3)
        XCTAssertEqual(rows[0].stats["line_deviation_median"] ?? .nan, 0.0, accuracy: 1e-9)
        XCTAssertEqual(rows[1].stats["line_deviation_median"] ?? .nan, 0.0, accuracy: 1e-9)
        XCTAssertEqual(rows[0].stats["hip_angle_median"] ?? .nan, 180.0, accuracy: 0.5)
        XCTAssertEqual(rows[0].stats["hip_angle_iqr"] ?? .nan, 0.0, accuracy: 0.5)

        // Every numeric column of HOLD_SUMMARY_COLUMNS is in the row, and
        // nothing else is.
        let fields: Set<String> = [
            "hold_id", "hold_frames", "valid_frames", "hold_start_ms", "hold_end_ms",
            "hold_duration_s",
        ]
        XCTAssertEqual(
            Set(rows[0].stats.keys),
            Set(Features.holdSummaryColumns).subtracting(fields))
        XCTAssertEqual(
            Set(rows[0].stats.keys),
            Set(Features.featureNames.flatMap { ["\($0)_median", "\($0)_iqr"] })
                .union(Features.holdStabilityNames))
    }

    /// `test_a_hold_is_summarised_over_the_frames_that_could_be_measured`.
    func testAHoldIsSummarisedOverTheFramesThatCouldBeMeasured() throws {
        // Four hold frames, one of them a frame the model saw nobody in: the
        // medians are over the other three and the row says so.
        var shifted = FeaturesTests.line
        shifted.uHip = 0.10
        var invisible = FeaturesTests.line
        invisible.visible = false
        let features = try extract(
            [FeaturesTests.line, shifted, FeaturesTests.line, invisible],
            labels: [String](repeating: "hold", count: 4))
        let row = Features.holdRows(features)[0]

        XCTAssertEqual(row.holdFrames, 4)
        XCTAssertEqual(row.validFrames, 3)
        // Three measurable hold frames: 0, 0.10 and 0, so the median is the
        // line and the IQR is the spread of the frames that moved.
        XCTAssertEqual(row.stats["off_hip_median"] ?? .nan, 0.0, accuracy: 1e-6)
        XCTAssertEqual(row.stats["off_hip_iqr"] ?? .nan, 0.05, accuracy: 1e-6)
    }

    /// `test_the_longest_hold_is_the_one_a_clip_is_ranked_by`.
    func testTheLongestHoldIsTheOneAClipIsRankedBy() throws {
        let labels = [String](repeating: "hold", count: 5)
            + [String](repeating: "exit", count: 2)
            + [String](repeating: "hold", count: 9)
        let features = try extract(repeated(FeaturesTests.line, 16), labels: labels)

        XCTAssertEqual(features.holdIds(), [0, 1])
        XCTAssertEqual(features.longestHoldId(), 1)
        XCTAssertEqual(
            features.holdDurationS(0), Double(4 * FeaturesTests.frameMs) / 1000.0,
            accuracy: 1e-9)
        XCTAssertEqual(
            features.holdDurationS(1), Double(8 * FeaturesTests.frameMs) / 1000.0,
            accuracy: 1e-9)
        XCTAssertEqual(
            features.holdMask(1),
            [Bool](repeating: false, count: 7) + [Bool](repeating: true, count: 9))
        XCTAssertEqual(
            features.measurableHoldMask(1),
            [Bool](repeating: false, count: 7) + [Bool](repeating: true, count: 9))
    }

    /// `test_a_clip_with_no_hold_has_no_longest_one`.
    func testAClipWithNoHoldHasNoLongestOne() throws {
        let features = try extract(
            repeated(FeaturesTests.line, 4), labels: [String](repeating: "pre", count: 4))

        XCTAssertEqual(features.holdIds(), [])
        XCTAssertEqual(features.longestHoldId(), PhaseSegmenter.noHold)
        XCTAssertTrue(Features.holdRows(features).isEmpty)
    }

    /// `test_a_clip_with_no_scale_has_no_features`: without `L` there is no
    /// unit to measure an offset or an angle in.
    func testAClipWithNoScaleHasNoFeatures() throws {
        let poses = repeated(FeaturesTests.line, 3)
        let features = try extract(
            poses, labels: [String](repeating: "hold", count: 3), bodyLength: .nan)

        XCTAssertFalse(features.usable)
        // Python's `f"no body length ({body_length!r} px)"` with `nan`.
        XCTAssertEqual(features.unusableReason, "no body length (nan px)")
        XCTAssertFalse(features.valid.contains(true))
        for name in Features.featureNames {
            XCTAssertTrue(features.value(name).allSatisfy { $0.isNaN }, name)
        }
        XCTAssertTrue(features.comComplete.allSatisfy { !$0 })
        XCTAssertTrue(features.balanceZone.allSatisfy { $0 == nil })

        let rows = Features.holdRows(features)
        XCTAssertEqual(rows.count, 1)
        XCTAssertTrue(rows[0].stats["line_deviation_median"]?.isNaN ?? false)
        XCTAssertTrue(rows[0].stats["com_sway_sd"]?.isNaN ?? false)

        // The other refusal path: a usable sidecar whose total is zero.
        let zero = try extract(
            poses, labels: [String](repeating: "hold", count: 3), bodyLength: 0.0)
        XCTAssertEqual(zero.unusableReason, "no body length (0.0 px)")
        XCTAssertFalse(zero.usable)
    }

    /// `test_a_frame_the_trainer_is_in_the_way_is_not_measured`.
    func testAFrameTheTrainerIsInTheWayIsNotMeasured() throws {
        let features = try extract(
            repeated(FeaturesTests.line, 3),
            labels: [String](repeating: "hold", count: 3),
            trainerContact: [false, true, false])

        XCTAssertEqual(features.valid, [true, false, true])
        // The numbers of the contact frame are still in the table, as
        // everywhere else in the pipeline: what is not true is that they are
        // the athlete's.
        XCTAssertEqual(features.value("line_deviation")[1], 0.0, accuracy: 1e-9)
        // ... and it is out of the hold's summary.
        let row = Features.holdRows(features)[0]
        XCTAssertEqual(row.holdFrames, 3)
        XCTAssertEqual(row.validFrames, 2)
    }

    // MARK: - The centre of mass and balance

    /// `test_the_centre_of_mass_columns_are_written_per_frame`.
    func testTheCentreOfMassColumnsAreWrittenPerFrame() throws {
        let features = try extract([FeaturesTests.line], labels: ["hold"])

        // The symmetric line's CoM is on the vertical through the hands to
        // within the head's own pull — the nose is the one joint off the line
        // — and every segment was there.
        XCTAssertEqual(
            features.value("com_u")[0], CentreOfMass.massHeadNeck * 0.04, accuracy: 1e-9)
        XCTAssertGreaterThan(features.value("com_v")[0], 0.0)
        XCTAssertLessThan(features.value("com_v")[0], FeaturesTests.line.vHip)
        XCTAssertTrue(features.comComplete[0])
        // The nose is at +u, so the athlete faces +u, so the CoM towards the
        // fingers is the CoM to the right: positive, and well inside the base.
        XCTAssertEqual(features.value("facing_sign")[0], 1.0)
        XCTAssertEqual(
            features.value("com_forward")[0], features.value("com_u")[0], accuracy: 1e-12)
        XCTAssertEqual(features.balanceZone[0], .ok)

        // The same clip shot from the other side of the athlete: every u
        // negated, so the sign flips and com_forward — the balance — does not.
        let mirrored = try extract([FeaturesTests.line.mirrored()], labels: ["hold"])
        XCTAssertEqual(mirrored.value("facing_sign")[0], -1.0)
        XCTAssertEqual(
            mirrored.value("com_u")[0], -features.value("com_u")[0], accuracy: 1e-9)
        XCTAssertEqual(
            mirrored.value("com_forward")[0], features.value("com_forward")[0],
            accuracy: 1e-9)
        XCTAssertEqual(mirrored.balanceZone[0], .ok)

        // A frame the model saw nobody in has no CoM to speak of: NaN,
        // incomplete, and no zone — not the frame's last known position.
        var invisible = FeaturesTests.line
        invisible.visible = false
        let unseen = try extract([invisible], labels: ["hold"])
        XCTAssertTrue(unseen.value("com_u")[0].isNaN)
        XCTAssertTrue(unseen.value("com_forward")[0].isNaN)
        XCTAssertFalse(unseen.comComplete[0])
        XCTAssertNil(unseen.balanceZone[0])
    }

    /// `test_the_balance_zone_says_which_side_of_the_base_the_com_is_on`.
    func testTheBalanceZoneSaysWhichSideOfTheBaseTheComIsOn() throws {
        let over = try extract([shiftedForward(0.3)], labels: ["hold"])
        let under = try extract([shiftedForward(-0.3)], labels: ["hold"])

        // Forwards past the fingertips, and the athlete faces +u, so this is
        // the overbalance side; backwards past the heel of the hand is the
        // other.
        XCTAssertGreaterThan(
            over.value("com_forward")[0], CentreOfMass.baseFront)
        XCTAssertEqual(over.balanceZone[0], .over)
        XCTAssertLessThan(under.value("com_forward")[0], -CentreOfMass.baseBack)
        XCTAssertEqual(under.balanceZone[0], .under)
        // The zones are read off com_forward and nothing else, so a mirrored
        // clip with the same body in the same place lands in the same zone.
        let mirroredOver = try extract(
            [shiftedForward(0.3).mirrored()], labels: ["hold"])
        XCTAssertEqual(mirroredOver.balanceZone[0], .over)
        XCTAssertEqual(mirroredOver.value("facing_sign")[0], -1.0)
    }

    /// `test_the_facing_sign_is_the_holds_vote_and_not_each_frames`.
    func testTheFacingSignIsTheHoldsVoteAndNotEachFrames() throws {
        var first = FeaturesTests.line
        first.uNose = 0.05
        var second = FeaturesTests.line
        second.uNose = -0.02
        var third = FeaturesTests.line
        third.uNose = 0.04
        var fourth = FeaturesTests.line
        fourth.uNose = 0.06
        var fifth = FeaturesTests.line
        fifth.uNose = -0.05
        let features = try extract(
            [first, second, third, fourth, fifth],
            labels: ["hold", "hold", "hold", "hold", "pre"])

        XCTAssertEqual(
            features.value("facing_sign"), [1.0, 1.0, 1.0, 1.0, -1.0])
        // com_forward is com_u under the hold's sign on every hold frame — and
        // the nose still moved the CoM itself, which is a measurement, not a
        // vote: the second frame's CoM is to the left of the hands, and stays
        // there.
        for index in 0..<4 {
            XCTAssertEqual(
                features.value("com_forward")[index], features.value("com_u")[index],
                accuracy: 1e-12, "frame \(index)")
        }
        XCTAssertLessThan(features.value("com_u")[1], 0.0)
        XCTAssertLessThan(features.value("com_forward")[1], 0.0)
        // The frame outside the hold is measured with its own sign instead.
        XCTAssertEqual(
            features.value("com_forward")[4], -features.value("com_u")[4], accuracy: 1e-12)
    }

    /// `test_a_hold_that_never_moves_has_no_sway`.
    func testAHoldThatNeverMovesHasNoSway() throws {
        let features = try extract(
            repeated(FeaturesTests.line, 6), labels: [String](repeating: "hold", count: 6))
        let stability = Features.holdStability(features, holdId: 0)
        let row = Features.holdRows(features)[0]

        // Six frames of the same pose: the CoM is in one place, so the sway,
        // the range and the speed are all zero — to within a float's last
        // bits, which is why the written row, rounded to five decimals, is
        // exactly zero.
        for name in ["com_sway_sd", "com_sway_range", "com_speed_rms"] {
            XCTAssertEqual(stability[name] ?? .nan, 0.0, accuracy: 1e-12, name)
            XCTAssertEqual(row.stats[name] ?? .nan, 0.0, accuracy: 0.0, name)
        }
        // Where it sat is where it is, and nothing left the base of support.
        XCTAssertEqual(row.stats["com_forward_median"] ?? .nan, 0.00324, accuracy: 1e-6)
        XCTAssertEqual(row.stats["pct_over"] ?? .nan, 0.0, accuracy: 0.0)
        XCTAssertEqual(row.stats["pct_under"] ?? .nan, 0.0, accuracy: 0.0)
        // The hip and the shoulder were both doing nothing.
        XCTAssertEqual(row.stats["hip_angle_sd"] ?? .nan, 0.0, accuracy: 0.0)
        XCTAssertEqual(row.stats["shoulder_angle_sd"] ?? .nan, 0.0, accuracy: 0.0)
    }

    /// `test_a_hold_that_could_not_be_measured_has_no_balance_numbers`.
    func testAHoldThatCouldNotBeMeasuredHasNoBalanceNumbers() throws {
        var invisible = FeaturesTests.line
        invisible.visible = false
        let features = try extract(
            repeated(invisible, 4), labels: [String](repeating: "hold", count: 4))
        let row = Features.holdRows(features)[0]

        XCTAssertEqual(row.validFrames, 0)
        for name in Features.holdStabilityNames {
            XCTAssertTrue(
                row.stats[name]?.isNaN ?? false,
                "\(name) is a number on a hold nobody could see")
        }
    }

    /// A one-frame hold has no spread: the SDs, the range and the speed need
    /// two frames, the median and the percentages do not.
    func testOneFrameHoldHasNaNSpreads() throws {
        let features = try extract([FeaturesTests.line], labels: ["hold"])
        let stability = Features.holdStability(features, holdId: 0)
        let rows = Features.holdRows(features)

        XCTAssertEqual(rows.count, 1)
        let row = rows[0]
        XCTAssertEqual(row.holdFrames, 1)
        XCTAssertEqual(row.validFrames, 1)
        XCTAssertEqual(row.holdStartMs, 0)
        XCTAssertEqual(row.holdEndMs, 0)
        XCTAssertEqual(row.holdDurationS, 0.0, accuracy: 0.0)
        for name in ["com_sway_sd", "com_sway_range", "com_speed_rms", "hip_angle_sd",
            "shoulder_angle_sd"]
        {
            XCTAssertTrue(stability[name]?.isNaN ?? false, "\(name) on a one-frame hold")
            XCTAssertTrue(row.stats[name]?.isNaN ?? false, "\(name) written for one frame")
        }
        // The median and the percentages are defined on one.
        XCTAssertEqual(stability["com_forward_median"] ?? .nan, 0.00324, accuracy: 1e-6)
        XCTAssertEqual(stability["pct_over"] ?? .nan, 0.0, accuracy: 0.0)
        XCTAssertEqual(row.stats["line_deviation_median"] ?? .nan, 0.0, accuracy: 1e-9)
        XCTAssertEqual(row.stats["line_deviation_iqr"] ?? .nan, 0.0, accuracy: 1e-9)
    }

    /// The stability numbers on a hand-computable trajectory: the **population**
    /// SD (the sample SD would be 0.05, not 0.0408…) and the speed taken over
    /// the **real** gaps between the timestamps, which are deliberately uneven
    /// because these clips are variable frame rate.
    func testTheStabilityUsesThePopulationSDAndTheRealTimeGaps() throws {
        let clip = swayClip(tMs: [0, 250, 1000], forward: [0.0, 0.1, 0.05])
        let stability = Features.holdStability(clip, holdId: 0)

        XCTAssertEqual(stability["com_forward_median"] ?? .nan, 0.05, accuracy: 1e-12)
        // sqrt(((0.05)^2 + (0.05)^2 + 0) / 3) — ddof = 0.
        XCTAssertEqual(
            stability["com_sway_sd"] ?? .nan, 0.040824829046386304, accuracy: 1e-12)
        // p95 − p5 of three samples: 0.095 − 0.005.
        XCTAssertEqual(stability["com_sway_range"] ?? .nan, 0.09, accuracy: 1e-12)
        // RMS of 0.1 L over 0.25 s and −0.05 L over 0.75 s — a fixed step
        // would give a different answer.
        XCTAssertEqual(
            stability["com_speed_rms"] ?? .nan, 0.28674417556808757, accuracy: 1e-9)
        XCTAssertEqual(
            stability["pct_over"] ?? .nan, 100.0 / 3.0, accuracy: 1e-9)
        XCTAssertEqual(stability["pct_under"] ?? .nan, 0.0, accuracy: 1e-9)
        // The angles were all NaN, so neither strategy could be read.
        XCTAssertTrue(stability["hip_angle_sd"]?.isNaN ?? false)
        XCTAssertTrue(stability["shoulder_angle_sd"]?.isNaN ?? false)
    }

    /// `test_the_sway_of_a_sinusoid_is_what_the_analytic_answer_says`: an
    /// analytic answer is the only check that cannot be satisfied by a
    /// consistently wrong implementation.
    func testTheSwayOfASinusoidIsWhatTheAnalyticAnswerSays() throws {
        let amplitude = 0.05
        let frequency = 1.0
        let timesMs = unevenTimes()
        let timesS = timesMs.map { Double($0) / 1000.0 }
        let forward = timesS.map { amplitude * sin(2.0 * Double.pi * frequency * $0) }
        let stability = Features.holdStability(
            swayClip(tMs: timesMs, forward: forward), holdId: 0)

        let analyticSD = amplitude / sqrt(2.0)
        let analyticSpeed = 2.0 * Double.pi * frequency * amplitude / sqrt(2.0)
        let sd = stability["com_sway_sd"] ?? .nan
        let speed = stability["com_speed_rms"] ?? .nan
        XCTAssertEqual(sd, analyticSD, accuracy: analyticSD * 0.05, "population SD of a sine")
        XCTAssertEqual(
            speed, analyticSpeed, accuracy: analyticSpeed * 0.05, "RMS of a sine's derivative")
        XCTAssertLessThan(abs(stability["com_forward_median"] ?? 1.0), 0.01)
        // The 5th and 95th percentiles of a sine sit at 0.988 of it.
        XCTAssertEqual(
            stability["com_sway_range"] ?? .nan, 1.9754 * amplitude,
            accuracy: 1.9754 * amplitude * 0.05)
        XCTAssertEqual(stability["pct_over"] ?? .nan, 0.0, accuracy: 1e-9)
        // The share of the hold behind the heel of the hand, where a 0.05 L
        // sway dips below -0.03 L.
        XCTAssertEqual(stability["pct_under"] ?? .nan, 29.5, accuracy: 4.0)

        // The same sway sampled three times as slowly gives the same speed:
        // the derivative is taken over the real gap between frames, so the
        // answer does not depend on how often the camera happened to fire.
        let coarse = stride(from: 0, through: timesMs[timesMs.count - 1], by: 67).map { $0 }
        let slow = Features.holdStability(
            swayClip(
                tMs: coarse,
                forward: coarse.map {
                    amplitude * sin(2.0 * Double.pi * frequency * Double($0) / 1000.0)
                }),
            holdId: 0)
        XCTAssertEqual(
            slow["com_speed_rms"] ?? .nan, analyticSpeed, accuracy: analyticSpeed * 0.05)
        XCTAssertEqual(
            slow["com_sway_sd"] ?? .nan, analyticSD, accuracy: analyticSD * 0.05)
    }

    /// 182 unevenly spaced timestamps — a variable frame rate, which is what
    /// every threshold in the pipeline is stated against. Python draws these
    /// from `np.random.default_rng(7)`; what the test needs is *uneven* gaps,
    /// not numpy's exact stream, so a 64-bit LCG stands in for it.
    private func unevenTimes() -> [Int] {
        var state: UInt64 = 0x2545_F491_4F6C_DD1D
        var times: [Int] = [0]
        for _ in 0..<181 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            let gap = 20 + Int((state >> 33) % 30)
            times.append(times[times.count - 1] + gap)
        }
        return times
    }
}
