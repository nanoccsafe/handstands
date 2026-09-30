import XCTest

@testable import HandstandCore

/// The stress diagram's logic (chainlink #48): severity with and without a
/// reference, the joint–feature table, the colour bands, the stack line, the
/// CoM inverse transform, the playback-time lookup — and the tolerance table
/// pinned against Python's `TOLERANCES`.
///
/// Everything here is built from synthetic frames in the test itself (the
/// way `FeaturesTests`/`ScorerTests` do their unit cases): no fixtures, no
/// real videos, no views.
final class StressDiagramTests: XCTestCase {
    // MARK: - The synthetic clip

    /// The clip's body length in display pixels — `FeaturesTests.length`'s
    /// number, so 0.1 L is exactly 30 px.
    private static let length = 300.0
    /// Where the wrist midpoint sits in the display frame.
    private static let wristX = 200.0
    private static let wristY = 500.0
    /// The synthetic clip's frame timestamps.
    private static let stamps = [0, 33, 66, 99]

    /// Where each joint sits in the display frame: a column up from the
    /// hands, the two sides a few pixels apart, the nose nearer the left
    /// shoulder (so "the nearer shoulder" has an answer).
    private static func station(_ joint: Joint) -> Point2 {
        switch joint {
        case .nose: return Point2(x: wristX - 10, y: wristY - 140)
        case .leftWrist: return Point2(x: wristX - 15, y: wristY)
        case .rightWrist: return Point2(x: wristX + 15, y: wristY)
        case .leftElbow, .rightElbow:
            let dx = joint == .leftElbow ? -10.0 : 10.0
            return Point2(x: wristX + dx, y: wristY - 60)
        case .leftShoulder, .rightShoulder:
            let dx = joint == .leftShoulder ? -5.0 : 5.0
            return Point2(x: wristX + dx, y: wristY - 120)
        case .leftHip, .rightHip:
            let dx = joint == .leftHip ? -5.0 : 5.0
            return Point2(x: wristX + dx, y: wristY - 220)
        case .leftKnee, .rightKnee:
            let dx = joint == .leftKnee ? -5.0 : 5.0
            return Point2(x: wristX + dx, y: wristY - 320)
        case .leftAnkle, .rightAnkle:
            let dx = joint == .leftAnkle ? -5.0 : 5.0
            return Point2(x: wristX + dx, y: wristY - 400)
        case .leftFootIndex, .rightFootIndex:
            let dx = joint == .leftFootIndex ? -5.0 : 5.0
            return Point2(x: wristX + dx, y: wristY - 420)
        }
    }

    /// A clip of `frames` frames: every joint at its station, every frame in
    /// a hold (unless `phase` says otherwise), every feature `NaN` until a
    /// test overrides it — and `facing_sign` 1, because a side view is
    /// *seen* and a test that cares flips it itself.
    ///
    /// `invalidJoints` drops joints out of `processed.frames[].valid`, and
    /// `invalidFrames` makes whole frames unmeasured (`features.valid`).
    private func makeAnalysis(
        frames: Int = 4,
        phase: Phase = .hold,
        valid: Bool = true,
        overrides: [String: [Double]] = [:],
        invalidJoints: Set<Joint> = [],
        invalidFrames: Set<Int> = [],
        balanceZone: [BalanceZone?]? = nil,
        bodyLength: Double = StressDiagramTests.length
    ) -> Analysis {
        precondition(frames >= 1, "at least one frame")

        // The processed trajectory: stations in pixels, `valid` minus what
        // the test hides.
        var processedFrames: [ProcessedFrame] = []
        for _ in 0..<frames {
            var joints: [Joint: Point2] = [:]
            var validJoints: Set<Joint> = []
            for joint in Joint.allCases {
                joints[joint] = Self.station(joint)
                if !invalidJoints.contains(joint) {
                    validJoints.insert(joint)
                }
            }
            processedFrames.append(
                ProcessedFrame(joints: joints, valid: validJoints, filled: []))
        }
        let processed = ProcessedClip(
            frames: processedFrames,
            bodyLength: BodyLength(
                reason: nil, torsoPx: bodyLength, thighPx: 0, shinPx: 0, frames: [:])
        )

        // The phases: one label per frame, holds numbered 0.
        let phases = ClipPhases(
            tMs: Self.stamps,
            phase: [Phase](repeating: phase, count: frames),
            holdId: [Int](repeating: phase == .hold ? 0 : PhaseSegmenter.noHold, count: frames),
            signals: PhaseSegmenter.unusableSignals(frames: frames, reason: ""),
            runs: [],
            usable: true,
            unusableReason: ""
        )

        // The features: every column `NaN` (measured where it is measured),
        // with the test's overrides on top.
        var values: [String: [Double]] = [:]
        for name in Features.featureNames + Features.comColumns {
            values[name] = [Double](repeating: .nan, count: frames)
        }
        values["facing_sign"] = [Double](repeating: 1.0, count: frames)
        for (name, column) in overrides {
            precondition(
                column.count == frames, "expected \(frames) values for \(name), got \(column.count)")
            values[name] = column
        }

        let features = ClipFeatures(
            tMs: Self.stamps,
            phase: [Phase](repeating: phase, count: frames),
            holdId: [Int](repeating: phase == .hold ? 0 : PhaseSegmenter.noHold, count: frames),
            values: values,
            balanceZone: balanceZone ?? [BalanceZone?](repeating: nil, count: frames),
            comComplete: [Bool](repeating: true, count: frames),
            valid: (0..<frames).map { !invalidFrames.contains($0) && valid },
            unusableReason: ""
        )

        return Analysis(
            processed: processed,
            phases: phases,
            features: features,
            holdScores: [],
            clipScore: nil
        )
    }

    /// A reference carrying only the named features at the given
    /// `(mean, sd)` — the tests' own numbers, never a real one.
    private func scoreReference(
        _ entries: [String: (mean: Double, sd: Double)]
    ) -> ScoreReference {
        ScoreReference(
            features: entries.mapValues { ScoreReference.Stat(mean: $0.mean, sd: $0.sd, n: 42) },
            nHolds: 42,
            builtFrom: "StressDiagramTests' inline reference"
        )
    }

    /// One joint of a drawn frame, by name.
    private func joint(
        _ name: Joint, in frame: DiagramFrame,
        file: StaticString = #filePath, line: UInt = #line
    ) throws -> DiagramJoint {
        try XCTUnwrap(
            frame.joints.first { $0.joint == name }, "no \(name.rawValue) in the frame",
            file: file, line: line)
    }

    /// One bone of a drawn frame, by its endpoints (either direction).
    private func bone(
        _ from: Joint, _ to: Joint, in frame: DiagramFrame,
        file: StaticString = #filePath, line: UInt = #line
    ) throws -> DiagramBone {
        try XCTUnwrap(
            frame.bones.first {
                ($0.from == from && $0.to == to) || ($0.from == to && $0.to == from)
            },
            "no bone \(from.rawValue)–\(to.rawValue) in the frame", file: file, line: line)
    }

    private func column(_ value: Double, _ count: Int) -> [Double] {
        [Double](repeating: value, count: count)
    }

    // MARK: - Outside a hold nothing is judged

    func testOutsideAHoldEveryJointIsNeutralAndNothingHasASeverity() throws {
        // A perfectly failing hip angle, on a frame nobody is holding on:
        // the fault does not exist yet.
        let analysis = makeAnalysis(phase: .pre, overrides: ["hip_angle": column(140, 4)])

        let frame = StressDiagram.frame(1, analysis: analysis, reference: nil, height: 640)

        XCTAssertFalse(frame.inHold)
        XCTAssertGreaterThan(frame.joints.count, 0, "the skeleton is still drawn")
        for joint in frame.joints {
            XCTAssertEqual(joint.band, .neutral, "\(joint.joint.rawValue) is grey outside a hold")
            XCTAssertNil(joint.severity, "\(joint.joint.rawValue) has no severity outside a hold")
        }
        for bone in frame.bones {
            XCTAssertEqual(bone.band, .neutral)
            XCTAssertNil(bone.severity)
        }
        XCTAssertEqual(frame.stackBand, .neutral)
        XCTAssertNil(frame.com, "no CoM column was measured on this clip")
    }

    /// A frame inside a hold the features could not measure is measured by
    /// nobody either — a trainer in front of the camera is not a fault.
    func testAnUnmeasurableFrameInsideAHoldIsNeutralToo() throws {
        let analysis = makeAnalysis(
            valid: false, overrides: ["hip_angle": column(140, 4)])

        let frame = StressDiagram.frame(0, analysis: analysis, reference: nil, height: 640)

        XCTAssertTrue(frame.inHold)
        XCTAssertEqual(try joint(.leftHip, in: frame).band, .neutral)
        XCTAssertNil(try joint(.leftHip, in: frame).severity)
        XCTAssertEqual(frame.stackBand, .neutral)
    }

    // MARK: - Without a reference: the built-in tolerances

    func testAThresholdFailureMakesTheHipJointBad() throws {
        // hip_angle 140° against PIKE_HIP_DEG 165, no reference: the excess
        // is 25° over a 20° band, so severity is min(1, 0.3 + 25/20) = 1.
        let analysis = makeAnalysis(overrides: ["hip_angle": column(140, 4)])

        let frame = StressDiagram.frame(0, analysis: analysis, reference: nil, height: 640)

        let hip = try joint(.leftHip, in: frame)
        XCTAssertEqual(hip.severity ?? -1, 1.0, accuracy: 1e-9)
        XCTAssertEqual(hip.band, .bad)
        XCTAssertEqual(try joint(.rightHip, in: frame).band, .bad, "both hips share the feature")
        // The knee passes its target: measured, on target, green.
        let knee = try joint(.leftKnee, in: frame)
        XCTAssertNil(knee.severity, "a NaN knee_angle is nobody's number")
        XCTAssertEqual(knee.band, .ok, "in a hold, unjudged joints are ok, not neutral")
    }

    func testAPassingToleranceIsZeroSeverityAndAGreenJoint() throws {
        // hip_angle 172° passes PIKE_HIP_DEG 165, and 0 is exactly on the
        // band boundary: `.ok`, never `.warn`.
        let analysis = makeAnalysis(overrides: ["hip_angle": column(172, 4)])

        let frame = StressDiagram.frame(0, analysis: analysis, reference: nil, height: 640)

        let hip = try joint(.leftHip, in: frame)
        XCTAssertEqual(hip.severity ?? -1, 0)
        XCTAssertEqual(hip.band, .ok)
    }

    func testTheStackLineBandTakesTheWorseOfLineDeviationAndBodyAngle() throws {
        // line_deviation 0.2 L against a 0.1 L target: excess 0.1 over the
        // 0.2 L band -> 0.3 + 0.5 = 0.8, bad on its own.
        let analysis = makeAnalysis(overrides: ["line_deviation": column(0.2, 4)])

        let frame = StressDiagram.frame(0, analysis: analysis, reference: nil, height: 640)

        XCTAssertEqual(frame.stackBand, .bad)
    }

    func testTheStackLineRunsTheFullHeightThroughTheWristMidpoint() throws {
        let frame = StressDiagram.frame(
            0, analysis: makeAnalysis(), reference: nil, height: 640)

        let line = try XCTUnwrap(frame.stackLine)
        // Wrists at 185 and 215 px: their midpoint is the column x = 200.
        XCTAssertEqual(line.top.x, Self.wristX, accuracy: 1e-9)
        XCTAssertEqual(line.bottom.x, Self.wristX, accuracy: 1e-9)
        XCTAssertEqual(line.top.y, 0)
        XCTAssertEqual(line.bottom.y, 640)
    }

    // MARK: - With a reference: z-scores

    func testWithAReferenceTwoStandardDeviationsIsSeverityHalfAndBad() throws {
        // hip_angle 182 against mean 176, sd 3: z = 2, severity 2/4 = 0.5 —
        // exactly the .warn/.bad boundary, which `.bad` owns.
        let analysis = makeAnalysis(overrides: ["hip_angle": column(182, 4)])
        let reference = scoreReference(["hip_angle": (mean: 176, sd: 3)])

        let frame = StressDiagram.frame(0, analysis: analysis, reference: reference, height: 640)

        let hip = try joint(.leftHip, in: frame)
        XCTAssertEqual(hip.severity ?? -1, 0.5, accuracy: 1e-9)
        XCTAssertEqual(hip.band, .bad)
    }

    func testWithAReferenceATargetHitIsSeverityZeroAndOk() throws {
        let analysis = makeAnalysis(overrides: ["hip_angle": column(176, 4)])
        let reference = scoreReference(["hip_angle": (mean: 176, sd: 3)])

        let frame = StressDiagram.frame(0, analysis: analysis, reference: reference, height: 640)

        let hip = try joint(.leftHip, in: frame)
        XCTAssertEqual(hip.severity ?? -1, 0)
        XCTAssertEqual(hip.band, .ok)
    }

    /// The SD floor, not a zero spread: a reference whose good holds agree
    /// to within 0.001 L still divides by `Scorer.sdFloorL` (0.01), exactly
    /// as the scorer does — `Scorer.sdFloors` is *the* table, not a copy.
    func testTheReferenceModeUsesTheScorersOwnSDFloors() throws {
        let analysis = makeAnalysis(overrides: ["off_hip": column(0.04, 4)])
        let reference = scoreReference(["off_hip": (mean: 0.0, sd: 0.001)])

        let frame = StressDiagram.frame(0, analysis: analysis, reference: reference, height: 640)

        // z = 0.04 / max(0.001, 0.01) = 4 -> capped at Z_CAP -> severity 1.
        let hip = try joint(.leftHip, in: frame)
        XCTAssertEqual(hip.severity ?? -1, 1.0, accuracy: 1e-9)
        XCTAssertEqual(hip.band, .bad)
    }

    func testFacingSignMinusOneFlipsOffHipBeforeTheZScoreIsTaken() throws {
        let overrides = ["off_hip": column(0.06, 4), "hip_angle": column(176, 4)]
        let reference = scoreReference([
            "off_hip": (mean: 0.03, sd: 0.03),
            "hip_angle": (mean: 176, sd: 3),
        ])

        // Facing right (sign +1): 0.06 is one SD above the mean -> 0.25.
        var plusOverrides = overrides
        plusOverrides["facing_sign"] = column(1, 4)
        let plusFrame = StressDiagram.frame(
            0, analysis: makeAnalysis(overrides: plusOverrides),
            reference: reference, height: 640)
        XCTAssertEqual(try joint(.leftHip, in: plusFrame).severity ?? -1, 0.25, accuracy: 1e-9)
        XCTAssertEqual(try joint(.leftHip, in: plusFrame).band, .warn)

        // Facing left (sign -1): the same *image* offset is now the far side
        // of the mean — three SDs off, 0.75, bad. The flip happens before
        // the z-score, never after it.
        var minusOverrides = overrides
        minusOverrides["facing_sign"] = column(-1, 4)
        let minusFrame = StressDiagram.frame(
            0, analysis: makeAnalysis(overrides: minusOverrides),
            reference: reference, height: 640)
        XCTAssertEqual(try joint(.leftHip, in: minusFrame).severity ?? -1, 0.75, accuracy: 1e-9)
        XCTAssertEqual(try joint(.leftHip, in: minusFrame).band, .bad)
    }

    func testANaNFacingSignSkipsTheSignedFeaturesRatherThanGuessing() throws {
        let overrides = ["off_hip": column(0.1, 4), "hip_angle": column(176, 4)]
        let reference = scoreReference([
            "off_hip": (mean: 0.03, sd: 0.03),
            "hip_angle": (mean: 176, sd: 3),
        ])

        // With a sign, off_hip is ~2.33 SDs off: severity over 0.5.
        var seenOverrides = overrides
        seenOverrides["facing_sign"] = column(1, 4)
        let seenFrame = StressDiagram.frame(
            0, analysis: makeAnalysis(overrides: seenOverrides),
            reference: reference, height: 640)
        XCTAssertGreaterThan(try joint(.leftHip, in: seenFrame).severity ?? -1, 0.5)

        // With a NaN sign the signed feature contributes *nothing*; what is
        // left of the hip is the unsigned hip_angle at its mean: 0, ok.
        var unknownOverrides = overrides
        unknownOverrides["facing_sign"] = column(.nan, 4)
        let unknownFrame = StressDiagram.frame(
            0, analysis: makeAnalysis(overrides: unknownOverrides),
            reference: reference, height: 640)
        XCTAssertEqual(try joint(.leftHip, in: unknownFrame).severity ?? -1, 0)
        XCTAssertEqual(try joint(.leftHip, in: unknownFrame).band, .ok)
    }

    func testAFeatureTheReferenceDoesNotCarryGivesNoSeverity() throws {
        // A partial reference (as #28's may well be) scores only what it
        // carries: the shoulder is not judged, and not condemned either.
        let analysis = makeAnalysis(overrides: ["shoulder_angle": column(120, 4)])
        let reference = scoreReference(["hip_angle": (mean: 176, sd: 3)])

        let frame = StressDiagram.frame(0, analysis: analysis, reference: reference, height: 640)

        let shoulder = try joint(.leftShoulder, in: frame)
        XCTAssertNil(shoulder.severity)
        XCTAssertEqual(shoulder.band, .ok)
    }

    // MARK: - Bones

    func testABoneTakesTheWorseOfItsTwoJoints() throws {
        // Hip: 140° -> fails PIKE_HIP_DEG -> severity 1, bad. Shoulder:
        // 170° passes OPEN_SHOULDER_DEG -> severity 0, ok. The bone is the
        // hip's number, not an average of the two.
        let analysis = makeAnalysis(
            overrides: ["hip_angle": column(140, 4), "shoulder_angle": column(170, 4)])

        let frame = StressDiagram.frame(0, analysis: analysis, reference: nil, height: 640)

        let shoulderHip = try bone(.leftShoulder, .leftHip, in: frame)
        XCTAssertEqual(shoulderHip.severity ?? -1, 1.0, accuracy: 1e-9)
        XCTAssertEqual(shoulderHip.band, .bad)
        XCTAssertEqual(try joint(.leftShoulder, in: frame).band, .ok)

        // Two joints that each pass their target make a passing bone (both
        // with a number, not an unmeasured nil).
        let clean = StressDiagram.frame(
            0, analysis: makeAnalysis(
                overrides: ["hip_angle": column(172, 4), "knee_angle": column(170, 4)]),
            reference: nil, height: 640)
        let cleanBone = try bone(.leftHip, .leftKnee, in: clean)
        XCTAssertEqual(cleanBone.severity ?? -1, 0)
        XCTAssertEqual(cleanBone.band, .ok)
    }

    func testTheSkeletonHasEveryBoneOfTheTableAndAHeadLine() throws {
        let frame = StressDiagram.frame(0, analysis: makeAnalysis(), reference: nil, height: 640)

        for (from, to) in StressDiagram.bones {
            _ = try bone(from, to, in: frame)
        }
        // The schema has no shoulder-midpoint joint, so the head line is
        // drawn from the nose to the *nearer* shoulder — the nose sits 10 px
        // left of the column, so that is the left one.
        let head = try bone(.nose, .leftShoulder, in: frame)
        XCTAssertEqual(head.from, .nose)
        XCTAssertNil(
            frame.bones.first { $0.from == .nose && $0.to == .rightShoulder },
            "only one head line, to the nearer shoulder"
        )
    }

    func testAJointNobodyValidatedIsNotDrawnAndTakesItsBonesWithIt() throws {
        let analysis = makeAnalysis(invalidJoints: [.leftKnee])

        let frame = StressDiagram.frame(0, analysis: analysis, reference: nil, height: 640)

        XCTAssertNil(frame.joints.first { $0.joint == .leftKnee })
        XCTAssertNil(
            frame.bones.first {
                ($0.from == .leftHip && $0.to == .leftKnee)
                    || ($0.from == .leftKnee && $0.to == .leftHip)
            },
            "a bone needs both joints to exist")
        XCTAssertNil(
            frame.bones.first {
                ($0.from == .leftKnee && $0.to == .leftAnkle)
                    || ($0.from == .leftAnkle && $0.to == .leftKnee)
            })
        // The rest of the skeleton is untouched: the arm, the hip line and
        // the other leg all keep their bones.
        XCTAssertNoThrow(try bone(.leftShoulder, .leftHip, in: frame))
        XCTAssertNoThrow(try bone(.rightHip, .rightKnee, in: frame))
        XCTAssertNoThrow(try bone(.leftShoulder, .rightShoulder, in: frame))
    }

    // MARK: - The tables

    func testTheJointFeatureTableIsTheDocumentedOne() {
        XCTAssertEqual(StressDiagram.jointFeatures[.nose], ["head"])
        XCTAssertEqual(
            StressDiagram.jointFeatures[.leftShoulder], ["shoulder_angle", "off_shoulder"])
        XCTAssertEqual(
            StressDiagram.jointFeatures[.rightShoulder], ["shoulder_angle", "off_shoulder"])
        XCTAssertEqual(StressDiagram.jointFeatures[.leftElbow], ["elbow_angle"])
        XCTAssertEqual(StressDiagram.jointFeatures[.rightElbow], ["elbow_angle"])
        // The wrists are the origin of every offset: nothing can be off.
        XCTAssertTrue(StressDiagram.jointFeatures[.leftWrist]?.isEmpty ?? false)
        XCTAssertTrue(StressDiagram.jointFeatures[.rightWrist]?.isEmpty ?? false)
        XCTAssertEqual(
            StressDiagram.jointFeatures[.leftHip], ["hip_angle", "off_hip", "banana"])
        XCTAssertEqual(
            StressDiagram.jointFeatures[.rightHip], ["hip_angle", "off_hip", "banana"])
        XCTAssertEqual(StressDiagram.jointFeatures[.leftKnee], ["knee_angle", "off_knee"])
        XCTAssertEqual(StressDiagram.jointFeatures[.rightKnee], ["knee_angle", "off_knee"])
        for ankle in [Joint.leftAnkle, .rightAnkle, .leftFootIndex, .rightFootIndex] {
            XCTAssertEqual(
                StressDiagram.jointFeatures[ankle], ["off_ankle", "leg_separation"],
                "\(ankle.rawValue) reads the ankle's features")
        }
        // Every joint of the schema has a row, and every row names real
        // features: the table is the whole story, nothing is read ad hoc.
        XCTAssertEqual(Set(StressDiagram.jointFeatures.keys), Set(Joint.allCases))
        for features in StressDiagram.jointFeatures.values {
            for name in features {
                XCTAssertNotNil(
                    Features.all.first { $0.name == name }, "\(name) is not a feature")
            }
        }
    }

    /// `TOLERANCES` of `pipeline/handstand/features.py`, row for row: same
    /// features, same order, same thresholds, same sides.
    func testTheToleranceTableEqualsPythons() {
        let expected: [(feature: String, threshold: Double, side: FeatureSide)] = [
            ("line_deviation", 0.1, .either),  // LINE_TOLERANCE_L
            ("body_angle", 10.0, .either),  // BODY_ANGLE_TOL_DEG
            ("shoulder_angle", 160.0, .below),  // OPEN_SHOULDER_DEG
            ("hip_angle", 165.0, .below),  // PIKE_HIP_DEG
            ("knee_angle", 165.0, .below),  // STRAIGHT_KNEE_DEG
            ("elbow_angle", 160.0, .below),  // BENT_ELBOW_DEG
            ("banana", 0.1, .either),  // BANANA_TOLERANCE_L
            ("head", 60.0, .either),  // HEAD_FLEXION_DEG
            ("leg_separation", 30.0, .above),  // SPLIT_DEG
        ]

        XCTAssertEqual(FeatureTolerances.all.count, expected.count)
        for (index, row) in expected.enumerated() {
            let tolerance = FeatureTolerances.all[index]
            XCTAssertEqual(tolerance.feature, row.feature, "row \(index)")
            XCTAssertEqual(tolerance.threshold, row.threshold, accuracy: 0, "row \(index)")
            XCTAssertEqual(tolerance.side, row.side, "row \(index)")
        }
        XCTAssertEqual(
            Set(FeatureTolerances.all.map(\.side)), Set(FeatureSide.allCases),
            "all three sides are used, as Python's test asserts")
    }

    /// `fails_tolerance`'s four rules, the same cases
    /// `pipeline/tests/test_features.py::test_the_tolerance_tests_ask_the_right_side_of_each_target`
    /// asks — including "an unmeasured hold fails nothing".
    func testFailsToleranceAsksTheRightSideOfEachTarget() {
        XCTAssertTrue(FeatureTolerances.fails(0.2, threshold: 0.1, side: .either))
        XCTAssertTrue(FeatureTolerances.fails(-0.2, threshold: 0.1, side: .either))
        XCTAssertFalse(FeatureTolerances.fails(0.05, threshold: 0.1, side: .either))
        XCTAssertTrue(FeatureTolerances.fails(150, threshold: 165, side: .below))
        XCTAssertFalse(FeatureTolerances.fails(180, threshold: 165, side: .below))
        XCTAssertTrue(FeatureTolerances.fails(40, threshold: 30, side: .above))
        XCTAssertFalse(FeatureTolerances.fails(40, threshold: 30, side: .below))
        XCTAssertFalse(FeatureTolerances.fails(.nan, threshold: 0.1, side: .either))
    }

    func testTheSeverityBandBoundariesAreTheOnesTheDiagramDocuments() {
        XCTAssertEqual(Severity.colourBand(0), .ok)
        XCTAssertEqual(Severity.colourBand(0.2499), .ok)
        XCTAssertEqual(Severity.colourBand(0.25), .warn)
        XCTAssertEqual(Severity.colourBand(0.4999), .warn)
        XCTAssertEqual(Severity.colourBand(0.5), .bad)
        XCTAssertEqual(Severity.colourBand(1), .bad)
    }

    // MARK: - Playback time

    func testFrameIndexFindsTheLastFrameAtOrBeforeThePlayhead() {
        let stamps = Self.stamps  // 0, 33, 66, 99

        XCTAssertNil(StressDiagram.frameIndex(atMs: -1, in: stamps), "before the first frame")
        XCTAssertEqual(StressDiagram.frameIndex(atMs: 0, in: stamps), 0, "exactly the first")
        XCTAssertEqual(StressDiagram.frameIndex(atMs: 50, in: stamps), 1, "between two frames")
        XCTAssertEqual(StressDiagram.frameIndex(atMs: 99, in: stamps), 3, "exactly the last")
        XCTAssertEqual(
            StressDiagram.frameIndex(atMs: 5_000, in: stamps), 3, "after the last stays there")
        XCTAssertNil(StressDiagram.frameIndex(atMs: 0, in: []), "an empty clip has no frames")
    }

    // MARK: - The centre of mass

    func testTheCentreOfMassRoundTripsThroughTheBodyFrameInverse() throws {
        // com_u / com_v are body-frame columns; the diagram places them back
        // where the features measured them: wrist midpoint (200, 500),
        // L = 300 px, (0.05, 0.7) -> (215, 290).
        let analysis = makeAnalysis(
            overrides: ["com_u": column(0.05, 4), "com_v": column(0.7, 4)],
            balanceZone: [BalanceZone?](
                repeating: BalanceZone.ok, count: 4)
        )

        let frame = StressDiagram.frame(2, analysis: analysis, reference: nil, height: 640)

        let com = try XCTUnwrap(frame.com)
        XCTAssertEqual(com.x, 215, accuracy: 1e-9)
        XCTAssertEqual(com.y, 290, accuracy: 1e-9)
        // comFloor is the same point on the hand line (v = 0): same column,
        // the wrist's own height.
        let floorPoint = try XCTUnwrap(frame.comFloor)
        XCTAssertEqual(floorPoint.x, com.x, accuracy: 1e-9)
        XCTAssertEqual(floorPoint.y, Self.wristY, accuracy: 1e-9)
        // The inverse of the inverse: back to the body frame it came from.
        let uv = try BodyFrame.toBodyFrame(
            x: com.x, y: com.y, wristMidX: Self.wristX, wristMidY: Self.wristY,
            bodyLength: Self.length)
        XCTAssertEqual(uv.u, 0.05, accuracy: 1e-12)
        XCTAssertEqual(uv.v, 0.7, accuracy: 1e-12)
        // The balance zone comes straight from the features.
        XCTAssertEqual(frame.balanceZone, .ok)
    }

    func testThereIsNoCentreOfMassWhenTheColumnsWereNotMeasured() throws {
        // Default clip: com_u / com_v are NaN, as on a frame nobody saw.
        let frame = StressDiagram.frame(0, analysis: makeAnalysis(), reference: nil, height: 640)

        XCTAssertNil(frame.com)
        XCTAssertNil(frame.comFloor)
        XCTAssertNil(frame.balanceZone)
    }

    // MARK: - The extension point for #28

    func testTheIdealSkeletonIsAlwaysNilUntilIssue28ProvidesOne() {
        for ref: ScoreReference? in [nil, scoreReference(["hip_angle": (mean: 176, sd: 3)])] {
            for phase: Phase in [.hold, .pre] {
                let frame = StressDiagram.frame(
                    0, analysis: makeAnalysis(phase: phase),
                    reference: ref, height: 640)
                XCTAssertNil(frame.ideal, "the ghost skeleton is chainlink #28's to provide")
            }
        }
    }

    // MARK: - The frame identity

    func testTheDiagramCarriesTheFramesOwnTimestamp() {
        let frame = StressDiagram.frame(2, analysis: makeAnalysis(), reference: nil, height: 640)
        XCTAssertEqual(frame.tMs, 66, "the clip's own clock, not a frame number")
    }
}
