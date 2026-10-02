import XCTest

@testable import HandstandCore

/// The parity check for `OrientationChooser` (chainlink #45): the Swift port
/// of `handstand.pose_mediapipe.choose_orientations` must answer **exactly**
/// what Python answered, frame by frame, on every synthetic case of
/// `Fixtures/golden/orientation_cases.json` — the file
/// `pipeline/handstand/orientation_cases.py` writes by running the real
/// Python chooser. Parity only, no real clips (chainlink #80).
///
/// The cases are the behaviours the choice is defined by: a steady upright
/// clip, a steady inverted clip, a one-frame confident misread, a sustained
/// switch, NaN gaps and variable frame-rate timestamps. On top of the file's
/// answers, the constants themselves are pinned twice — against Python's
/// values as the generator recorded them, and against the literal numbers —
/// so neither side can drift unnoticed.
final class OrientationChooserTests: XCTestCase {
    // MARK: - The fixture

    /// One case of `orientation_cases.json`: the input score sequences and
    /// Python's own `rotated` flags. `null` is the schema's "no score"
    /// (NaN) — same as everywhere else in `Fixtures/golden`.
    struct Case: Decodable {
        let name: String
        let tMs: [Double]
        let scoreUpright: [Double?]
        let scoreRotated: [Double?]
        let rotated: [Bool]

        enum CodingKeys: String, CodingKey {
            case name
            case tMs = "t_ms"
            case scoreUpright = "score_upright"
            case scoreRotated = "score_rotated"
            case rotated
        }
    }

    struct Fixture: Decodable {
        struct Meta: Decodable {
            let mode: String
            let generatorVersion: String
            let orientWindowS: Double
            let orientMargin: Double
            let orientMinSwitchS: Double
            let caseNames: [String]

            enum CodingKeys: String, CodingKey {
                case mode
                case generatorVersion = "generator_version"
                case orientWindowS = "orient_window_s"
                case orientMargin = "orient_margin"
                case orientMinSwitchS = "orient_min_switch_s"
                case caseNames = "case_names"
            }
        }

        let meta: Meta
        let cases: [Case]
    }

    private func loadFixture() throws -> Fixture {
        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: GoldenFixtures.orientationCasesName,
                withExtension: "json",
                subdirectory: "Fixtures/golden"),
            "Fixtures/golden/orientation_cases.json is missing from Bundle.module"
        )
        let decoder = JSONDecoder()
        return try decoder.decode(Fixture.self, from: Data(contentsOf: url))
    }

    /// `null` (no score) in, NaN out — what the chooser reads.
    private func scores(_ values: [Double?]) -> [Double] {
        values.map { $0 ?? Double.nan }
    }

    // MARK: - The parity itself

    func testEveryCaseMatchesThePythonAnswerFrameByFrame() throws {
        let fixture = try loadFixture()
        for testCase in fixture.cases {
            let chosen = OrientationChooser.chooseOrientations(
                tMs: testCase.tMs,
                scoreUpright: scores(testCase.scoreUpright),
                scoreRotated: scores(testCase.scoreRotated)
            )
            XCTAssertEqual(
                chosen, testCase.rotated,
                "\(testCase.name): the chooser disagreed with Python"
            )
        }
    }

    func testTheFixtureCoversEveryCaseTheGeneratorPromised() throws {
        let fixture = try loadFixture()
        XCTAssertEqual(
            fixture.cases.map(\.name), fixture.meta.caseNames,
            "the case list in meta must be the cases in the file"
        )
        XCTAssertEqual(
            Set(fixture.cases.map(\.name)),
            [
                "steady_upright", "steady_inverted", "one_frame_misread",
                "sustained_switch", "nan_gaps", "vfr_timestamps",
            ],
            "the six behaviours chainlink #45 lists"
        )
        for testCase in fixture.cases {
            XCTAssertEqual(testCase.rotated.count, testCase.tMs.count, testCase.name)
            XCTAssertFalse(testCase.tMs.isEmpty, testCase.name)
        }
        XCTAssertEqual(fixture.meta.mode, "synthetic", "parity checks stay synthetic (#80)")
    }

    // MARK: - The constants, pinned twice

    func testTheConstantsAreTheOnesThePipelineUses() throws {
        // The literal values of pose_mediapipe.py…
        XCTAssertEqual(OrientationChooser.windowS, 0.5)
        XCTAssertEqual(OrientationChooser.margin, 0.05)
        XCTAssertEqual(OrientationChooser.minSwitchS, 0.3)
        // …and the same numbers the Python generator recorded in the fixture,
        // so a changed constant on either side fails here.
        let fixture = try loadFixture()
        XCTAssertEqual(fixture.meta.orientWindowS, OrientationChooser.windowS)
        XCTAssertEqual(fixture.meta.orientMargin, OrientationChooser.margin)
        XCTAssertEqual(fixture.meta.orientMinSwitchS, OrientationChooser.minSwitchS)
    }

    // MARK: - Variable frame rate, read as clip time

    func testTheWindowIsMeasuredInTimeNotInFrames() {
        // The smooth_margin window test of test_pose_mediapipe.py: a spike
        // one fifth of a five-frame window away at 100 ms a frame is
        // averaged over five frames; at 20 ms a frame the same window
        // reaches every frame. Same numbers, same answers as Python.
        let slow = OrientationChooser.smoothMargin(
            tMs: [0, 200, 400], margin: [0.0, 1.0, 0.0])
        XCTAssertEqual(slow[0], 0.5, accuracy: 1e-12)
        XCTAssertEqual(slow[1], 1.0 / 3.0, accuracy: 1e-12)
        XCTAssertEqual(slow[2], 0.5, accuracy: 1e-12)

        let fast = OrientationChooser.smoothMargin(
            tMs: [0, 20, 40], margin: [0.0, 1.0, 0.0])
        XCTAssertEqual(fast[0], 1.0 / 3.0, accuracy: 1e-12)
        XCTAssertEqual(fast[1], 1.0 / 3.0, accuracy: 1e-12)
        XCTAssertEqual(fast[2], 1.0 / 3.0, accuracy: 1e-12)
    }

    func testASustainedPreferenceSwitchesOnTheFrameTheTimeRanOut() {
        // test_choose_orientations_starts_from_the_first_half_second's
        // numbers: upright a second, then confidently rotated — the switch
        // lands on frame 13 of 40 at 100 ms a frame, not a frame later.
        let times = (0..<40).map { Double($0 * 100) }
        let chosen = OrientationChooser.chooseOrientations(
            tMs: times,
            scoreUpright: [Double](repeating: 0.9, count: 40),
            scoreRotated: [Double](repeating: 0.9, count: 10)
                + [Double](repeating: 0.99, count: 30)
        )
        XCTAssertEqual(chosen, [Bool](repeating: false, count: 13)
            + [Bool](repeating: true, count: 27))
    }

    func testAnEmptyClipHasNoChoices() {
        XCTAssertTrue(
            OrientationChooser.chooseOrientations(tMs: [], scoreUpright: [], scoreRotated: [])
                .isEmpty)
        XCTAssertTrue(OrientationChooser.smoothMargin(tMs: [], margin: []).isEmpty)
        XCTAssertEqual(OrientationChooser.frameMargin(scoreUpright: [], scoreRotated: []), [])
    }

    func testAnUnscoredFrameHasNoMarginAtAll() {
        let margin = OrientationChooser.frameMargin(
            scoreUpright: [0.9, .nan, 0.9],
            scoreRotated: [0.4, 0.4, .nan]
        )
        XCTAssertEqual(margin[0], -0.5, accuracy: 1e-12)
        XCTAssertTrue(margin[1].isNaN, "a frame nobody was found in is NaN, not ±inf")
        XCTAssertTrue(margin[2].isNaN)
        // `+inf - inf` and friends also come out NaN, never a number.
        XCTAssertTrue(
            OrientationChooser.frameMargin(scoreUpright: [.infinity], scoreRotated: [0.4])[0]
                .isNaN)
    }

    // MARK: - The 33 -> 15 landmark map

    func testEverySchemaJointMapsToItsMediaPipeLandmark() {
        // The table, spelled out independently: pose_mediapipe.JOINT_NAMES
        // indices of the 15 tracked joints.
        XCTAssertEqual(mediaPipeLandmarkIndex.count, Joint.allCases.count)
        let expected: [(Joint, Int)] = [
            (.nose, 0),
            (.leftShoulder, 11), (.rightShoulder, 12),
            (.leftElbow, 13), (.rightElbow, 14),
            (.leftWrist, 15), (.rightWrist, 16),
            (.leftHip, 23), (.rightHip, 24),
            (.leftKnee, 25), (.rightKnee, 26),
            (.leftAnkle, 27), (.rightAnkle, 28),
            (.leftFootIndex, 31), (.rightFootIndex, 32),
        ]
        for (joint, index) in expected {
            XCTAssertEqual(mediaPipeLandmarkIndex[joint], index, joint.rawValue)
            XCTAssertTrue((0..<OrientationChooser.landmarkCount).contains(index), joint.rawValue)
        }
        // The main-joint score runs over the 12 body joints only — no face,
        // no fingers — in MAIN_JOINTS' order.
        XCTAssertEqual(OrientationChooser.mainJointIndices, [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28])
        XCTAssertEqual(OrientationChooser.mainJointIndices.count, 12)
        XCTAssertEqual(OrientationChooser.landmarkCount, 33)
    }
}
