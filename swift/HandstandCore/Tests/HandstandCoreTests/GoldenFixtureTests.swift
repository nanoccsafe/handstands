import XCTest

import HandstandCore

/// The golden parity fixtures of chainlink #25 — a smoke test, not the parity
/// check itself.
///
/// `pipeline/handstand/golden.py` writes one JSON file per synthetic case into
/// `Fixtures/golden/` (its README documents the format); this test loads every
/// one of them through `Bundle.module` and checks that it **decodes** and that
/// its rows line up: as many post-process, phase and feature rows as there are
/// input frames, `valid`/`filled` arrays the length of `meta.joint_names`, one
/// hold-summary row per hold.
///
/// The ports of post-process (#39, whose parity check is
/// `PostProcessTests.testEveryGoldenFixtureMatchesThePythonPostProcess`),
/// phases + features (#40) and the scorer (#41) decode the same files and
/// compare their own numbers against `expected` at `meta.tolerances`; what is
/// asserted here is that the fixture contract they are written against still
/// holds — every joint name is one this package knows, every feature column is
/// the documented one, and nothing derived from a real video has found its way
/// into a public tree.
final class GoldenFixtureTests: XCTestCase {
    // MARK: - The fixture schema, as Swift sees it
    //
    // The decoding lives in `GoldenFixtures.swift`, shared with the parity
    // tests of the ports (#39 and later); these aliases keep the test bodies
    // below reading the way they did when the types were declared here.

    typealias Meta = GoldenFixtures.Meta
    typealias InputFrame = GoldenFixtures.InputFrame
    typealias PostFrame = GoldenFixtures.PostFrame
    typealias BodyLength = GoldenFixtures.BodyLength
    typealias PhaseRow = GoldenFixtures.PhaseRow
    typealias Value = GoldenFixtures.Value
    typealias Expected = GoldenFixtures.Expected
    typealias Fixture = GoldenFixtures.Fixture

    // MARK: - What the fixtures promise

    /// The five synthetic cases of `golden.CASES`.
    static let expectedCases: [String] = [
        "line_hold",
        "banana_hold",
        "hand_step",
        "gaps_and_noise",
        "trainer_contact",
    ]

    /// `golden.FEATURE_COLUMNS`: the features the ports carry, in order. The
    /// list is duplicated here on purpose — a column renamed on either side
    /// must fail this test rather than quietly agree with itself.
    static let expectedColumns: [String] = [
        "off_shoulder",
        "off_hip",
        "off_knee",
        "off_ankle",
        "line_deviation",
        "body_angle",
        "shoulder_angle",
        "hip_angle",
        "knee_angle",
        "elbow_angle",
        "banana",
        "head",
        "leg_separation",
        "com_u",
        "com_v",
        "com_forward",
        "facing_sign",
        "balance_zone",
    ]

    // MARK: - Loading

    /// Every committed fixture, in file-name order so a failure names a stable list.
    private func goldenURLs() throws -> [URL] {
        let urls = GoldenFixtures.urls()
        XCTAssertFalse(urls.isEmpty, "Fixtures/golden/*.json is missing from Bundle.module")
        return urls
    }

    private func url(for name: String) throws -> URL {
        try XCTUnwrap(
            GoldenFixtures.urls().first { $0.lastPathComponent == "\(name).json" },
            "\(name).json is missing from Bundle.module"
        )
    }

    private func load(_ url: URL) throws -> Fixture {
        try GoldenFixtures.load(url)
    }

    // MARK: - Tests

    /// The smoke test the ports start from: the file parses, and every row
    /// array is exactly as long as the input it describes.
    func testEveryGoldenFixtureDecodesAndItsRowsLineUp() throws {
        var seen: [String] = []
        for url in try goldenURLs() {
            let fixture = try load(url)
            let name = url.deletingPathExtension().lastPathComponent
            seen.append(name)
            XCTAssertEqual(fixture.meta.caseName, name, "\(name): meta.case is not the file name")
            XCTAssertFalse(fixture.input.isEmpty, name)

            let frames = fixture.input.count
            XCTAssertEqual(fixture.meta.frameCount, frames, "\(name): meta.frame_count")
            XCTAssertEqual(
                fixture.expected.postprocess.frames.count, frames, "\(name): postprocess rows")
            XCTAssertEqual(fixture.expected.phases.count, frames, "\(name): phase rows")
            XCTAssertEqual(fixture.expected.features.count, frames, "\(name): feature rows")

            for (index, frame) in fixture.input.enumerated() {
                XCTAssertEqual(frame.frameIdx, index, "\(name): frame_idx must count up")
                if index > 0 {
                    XCTAssertGreaterThan(
                        frame.tMs, fixture.input[index - 1].tMs, "\(name): t_ms must increase")
                }
                XCTAssertEqual(
                    Set(frame.joints.keys), Set(fixture.meta.jointNames),
                    "\(name): frame \(index) does not hold the declared joints")
            }

            let joints = fixture.meta.jointNames.count
            for (index, row) in fixture.expected.postprocess.frames.enumerated() {
                XCTAssertEqual(row.valid.count, joints, "\(name): frame \(index) valid")
                XCTAssertEqual(row.filled.count, joints, "\(name): frame \(index) filled")
                XCTAssertEqual(
                    Set(row.joints.keys), Set(fixture.meta.jointNames),
                    "\(name): frame \(index) joints")
                for key in fixture.meta.jointNames {
                    let pair = try XCTUnwrap(
                        row.joints[key], "\(name): frame \(index) has no \(key)")
                    XCTAssertEqual(pair.count, 2, "\(name): frame \(index) \(key) is not [x, y]")
                }
            }

            let holds = Set(fixture.expected.phases.map(\.holdId).filter { $0 >= 0 })
            XCTAssertEqual(holds.count, fixture.meta.holdCount, "\(name): hold ids")
            XCTAssertEqual(
                fixture.expected.holdSummary.count, fixture.meta.holdCount,
                "\(name): hold summary rows")

            // The scorer's answers (#41) decode as the document README says:
            // one score per hold, and the clip's own hold beside them.
            guard case .object(let score) = fixture.expected.score else {
                XCTFail("\(name): expected.score is not an object")
                continue
            }
            guard case .array(let scored)? = score["holds"] else {
                XCTFail("\(name): expected.score has no holds array")
                continue
            }
            XCTAssertEqual(scored.count, fixture.meta.holdCount, "\(name): score rows")
            XCTAssertNotNil(score["clip_hold_id"], "\(name): clip_hold_id is missing")
        }
        XCTAssertEqual(Set(seen), Set(Self.expectedCases), "the five committed cases")
    }

    /// The joint names in a fixture are the ones this package has a case for,
    /// so a port can key its own arrays by `Joint` instead of by string.
    func testJointNamesAreTheSharedSchemaSwiftKnows() throws {
        let schema = Joint.allCases.map(\.rawValue)
        for url in try goldenURLs() {
            let fixture = try load(url)
            XCTAssertEqual(
                fixture.meta.jointNames, schema, "\(url.lastPathComponent): joint schema")
        }
    }

    func testFeatureColumnsAreTheDocumentedList() throws {
        for url in try goldenURLs() {
            let fixture = try load(url)
            let name = url.lastPathComponent
            XCTAssertEqual(fixture.meta.featureColumns, Self.expectedColumns, name)
            for (index, row) in fixture.expected.features.enumerated() {
                XCTAssertEqual(
                    Set(row.keys), Set(Self.expectedColumns), "\(name): feature row \(index)")
            }
            XCTAssertFalse(fixture.meta.tolerances.isEmpty, "\(name): no tolerance to compare at")
        }
    }

    /// The repo is public: everything under `swift/` is synthetic, produced from
    /// a seed. Real-clip fixtures only ever land in the git-ignored data tree
    /// (`golden.require_outside_repo` is what enforces it on the Python side).
    func testEveryFixtureUnderSwiftIsSynthetic() throws {
        for url in try goldenURLs() {
            let fixture = try load(url)
            XCTAssertEqual(
                fixture.meta.mode, "synthetic",
                "\(url.lastPathComponent): real keypoints must never be committed")
            XCTAssertNotNil(fixture.meta.seed, "\(url.lastPathComponent): synthetic needs a seed")
        }
    }

    /// The two cases whose *reason to exist* is visible in the stored answers.
    func testHandStepHasTwoHoldsAndTrainerContactFramesAreInvalid() throws {
        let stepped = try load(try url(for: "hand_step"))
        XCTAssertEqual(stepped.meta.holdCount, 2, "hand_step must split the hold in two")
        XCTAssertEqual(
            Set(stepped.expected.phases.map(\.holdId).filter { $0 >= 0 }), [0, 1],
            "the two holds are numbered 0 and 1"
        )

        let contact = try load(try url(for: "trainer_contact"))
        let flagged = contact.input.indices.filter { contact.input[$0].trainerContact }
        XCTAssertFalse(flagged.isEmpty, "the trainer contact case has no contact frames")
        for index in flagged {
            let row = contact.expected.postprocess.frames[index]
            XCTAssertFalse(
                row.valid.contains(true), "frame \(index) is trainer contact but has a position")
        }
    }

    /// The values are the kinds of values the format says: numbers where a
    /// number belongs, and the balance zone spelled as words.
    func testFeaturesCarryNumbersAndBalanceZones() throws {
        let fixture = try load(try url(for: "line_hold"))
        var numbers = 0
        var zones = 0
        for row in fixture.expected.features {
            let angle = row["body_angle"] ?? Value.missing
            let zone = row["balance_zone"] ?? Value.missing
            if case .number = angle {
                numbers += 1
            }
            if case .text = zone {
                zones += 1
            }
        }
        XCTAssertGreaterThan(numbers, 0, "body_angle is a number on every measurable frame")
        XCTAssertGreaterThan(zones, 0, "balance_zone is a label")
    }
}
