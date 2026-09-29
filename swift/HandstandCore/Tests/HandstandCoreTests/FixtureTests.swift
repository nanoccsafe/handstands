import XCTest

@testable import HandstandCore

/// The fixture test: a hand-written JSON file of a few `PoseFrame`s, read
/// through `Bundle.module`.
///
/// It proves the resource plumbing the parity fixtures of chainlink #25/#42
/// will ride on — and that `PoseFrame`'s hand-written Codable really does
/// decode a JSON object keyed by joint names, rather than the array a
/// synthesized `[Joint: Keypoint]` conformance would produce.
final class FixtureTests: XCTestCase {
    private func fixtureData() throws -> Data {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "tiny_frames", withExtension: "json", subdirectory: "Fixtures"),
            "Tests/HandstandCoreTests/Fixtures/tiny_frames.json is missing from the bundle"
        )
        return try Data(contentsOf: url)
    }

    private func fixtureFrames() throws -> [PoseFrame] {
        try JSONDecoder().decode([PoseFrame].self, from: fixtureData())
    }

    func testFixtureDecodesAsPoseFrames() throws {
        let frames = try fixtureFrames()
        XCTAssertEqual(frames.count, 3)
        XCTAssertEqual(frames.map(\.frameIndex), [0, 1, 2])
        XCTAssertEqual(frames.map(\.tMs), [0, 33, 66])
    }

    func testFixtureKeepsTheHandWrittenValues() throws {
        let first = try fixtureFrames()[0]

        let nose = try XCTUnwrap(first.joints[.nose])
        XCTAssertEqual(nose.x, 100.5, accuracy: 1e-12)
        XCTAssertEqual(nose.y, 40.25, accuracy: 1e-12)
        XCTAssertEqual(nose.visibility, 0.98, accuracy: 1e-12)

        let leftWrist = try XCTUnwrap(first.joints[.leftWrist])
        XCTAssertEqual(leftWrist.x, 70, accuracy: 0)
        XCTAssertEqual(leftWrist.y, 200, accuracy: 0)

        // The joint names in the file are the Python spellings...
        XCTAssertNil(Joint(rawValue: "left_eye"), "eyes are not in the tracked schema")
        XCTAssertEqual(first.joints.count, 5)
    }

    func testFixtureMidpointAndBodyFrame() throws {
        let first = try fixtureFrames()[0]

        // Wrist midpoint — the body frame's origin.
        let origin = try XCTUnwrap(first.midpoint(.leftWrist, .rightWrist))
        XCTAssertEqual(origin.x, 100, accuracy: 0)
        XCTAssertEqual(origin.y, 200, accuracy: 0)
        XCTAssertEqual(origin.visibility, 0.94, accuracy: 1e-12)

        // A point one and a half-ish body lengths above the hands: v > 0.
        let nose = try XCTUnwrap(first.joints[.nose])
        let uv = try BodyFrame.toBodyFrame(
            x: nose.x,
            y: nose.y,
            wristMidX: origin.x,
            wristMidY: origin.y,
            bodyLength: 100
        )
        XCTAssertEqual(uv.u, 0.005, accuracy: 1e-12)
        XCTAssertEqual(uv.v, 1.5975, accuracy: 1e-12)
        XCTAssertGreaterThan(uv.v, 0)
    }

    func testFixtureFrameIsNotValidBecauseItCarriesFiveJointsOfFifteen() throws {
        let first = try fixtureFrames()[0]
        XCTAssertEqual(first.joints.count, 5)
        XCTAssertFalse(first.isValid(minVisibility: 0.5))
    }

    func testPoseFrameRoundTripsThroughJSON() throws {
        let original = try fixtureFrames()[1]
        let encoded = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(PoseFrame.self, from: encoded)
        XCTAssertEqual(decoded, original)

        // And the object really is keyed by joint name...
        let object = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: encoded) as? [String: Any]
        )
        let joints = try XCTUnwrap(object["joints"] as? [String: Any])
        XCTAssertNotNil(joints["left_shoulder"])
        XCTAssertNotNil(joints["nose"])
        XCTAssertEqual(joints.count, original.joints.count)
    }

    func testUnknownJointNameFailsToDecode() throws {
        let json = """
        {"frameIndex": 0, "tMs": 0, "joints": {"neck": {"x": 1, "y": 2, "visibility": 1}}}
        """
        XCTAssertThrowsError(try JSONDecoder().decode(PoseFrame.self, from: Data(json.utf8))) { error in
            XCTAssertTrue("\(error)".contains("unknown joint 'neck'"), "\(error)")
        }
    }
}
