import XCTest

@testable import HandstandCore

/// The framing guide's cases, one test per verdict the task pins down, plus
/// the sentences the Record screen shows for each. Every body is built from
/// normalised coordinates (0...1, origin top-left, y down) by hand — no
/// camera, no Vision, no video file.
final class FramingCheckTests: XCTestCase {
    // MARK: - Bodies

    private func joint(_ x: Double, _ y: Double, confidence: Double = 1.0) -> Keypoint {
        Keypoint(x: x, y: y, visibility: confidence)
    }

    /// A whole standing body: nose at the top, ankles near the bottom, all
    /// joints confidently detected and comfortably away from every edge —
    /// the frame `docs/recording_protocol.md` asks for.
    private func standingBody(centreX: Double = 0.5, top: Double = 0.10, bottom: Double = 0.88) -> [Joint: Keypoint] {
        let span = bottom - top
        return [
            .nose: joint(centreX, top),
            .leftShoulder: joint(centreX - 0.05, top + span * 0.14),
            .rightShoulder: joint(centreX + 0.05, top + span * 0.14),
            .leftElbow: joint(centreX - 0.06, top + span * 0.28),
            .rightElbow: joint(centreX + 0.06, top + span * 0.28),
            .leftWrist: joint(centreX - 0.06, top + span * 0.42),
            .rightWrist: joint(centreX + 0.06, top + span * 0.42),
            .leftHip: joint(centreX - 0.04, top + span * 0.55),
            .rightHip: joint(centreX + 0.04, top + span * 0.55),
            .leftKnee: joint(centreX - 0.04, top + span * 0.78),
            .rightKnee: joint(centreX + 0.04, top + span * 0.78),
            .leftAnkle: joint(centreX - 0.04, bottom),
            .rightAnkle: joint(centreX + 0.04, bottom),
        ]
    }

    /// The same body upside down: feet at the top, hands near the bottom — a
    /// handstand, which is the frame this app exists for.
    private func invertedBody(ankleY: Double = 0.10, wristY: Double = 0.84) -> [Joint: Keypoint] {
        [
            .nose: joint(0.5, wristY + 0.04),
            .leftShoulder: joint(0.45, wristY - 0.20),
            .rightShoulder: joint(0.55, wristY - 0.20),
            .leftElbow: joint(0.44, wristY - 0.10),
            .rightElbow: joint(0.56, wristY - 0.10),
            .leftWrist: joint(0.44, wristY),
            .rightWrist: joint(0.56, wristY),
            .leftHip: joint(0.46, ankleY + 0.34),
            .rightHip: joint(0.54, ankleY + 0.34),
            .leftKnee: joint(0.46, ankleY + 0.15),
            .rightKnee: joint(0.54, ankleY + 0.15),
            .leftAnkle: joint(0.46, ankleY),
            .rightAnkle: joint(0.54, ankleY),
        ]
    }

    /// A body whole but far away: the whole thing fits between y 0.40 and
    /// y 0.65, a quarter of the frame's height.
    private func distantBody() -> [Joint: Keypoint] {
        var body = standingBody(centreX: 0.5, top: 0.40, bottom: 0.65)
        body[.nose] = joint(0.5, 0.40)
        return body
    }

    // MARK: - The verdicts

    func testWholeBodyInsideIsOk() {
        XCTAssertEqual(FramingCheck.status(people: [standingBody()]), .ok)
    }

    func testInvertedBodyWhollyInsideIsOk() {
        // A handstand is read the same way up as a stand: the rules are about
        // the frame, not about which end the head is on.
        XCTAssertEqual(FramingCheck.status(people: [invertedBody()]), .ok)
    }

    func testAnklesAtTheTopEdgeAreCutOffAtTheTop() {
        // Inverted body whose feet touch the top of the picture.
        let status = FramingCheck.status(people: [invertedBody(ankleY: 0.02)])
        XCTAssertEqual(status, .partlyOutOfFrame(edges: [.top], missing: []))
    }

    func testWristsAtTheBottomEdgeAreCutOffAtTheBottom() {
        // Standing with the hands hanging off the bottom of the picture.
        var body = standingBody()
        body[.leftElbow] = joint(0.44, 0.92)
        body[.rightElbow] = joint(0.56, 0.92)
        body[.leftWrist] = joint(0.44, 0.98)
        body[.rightWrist] = joint(0.56, 0.98)
        XCTAssertEqual(
            FramingCheck.status(people: [body]),
            .partlyOutOfFrame(edges: [.bottom], missing: [])
        )
    }

    func testNoPeopleIsNoPerson() {
        XCTAssertEqual(FramingCheck.status(people: []), .noPerson)
        // A "person" nothing confident about is nobody the guide can judge.
        let ghost = standingBody().mapValues { Keypoint(x: $0.x, y: $0.y, visibility: 0.1) }
        XCTAssertEqual(FramingCheck.status(people: [ghost]), .noPerson)
    }

    func testTwoPeopleIsMultiplePeople() {
        let first = standingBody(centreX: 0.25)
        let second = standingBody(centreX: 0.75)
        XCTAssertEqual(first.values.filter { $0.visibility >= FramingCheck.confidentThreshold }.count, 13)
        XCTAssertEqual(FramingCheck.status(people: [first, second]), .multiplePeople)
    }

    func testASecondBodyNeedsSixConfidentJointsToCount() {
        // A glimpse of someone at the edge of the frame is not a second
        // athlete: only three joints are confidently theirs.
        let athlete = standingBody()
        var passerBy = standingBody(centreX: 0.75)
        for (index, jointName) in Joint.allCases.enumerated() where index >= 3 {
            passerBy[jointName] = Keypoint(x: 0.9, y: 0.5, visibility: 0.1)
        }
        XCTAssertEqual(FramingCheck.status(people: [athlete, passerBy]), .ok)
    }

    func testSmallBodyIsTooSmall() {
        XCTAssertEqual(FramingCheck.status(people: [distantBody()]), .tooSmall)
    }

    func testAJointNobodyCanSeeIsReportedAsMissing() {
        var body = standingBody()
        body.removeValue(forKey: .leftWrist)
        XCTAssertEqual(
            FramingCheck.status(people: [body]),
            .partlyOutOfFrame(edges: [], missing: [.leftWrist])
        )

        // Below the confidence threshold reads exactly like absent.
        body = standingBody()
        body[.leftWrist] = joint(0.44, 0.4, confidence: FramingCheck.confidentThreshold - 0.01)
        XCTAssertEqual(
            FramingCheck.status(people: [body]),
            .partlyOutOfFrame(edges: [], missing: [.leftWrist])
        )
    }

    func testAJointExactlyOnTheConfidenceThresholdCounts() {
        var body = standingBody()
        body[.leftWrist] = joint(0.44, 0.4, confidence: FramingCheck.confidentThreshold)
        XCTAssertEqual(FramingCheck.status(people: [body]), .ok)
    }

    func testACutOffJointIsAnEdgeAndNotAMissingJoint() {
        // Both readings side by side: seen-but-clipped goes on the edge list,
        // never seen at all goes on the missing list.
        let clipped = FramingCheck.status(people: [invertedBody(ankleY: 0.02)])
        XCTAssertEqual(clipped, .partlyOutOfFrame(edges: [.top], missing: []))
        var unseen = invertedBody()
        unseen.removeValue(forKey: .leftAnkle)
        unseen.removeValue(forKey: .rightAnkle)
        XCTAssertEqual(
            FramingCheck.status(people: [unseen]),
            .partlyOutOfFrame(edges: [], missing: [.leftAnkle, .rightAnkle])
        )
    }

    func testAJointOutsideTheFrameIsOnThatEdgeToo() {
        var body = standingBody()
        body[.leftAnkle] = joint(-0.2, 0.88)
        body[.rightAnkle] = joint(-0.3, 0.88)
        XCTAssertEqual(
            FramingCheck.status(people: [body]),
            .partlyOutOfFrame(edges: [.left], missing: [])
        )
    }

    // MARK: - The sentences

    func testOneMessagePerStatus() {
        XCTAssertEqual(FramingCheck.message(for: .ok), "Looks good")
        XCTAssertEqual(FramingCheck.message(for: .noPerson), "Step into the frame")
        XCTAssertEqual(FramingCheck.message(for: .tooSmall), "Move closer")
        XCTAssertEqual(FramingCheck.message(for: .multiplePeople), "Only you in the frame please")
        XCTAssertEqual(
            FramingCheck.message(for: .partlyOutOfFrame(edges: [], missing: [.leftAnkle, .rightAnkle])),
            "Step back: your feet are out of frame"
        )
        XCTAssertEqual(
            FramingCheck.message(for: .partlyOutOfFrame(edges: [.top], missing: [])),
            "Step back: you are cut off at the top of the frame"
        )
        XCTAssertEqual(
            FramingCheck.message(for: .partlyOutOfFrame(edges: [.bottom], missing: [.leftWrist])),
            "Step back: you are cut off at the bottom of the frame and your hands are out of frame"
        )
    }

    func testEveryStatusHasASentenceThatNamesTheProblem() {
        let bodies: [(FramingStatus, String)] = [
            (.ok, "Looks good"),
            (.noPerson, "Step into the frame"),
            (.tooSmall, "Move closer"),
            (.multiplePeople, "Only you in the frame please"),
        ]
        for (status, expected) in bodies {
            XCTAssertEqual(FramingCheck.message(for: status), expected, "\(status)")
            XCTAssertFalse(FramingCheck.message(for: status).isEmpty)
        }
        // The two cut-off verdicts, as the guide produces them.
        XCTAssertEqual(
            FramingCheck.message(for: FramingCheck.status(people: [invertedBody(ankleY: 0.02)])),
            "Step back: you are cut off at the top of the frame"
        )
        XCTAssertEqual(
            FramingCheck.message(for: FramingCheck.status(people: [distantBody()])),
            "Move closer"
        )
    }

    func testTheSentenceStaysStableForSeveralMissingJoints() {
        // Both wrists and both ankles gone: one "hands and feet", no
        // repetition, deterministic order (the order the required joints are
        // declared in, not a Set's).
        XCTAssertEqual(
            FramingCheck.message(for: .partlyOutOfFrame(edges: [], missing: [.rightAnkle, .leftWrist, .leftAnkle, .rightWrist])),
            "Step back: your hands and feet are out of frame"
        )
        XCTAssertEqual(
            FramingCheck.message(for: .partlyOutOfFrame(edges: [.left, .top], missing: [])),
            "Step back: you are cut off at the top of the frame and the left edge"
        )
        XCTAssertEqual(
            FramingCheck.message(for: .partlyOutOfFrame(edges: [], missing: [])),
            "Step back: part of your body is out of frame"
        )
    }
}
