// The `--rotate auto` rule, on synthetic points.
//
// The rule is the MediaPipe runner's: turn the next frame 180° when the body in
// the frame just seen had its wrists below its ankles, judged from the person
// whose wrists are lowest. A handstand clips these, so the rule has to read a
// body that is upside down in the display frame as "rotate".

import XCTest

@testable import VisionPoseCore

final class AutoRotationTests: XCTestCase {
    /// A pose with both wrists at `wristsY` and both ankles at `anklesY`, in
    /// display pixels. Everything else sits in the middle, so only the
    /// wrist/ankle comparison can decide anything.
    private func pose(
        wristsY: Double,
        anklesY: Double,
        includeWrists: Bool = true,
        includeAnkles: Bool = true
    ) -> DisplayPose {
        var points: [VisionJoint: PixelPoint] = [
            .nose: PixelPoint(x: 100, y: 0.5 * 1023),
            .root: PixelPoint(x: 100, y: 0.5 * 1023),
        ]
        if includeWrists {
            points[.leftWrist] = PixelPoint(x: 80, y: wristsY)
            points[.rightWrist] = PixelPoint(x: 120, y: wristsY)
        }
        if includeAnkles {
            points[.leftAnkle] = PixelPoint(x: 90, y: anklesY)
            points[.rightAnkle] = PixelPoint(x: 110, y: anklesY)
        }
        return DisplayPose(points: points)
    }

    /// Standing: wrists high (small y), ankles low. Not a handstand.
    func testAnUprightBodyDoesNotRotate() {
        XCTAssertFalse(AutoRotation.isInverted(self.pose(wristsY: 200, anklesY: 800)))
    }

    /// A handstand: the hands are on the mat, so the wrists are *below* the
    /// feet in the image.
    func testAHandstandRotates() {
        XCTAssertTrue(AutoRotation.isInverted(self.pose(wristsY: 800, anklesY: 200)))
    }

    func testTheRuleIsStrictlyGreater() {
        // Wrists level with the ankles: neither upside down nor clearly upright.
        XCTAssertFalse(AutoRotation.isInverted(self.pose(wristsY: 500, anklesY: 500)))
    }

    /// A pose that cannot be judged counts as "not inverted", as it does in
    /// `pose_mediapipe.is_inverted`.
    func testAnIncompletePoseCannotBeJudged() {
        XCTAssertFalse(AutoRotation.isInverted(self.pose(wristsY: 900, anklesY: 100, includeAnkles: false)))
        XCTAssertFalse(AutoRotation.isInverted(self.pose(wristsY: 900, anklesY: 100, includeWrists: false)))
        XCTAssertFalse(AutoRotation.isInverted(DisplayPose(points: [:])))
    }

    /// Several people: the rule is judged from the one whose wrists are lowest,
    /// because that is the one most likely to be on their hands.
    func testTheLowestWristsWin() {
        let trainer = self.pose(wristsY: 150, anklesY: 850)  // standing, hands up
        let athlete = self.pose(wristsY: 900, anklesY: 200)  // handstand
        // Order does not matter.
        XCTAssertTrue(AutoRotation.isInverted(AutoRotation.chosenPose([trainer, athlete])!))
        XCTAssertTrue(AutoRotation.isInverted(AutoRotation.chosenPose([athlete, trainer])!))
    }

    /// People whose wrists are not both visible cannot be ranked and are
    /// skipped, so a half-detected body never steals the judgement.
    func testAnUnrankablePersonIsSkipped() {
        let unknown = self.pose(wristsY: 500, anklesY: 500, includeWrists: false)
        let athlete = self.pose(wristsY: 900, anklesY: 200)
        XCTAssertTrue(AutoRotation.isInverted(AutoRotation.chosenPose([unknown, athlete])!))
    }

    /// When nobody can be ranked the first person is judged, so a
    /// single-person frame is judged exactly as it always was.
    func testTheFirstPersonIsJudgedWhenNobodyCanBeRanked() {
        let a = self.pose(wristsY: 900, anklesY: 200, includeWrists: false)
        let b = self.pose(wristsY: 100, anklesY: 800, includeWrists: false)
        XCTAssertEqual(AutoRotation.chosenPose([a, b]), a)
        XCTAssertEqual(AutoRotation.chosenPose([b, a]), b)
        // ...and a body whose wrists are missing cannot be judged either way.
        XCTAssertFalse(AutoRotation.isInverted(a))
    }

    func testASinglePersonIsJudgedAsItself() {
        let only = self.pose(wristsY: 900, anklesY: 200)
        XCTAssertEqual(AutoRotation.chosenPose([only]), only)
    }

    func testNoPeopleHasNoChosenPose() {
        XCTAssertNil(AutoRotation.chosenPose([]))
    }

    // MARK: - The state machine

    func testTheFirstFrameIsNeverRotated() {
        XCTAssertFalse(AutoRotation().rotateNextFrame)
        XCTAssertFalse(RotateMode.auto.isRotated(AutoRotation().rotateNextFrame))
    }

    func testTheJudgementIsAboutTheNextFrame() {
        var state = AutoRotation()
        // A frame whose body is a handstand leaves the *next* one rotated.
        state.update(with: [self.pose(wristsY: 900, anklesY: 200)])
        XCTAssertTrue(state.rotateNextFrame)
        // ...and an upright body puts it back.
        state.update(with: [self.pose(wristsY: 200, anklesY: 800)])
        XCTAssertFalse(state.rotateNextFrame)
    }

    /// A frame with nobody in it says nothing about the orientation, so the
    /// previous decision stands: flipping every frame of a detection dropout
    /// would be the wrong answer.
    func testAnEmptyFrameKeepsThePreviousDecision() {
        var state = AutoRotation()
        state.update(with: [self.pose(wristsY: 900, anklesY: 200)])
        XCTAssertTrue(state.rotateNextFrame)
        state.update(with: [])
        XCTAssertTrue(state.rotateNextFrame)
        state.update(with: [self.pose(wristsY: 200, anklesY: 800)])
        XCTAssertFalse(state.rotateNextFrame)
        state.update(with: [])
        XCTAssertFalse(state.rotateNextFrame)
    }

    /// A clip that goes into a handstand and comes back out: rotated only for
    /// the frames after the body was seen upside down.
    func testAClipGoingIntoAndOutOfAHandstand() {
        var state = AutoRotation()
        var rotated: [Bool] = []
        for wristsY in [200.0, 200, 900, 900, 200] {
            let rotatedThisFrame = state.rotateNextFrame
            rotated.append(rotatedThisFrame)
            state.update(with: [self.pose(wristsY: wristsY, anklesY: wristsY == 200 ? 800 : 200)])
        }
        // Frame 0 upright -> not rotated; frame 1 still upright (the body in
        // frame 0 was upright) -> not rotated; frame 2 sees an upright body
        // again -> not rotated; frame 3 follows the handstand of frame 2 ->
        // rotated; frame 4 also rotated.
        XCTAssertEqual(rotated, [false, false, false, true, true])
    }

    // MARK: - The rotate modes

    func testTheRotateModes() {
        XCTAssertFalse(RotateMode.none.isRotated(true))
        XCTAssertFalse(RotateMode.none.isRotated(false))
        XCTAssertTrue(RotateMode.halfTurn.isRotated(false))
        XCTAssertTrue(RotateMode.halfTurn.isRotated(true))
        XCTAssertTrue(RotateMode.auto.isRotated(true))
        XCTAssertFalse(RotateMode.auto.isRotated(false))
        XCTAssertEqual(RotateMode.allCases.map(\.rawValue), ["none", "180", "auto"])
    }

    // MARK: - meanWristY

    func testMeanWristY() {
        let pose = self.pose(wristsY: 400, anklesY: 900)
        XCTAssertEqual(AutoRotation.meanWristY(pose), 400, accuracy: 1e-12)
        XCTAssertTrue(AutoRotation.meanWristY(DisplayPose(points: [:])).isNaN)
    }
}
