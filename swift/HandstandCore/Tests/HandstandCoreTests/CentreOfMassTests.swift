import XCTest

@testable import HandstandCore

/// The Swift mirror of `pipeline/tests/test_com.py`: the same thirteen cases,
/// the same poses written by hand **in the body frame** — the wrist midpoint
/// at the origin, `v` up, both in body lengths — and the same expectations, so
/// each assertion is arithmetic on Winter's constants a reader can check with
/// a pencil.
///
/// Nothing here reads a video or a fixture: the pose is a symmetric vertical
/// handstand (`line`), and the tests move one part of it and say exactly how
/// far the centre of mass should have followed, which is the only way to tell a
/// segment model that works from one that happens to give a plausible number.
final class CentreOfMassTests: XCTestCase {
    // MARK: - A synthetic body, as `test_com.py` builds it

    /// The joints a side view reports — Python's `JOINTS`. There is no
    /// `foot_index` here, so this is both the schema Apple Vision has and the
    /// one the features tests drive: the foot segment ends at the ankle.
    private static let joints: [Joint] = (
        ["nose"]
            + ["left", "right"].flatMap { side in
                ["ankle", "elbow", "hip", "knee", "shoulder", "wrist"].map { "\(side)_\($0)" }
            }
    ).compactMap { Joint(rawValue: $0) }

    /// The same schema with the toes — Python's `FOOT_JOINTS`.
    private static let footJoints: [Joint] = joints + [.leftFootIndex, .rightFootIndex]

    /// The joints every test moves to shift the legs, both sides below the hip.
    private let legJoints: [Joint] = [
        .leftHip, .rightHip, .leftKnee, .rightKnee, .leftAnkle, .rightAnkle,
    ]

    /// A symmetric vertical handstand in the body frame — Python's `LINE`:
    /// every station on the vertical through the hands, the arms straight, the
    /// nose in front of the chest.
    private let line: [Joint: (Double, Double)] = [
        .nose: (0.04, 0.30),
        .leftWrist: (0.0, 0.0),
        .rightWrist: (0.0, 0.0),
        .leftElbow: (0.0, 0.195),
        .rightElbow: (0.0, 0.195),
        .leftShoulder: (0.0, 0.39),
        .rightShoulder: (0.0, 0.39),
        .leftHip: (0.0, 0.75),
        .rightHip: (0.0, 0.75),
        .leftKnee: (0.0, 0.95),
        .rightKnee: (0.0, 0.95),
        .leftAnkle: (0.0, 1.15),
        .rightAnkle: (0.0, 1.15),
    ]

    /// One pose as a `[frame][joint]` body-frame track — Python's `uv_of`.
    ///
    /// A joint that is not in `points` — or that `hidden` names — is `NaN`
    /// there, which is exactly how `features.bodyFrameTrack` writes a joint
    /// the processed data had nothing for.
    private func uvOf(
        _ points: [Joint: (Double, Double)],
        joints: [Joint] = CentreOfMassTests.joints,
        hidden: Set<Joint> = [],
        frames: Int = 1
    ) -> [[BodyPoint]] {
        var uv: [[BodyPoint]] = []
        for _ in 0..<frames {
            var row: [BodyPoint] = []
            for name in joints {
                if let point = points[name], !hidden.contains(name) {
                    row.append(BodyPoint(u: point.0, v: point.1))
                } else {
                    row.append(.nan)
                }
            }
            uv.append(row)
        }
        return uv
    }

    /// The centre of mass of one pose — Python's `com_of`.
    private func comOf(
        _ points: [Joint: (Double, Double)],
        joints: [Joint] = CentreOfMassTests.joints,
        hidden: Set<Joint> = [],
        frames: Int = 1
    ) throws -> CentreOfMassResult {
        try CentreOfMass.centreOfMass(
            uv: uvOf(points, joints: joints, hidden: hidden, frames: frames), joints: joints)
    }

    /// `points` with every named joint moved to `u`, keeping its `v` —
    /// Python's `at_u`.
    private func atU(
        _ points: [Joint: (Double, Double)], _ names: [Joint], _ u: Double
    ) -> [Joint: (Double, Double)] {
        var updated = points
        for name in names {
            guard let point = points[name] else { continue }
            updated[name] = (u, point.1)
        }
        return updated
    }

    // MARK: - The segment model

    /// `test_the_segments_are_winters_masses_and_add_to_a_whole_body`: the
    /// model only balances if the constants are the ones they say they are.
    func testTheSegmentsAreWintersMassesAndAddToAWholeBody() {
        XCTAssertTrue(CentreOfMass.segmentSource.hasPrefix("Winter, D. A. (2009)"))

        let sidedMass = CentreOfMass.segments.filter { $0.sides }
            .reduce(0.0) { $0 + $1.mass }
        let total = CentreOfMass.massHeadNeck + CentreOfMass.massTrunk + 2.0 * sidedMass
        XCTAssertEqual(total, 1.0, accuracy: 1e-12)
        XCTAssertEqual(CentreOfMass.massHeadNeck, 0.081)
        XCTAssertEqual(CentreOfMass.massTrunk, 0.497)

        // The segment table and the named constants are the same numbers, so
        // the two cannot drift apart.
        let masses = Dictionary(
            uniqueKeysWithValues: CentreOfMass.segments.map { ($0.name, $0.mass) })
        XCTAssertEqual(masses["trunk"] ?? .nan, CentreOfMass.massTrunk)
        XCTAssertEqual(masses["upper_arm"] ?? .nan, CentreOfMass.massUpperArm)
        XCTAssertEqual(masses["forearm_hand"] ?? .nan, CentreOfMass.massForearmHand)
        XCTAssertEqual(masses["thigh"] ?? .nan, CentreOfMass.massThigh)
        XCTAssertEqual(masses["shank"] ?? .nan, CentreOfMass.massShank)
        XCTAssertEqual(masses["foot"] ?? .nan, CentreOfMass.massFoot)

        let fractions = Dictionary(
            uniqueKeysWithValues: CentreOfMass.segments.map { ($0.name, $0.fraction) })
        XCTAssertEqual(fractions["upper_arm"] ?? .nan, 0.436)
        XCTAssertEqual(fractions["forearm_hand"] ?? .nan, 0.682)
        XCTAssertEqual(fractions["thigh"] ?? .nan, 0.433)
        XCTAssertEqual(fractions["shank"] ?? .nan, 0.433)
        XCTAssertEqual(fractions["trunk"] ?? .nan, 0.5)
        XCTAssertEqual(fractions["foot"] ?? .nan, 0.5)
        XCTAssertEqual(CentreOfMass.comHeadNeck, 1.0)

        XCTAssertEqual(CentreOfMass.headFallback, 0.5)
        XCTAssertEqual(CentreOfMass.baseBack, 0.03)
        XCTAssertEqual(CentreOfMass.baseFront, 0.06)
        XCTAssertEqual(CentreOfMass.sides, ["left", "right"])
    }

    /// `test_a_symmetric_vertical_handstand_has_its_com_over_the_hands`.
    func testASymmetricVerticalHandstandHasItsComOverTheHands() throws {
        let result = try comOf(line)

        XCTAssertEqual(result.frames, 1)
        // Symmetric about the vertical through the wrists: the only thing off
        // the line is the head, which the nose puts a whole 0.04 L to one side.
        XCTAssertEqual(result.u[0], CentreOfMass.massHeadNeck * 0.04, accuracy: 1e-12)
        XCTAssertLessThan(abs(result.u[0]), 0.01)
        // And it sits between the hands and the hips, below the trunk's own
        // centre.
        XCTAssertGreaterThan(result.v[0], 0.0)
        XCTAssertLessThan(result.v[0], 0.75)
        XCTAssertTrue(result.complete[0])
    }

    /// `test_the_com_follows_the_mass_that_moved`: the check that the masses
    /// add to one — a model whose masses summed to 0.9 would move by 1/0.9 of
    /// the answer here.
    func testTheComFollowsTheMassThatMoved() throws {
        let base = try comOf(line)

        // Whole body one body length sideways: every segment moved the same,
        // so the weighted mean of their centres moved the same.
        var sidewaysPoints: [Joint: (Double, Double)] = [:]
        for (joint, point) in line {
            sidewaysPoints[joint] = (point.0 + 0.3, point.1)
        }
        let sideways = try comOf(sidewaysPoints)
        XCTAssertEqual(sideways.u[0] - base.u[0], 0.3, accuracy: 1e-12)

        // The legs below the hip, both sides, forwards by x: thigh, shank and
        // foot take their centres with them, so the leg share of the body
        // (0.322) moves by x — and the trunk's lower end moves with the hips,
        // so its midpoint follows half the way, carrying MASS_TRUNK / 2 of the
        // body with it.
        let x = 0.1
        let legShare =
            2.0 * (CentreOfMass.massThigh + CentreOfMass.massShank + CentreOfMass.massFoot)
        XCTAssertEqual(legShare, 0.322, accuracy: 1e-12)
        let forward = try comOf(atU(line, legJoints, x))
        let expected = legShare * x + CentreOfMass.massTrunk * 0.5 * x
        XCTAssertEqual(forward.u[0] - base.u[0], expected, accuracy: 1e-12)
        XCTAssertGreaterThan(forward.u[0], base.u[0])
        XCTAssertLessThan(try comOf(atU(line, legJoints, -x)).u[0], base.u[0])

        // With the hip pinned, only the knee and the ankle move: the thigh's
        // centre is 0.433 of the way down it, so it follows only that
        // fraction, while the shank and the foot follow all the way.
        let belowJoints: [Joint] = [.leftKnee, .rightKnee, .leftAnkle, .rightAnkle]
        let below = try comOf(atU(line, belowJoints, x))
        let expectedBelow =
            2.0
            * (CentreOfMass.comThigh * CentreOfMass.massThigh + CentreOfMass.massShank
                + CentreOfMass.massFoot) * x
        XCTAssertEqual(below.u[0] - base.u[0], expectedBelow, accuracy: 1e-12)
        XCTAssertLessThan(expectedBelow, legShare * x)
    }

    /// `test_the_head_is_the_nose_and_falls_back_to_the_spine`.
    func testTheHeadIsTheNoseAndFallsBackToTheSpine() throws {
        let base = try comOf(line)

        // The head's mass is at the nose, so moving the nose moves the CoM by
        // the head's share of the move — which is where 0.081 shows up in a
        // measurement.
        var furtherPoints = line
        furtherPoints[.nose] = (0.14, 0.30)
        let further = try comOf(furtherPoints)
        XCTAssertEqual(
            further.u[0] - base.u[0], CentreOfMass.massHeadNeck * 0.10, accuracy: 1e-12)

        // Without a nose the head goes 0.5 of the way from the shoulder
        // midpoint towards the body's axis extended past the shoulder — here
        // straight down the image, to v = 0.39 + 0.5 (0.39 - 0.75) = 0.21 and
        // u = 0.
        let headV = 0.39 + CentreOfMass.headFallback * (0.39 - 0.75)
        XCTAssertEqual(headV, 0.21, accuracy: 1e-12)
        let fallback = try comOf(line, hidden: [.nose])
        var atNose = line
        atNose[.nose] = (0.0, headV)
        let atFallback = try comOf(atNose)
        XCTAssertEqual(fallback.u[0], atFallback.u[0], accuracy: 1e-12)
        XCTAssertEqual(fallback.v[0], atFallback.v[0], accuracy: 1e-12)
        // An estimated head is not a missing head: the frame stays complete.
        XCTAssertTrue(fallback.complete[0])

        // With neither a nose nor a hip to extend the spine from, the head is
        // a segment that is not there: the rest of the body is still weighed,
        // and the frame says so rather than pretending otherwise.
        let missing = try comOf(line, hidden: [.nose, .leftHip, .rightHip])
        XCTAssertFalse(missing.complete[0])
        XCTAssertTrue(missing.u[0].isFinite)
        XCTAssertTrue(missing.v[0].isFinite)
    }

    /// `test_a_foot_index_the_schema_does_not_have_ends_the_foot_at_the_ankle`.
    func testAFootIndexTheSchemaDoesNotHaveEndsTheFootAtTheAnkle() throws {
        // Apple Vision reports no toes: the foot is still a foot, it is just
        // zero-length, and its centre is the ankle's.
        var toesAtAnkle = line
        toesAtAnkle[.leftFootIndex] = line[.leftAnkle]
        toesAtAnkle[.rightFootIndex] = line[.rightAnkle]
        let withoutToes = try comOf(line, joints: CentreOfMassTests.joints)
        let withToes = try comOf(toesAtAnkle, joints: CentreOfMassTests.footJoints)
        XCTAssertEqual(withToes.u[0], withoutToes.u[0], accuracy: 1e-12)
        XCTAssertEqual(withToes.v[0], withoutToes.v[0], accuracy: 1e-12)

        // A schema that *has* the toes but lost them on this frame drops the
        // 2.9 % of the body they carry rather than inventing where they were.
        let lost = try comOf(line, joints: CentreOfMassTests.footJoints)
        XCTAssertFalse(lost.complete[0])
        XCTAssertTrue(lost.u[0].isFinite)
        XCTAssertTrue(lost.v[0].isFinite)

        // And real toes move the CoM by their mass share of the move, taken
        // where along the foot that move lands: both feet, 0.0145 each, the
        // toes 0.1 L forwards and the foot's centre 0.5 of the way to them.
        var toes = toesAtAnkle
        toes[.leftFootIndex] = (0.1, 1.2)
        toes[.rightFootIndex] = (0.1, 1.2)
        let moved = try comOf(toes, joints: CentreOfMassTests.footJoints)
        let expected =
            2.0 * CentreOfMass.massFoot * CentreOfMass.comFoot * (0.1 - 0.0)
        XCTAssertEqual(moved.u[0] - withToes.u[0], expected, accuracy: 1e-12)
        XCTAssertTrue(
            try CentreOfMass.centreOfMass(
                uv: uvOf(toes, joints: CentreOfMassTests.footJoints),
                joints: CentreOfMassTests.footJoints
            ).complete[0])
    }

    /// `test_a_segment_missing_on_one_side_is_taken_from_the_other`: a side
    /// view hides the far side, and in a side view the two sides overlap — the
    /// far thigh is at the near thigh's coordinates, not its mirror image.
    func testASegmentMissingOnOneSideIsTakenFromTheOther() throws {
        let both = try comOf(line)
        let oneSide = try comOf(
            line,
            hidden: [.leftHip, .leftKnee, .leftAnkle, .leftShoulder, .leftElbow])

        XCTAssertEqual(oneSide.u[0], both.u[0], accuracy: 1e-12)
        XCTAssertEqual(oneSide.v[0], both.v[0], accuracy: 1e-12)
        XCTAssertTrue(oneSide.complete[0])
    }

    /// `test_a_segment_missing_on_both_sides_is_renormalised_and_flagged`.
    func testASegmentMissingOnBothSidesIsRenormalisedAndFlagged() throws {
        // Both shanks and both feet out of the picture. The CoM of what is
        // there, over the mass that is there — and the frame flagged as
        // incomplete, because a quarter of the body went missing.
        let gone = try comOf(line, hidden: [.leftKnee, .rightKnee, .leftAnkle, .rightAnkle])
        let kept =
            1.0 - 2.0 * (CentreOfMass.massThigh + CentreOfMass.massShank + CentreOfMass.massFoot)

        XCTAssertFalse(gone.complete[0])
        XCTAssertEqual(kept, 0.678, accuracy: 1e-12)
        let keptByHand = CentreOfMass.massHeadNeck + CentreOfMass.massTrunk
            + 2.0 * (CentreOfMass.massUpperArm + CentreOfMass.massForearmHand)
        XCTAssertEqual(kept, keptByHand, accuracy: 1e-12)

        // The surviving segments, written out: the nose, the trunk's
        // midpoint, and per side the arm halfway to the elbow's own fraction
        // and the forearm's.
        let shoulder = BodyPoint(u: 0.0, v: 0.39)
        let elbow = BodyPoint(u: 0.0, v: 0.195)
        let wrist = BodyPoint(u: 0.0, v: 0.0)
        let hip = BodyPoint(u: 0.0, v: 0.75)
        let nose = BodyPoint(u: 0.04, v: 0.30)
        func along(_ proximal: BodyPoint, _ distal: BodyPoint, _ fraction: Double) -> BodyPoint {
            BodyPoint(
                u: proximal.u + fraction * (distal.u - proximal.u),
                v: proximal.v + fraction * (distal.v - proximal.v))
        }
        let trunk = along(shoulder, hip, CentreOfMass.comTrunk)
        let arm = along(shoulder, elbow, CentreOfMass.comUpperArm)
        let forearm = along(elbow, wrist, CentreOfMass.comForearmHand)
        let momentU =
            CentreOfMass.massHeadNeck * nose.u + CentreOfMass.massTrunk * trunk.u
            + 2.0 * (CentreOfMass.massUpperArm * arm.u + CentreOfMass.massForearmHand * forearm.u)
        let momentV =
            CentreOfMass.massHeadNeck * nose.v + CentreOfMass.massTrunk * trunk.v
            + 2.0 * (CentreOfMass.massUpperArm * arm.v + CentreOfMass.massForearmHand * forearm.v)
        XCTAssertEqual(gone.u[0], momentU / kept, accuracy: 1e-12)
        XCTAssertEqual(gone.v[0], momentV / kept, accuracy: 1e-12)

        // And a frame that loses even the trunk's ends still has something to
        // weigh.
        let trunkless = try comOf(
            line, hidden: [.leftShoulder, .rightShoulder, .leftElbow, .rightElbow])
        XCTAssertFalse(trunkless.complete[0])
        XCTAssertTrue(trunkless.u[0].isFinite)
        XCTAssertTrue(trunkless.v[0].isFinite)
    }

    /// `test_a_frame_with_nothing_visible_has_no_centre_of_mass`.
    func testAFrameWithNothingVisibleHasNoCentreOfMass() throws {
        let blank = [[BodyPoint]](
            repeating: [BodyPoint](repeating: .nan, count: CentreOfMassTests.joints.count),
            count: 2)
        let nothing = try CentreOfMass.centreOfMass(
            uv: blank, joints: CentreOfMassTests.joints)

        XCTAssertTrue(nothing.u.allSatisfy { $0.isNaN })
        XCTAssertTrue(nothing.v.allSatisfy { $0.isNaN })
        XCTAssertFalse(nothing.complete.contains(true))
    }

    /// `test_the_model_refuses_a_track_it_cannot_read`: Python's `ValueError`,
    /// message and all. (Python's second case — a point with three components
    /// — has no Swift shape to be written in: `BodyPoint` is two doubles.)
    func testTheModelRefusesATrackItCannotRead() {
        let wrongColumns = [[BodyPoint]](
            repeating: [BodyPoint](repeating: .nan, count: 4), count: 3)

        XCTAssertThrowsError(
            try CentreOfMass.centreOfMass(uv: wrongColumns, joints: CentreOfMassTests.joints)
        ) { error in
            XCTAssertEqual(
                "\(error)",
                "uv must be (frames, 13, 2) for those joint names, got (3, 4, 2)")
        }

        // A row that does not hold every joint name is refused the same way.
        var ragged = uvOf(line)
        ragged[0].removeLast()
        XCTAssertThrowsError(try CentreOfMass.centreOfMass(uv: ragged, joints: CentreOfMassTests.joints))
    }

    // MARK: - Which way the athlete faces

    /// The one-frame torso line of `LINE`, as Python's `torso_line` builds it.
    private func torsoLine() -> (shoulder: [BodyPoint], hip: [BodyPoint]) {
        (
            shoulder: [BodyPoint(u: 0.0, v: 0.39)],
            hip: [BodyPoint(u: 0.0, v: 0.75)]
        )
    }

    /// `test_the_facing_sign_is_the_side_of_the_torso_line_the_nose_is_on`.
    func testTheFacingSignIsTheSideOfTheTorsoLineTheNoseIsOn() {
        let torso = torsoLine()

        // The nose is in front of the chest, so the nose's side of the torso
        // line is the way the fingers point: +u here, +1.
        XCTAssertEqual(
            CentreOfMass.facingSign(
                shoulderMid: torso.shoulder, hipMid: torso.hip,
                nose: [BodyPoint(u: 0.04, v: 0.30)])[0],
            1.0)
        XCTAssertEqual(
            CentreOfMass.facingSign(
                shoulderMid: torso.shoulder, hipMid: torso.hip,
                nose: [BodyPoint(u: -0.04, v: 0.30)])[0],
            -1.0)
        // No nose is no direction, and neither is a nose sitting on the line.
        XCTAssertTrue(
            CentreOfMass.facingSign(
                shoulderMid: torso.shoulder, hipMid: torso.hip,
                nose: [BodyPoint.nan])[0].isNaN)
        XCTAssertTrue(
            CentreOfMass.facingSign(
                shoulderMid: torso.shoulder, hipMid: torso.hip,
                nose: [BodyPoint(u: 0.0, v: 0.30)])[0].isNaN)

        // The side of the line does not depend on which way up the torso is
        // drawn: a hip *below* the shoulder still has a nose at +u on its +u
        // side.
        XCTAssertEqual(
            CentreOfMass.facingSign(
                shoulderMid: torso.shoulder,
                hipMid: [BodyPoint(u: 0.0, v: 0.39 - 0.36)],
                nose: [BodyPoint(u: 0.04, v: 0.30)])[0],
            1.0)

        // And a shoulder and hip in the same place is not a line at all.
        let same = [BodyPoint(u: 0.0, v: 0.39)]
        XCTAssertTrue(
            CentreOfMass.facingSign(
                shoulderMid: same, hipMid: same, nose: [BodyPoint(u: 0.04, v: 0.30)])[0].isNaN)
    }

    /// `test_the_facing_sign_is_decided_per_hold_and_not_per_frame`.
    func testTheFacingSignIsDecidedPerHoldAndNotPerFrame() {
        let votes: [Double] = [
            1.0, -1.0, 1.0, 1.0, .nan, -1.0, -1.0, .nan, 1.0, -1.0, .nan,
        ]
        let holds = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, -1]

        let decided = CentreOfMass.majorityPerHold(votes, holdId: holds)

        // Hold 0 voted three to one for +1: every frame of it faces +u,
        // including the one that had no nose to vote with.
        XCTAssertEqual(Array(decided[0..<5]), [1.0, 1.0, 1.0, 1.0, 1.0])
        // Hold 1's only finite votes are -1, so its noseless frame takes them
        // too.
        XCTAssertEqual(Array(decided[5..<8]), [-1.0, -1.0, -1.0])
        // Hold 2 is an exact tie: no majority, so the frames keep their own
        // votes rather than being told a direction nothing agreed on.
        XCTAssertEqual(decided[8], 1.0)
        XCTAssertEqual(decided[9], -1.0)
        // A frame outside a hold keeps its own vote, nose or no nose.
        XCTAssertTrue(decided[10].isNaN)

        // The same two rules without the tie in the way: a hold that agrees
        // is unchanged, and a hold with no finite vote at all stays NaN.
        XCTAssertEqual(
            CentreOfMass.majorityPerHold([1.0, 1.0], holdId: [3, 3]), [1.0, 1.0])
        XCTAssertTrue(
            CentreOfMass.majorityPerHold([.nan, .nan], holdId: [3, 3]).allSatisfy { $0.isNaN })
    }

    /// `test_com_forward_is_the_com_in_the_direction_the_fingers_point`.
    func testComForwardIsTheComInTheDirectionTheFingersPoint() {
        let forward = CentreOfMass.comForward([0.05, -0.05, 0.05], [1.0, 1.0, -1.0])

        // A CoM to the right is overbalanced for an athlete facing right and
        // underbalanced for one facing left: the same position, the other way
        // up.
        XCTAssertEqual(forward, [0.05, -0.05, -0.05])
        // An unmeasured sign is an unmeasured direction, not a sign of +1.
        XCTAssertTrue(CentreOfMass.comForward([0.05], [.nan])[0].isNaN)
    }

    // MARK: - The zones

    /// `test_the_balance_zones_are_cut_at_the_edges_of_the_base_of_support`.
    func testTheBalanceZonesAreCutAtTheEdgesOfTheBaseOfSupport() {
        let zone = CentreOfMass.balanceZone([
            -CentreOfMass.baseBack - 1e-9,
            -CentreOfMass.baseBack,
            0.0,
            CentreOfMass.baseFront,
            CentreOfMass.baseFront + 1e-9,
            .nan,
        ])

        // Just past the heel of the hand is under, just inside the fingertips
        // is ok, just past them is over, and the edges themselves are still
        // ok: the test is strictly outside the base.
        XCTAssertEqual(zone.compactMap { $0 }, [.under, .ok, .ok, .ok, .over])
        // A frame with no CoM has no zone, and the NaN does not become a word.
        XCTAssertNil(zone[5])

        // The edges are the order of a hand's own length, they are
        // approximate, and they are asymmetric: there is more room towards the
        // fingertips.
        XCTAssertEqual(CentreOfMass.baseBack, 0.03)
        XCTAssertEqual(CentreOfMass.baseFront, 0.06)
        XCTAssertLessThan(CentreOfMass.baseBack, CentreOfMass.baseFront)
        XCTAssertEqual(CentreOfMass.zoneNames, [.under, .ok, .over])
        XCTAssertEqual(BalanceZone.allCases, [.under, .ok, .over])
        XCTAssertEqual(BalanceZone.allCases.map(\.rawValue), ["under", "ok", "over"])
    }
}
