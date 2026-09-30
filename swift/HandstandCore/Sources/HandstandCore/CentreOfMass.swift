import Foundation

// --------------------------------------------------------------------------- #
// The centre of mass over the hands: where it is, which way the athlete faces,
// and which side of the base of support it sits on.
//
// A port of `pipeline/handstand/com.py` (chainlink #81): Winter's (2009)
// mass fractions and CoM positions over seven segments per side plus two
// midline ones, the missing-side and renormalisation rules, `facing_sign`
// decided **per hold by majority**, and the two edges of the base of support.
//
// The input is one clip's joints already in the body frame — the `BodyTrack`
// of Features.swift, origin at the wrist midpoint, `u` to the right, `v` up,
// units of body length `L` — and the answer is in the same units, so "the CoM
// is at 0.02 L" is a distance a judge could see.
//
// The rules the port lives by, taken straight from Python:
//
// * a segment missing on **one** side is measured on the other side's
//   coordinates — mirror-free: in a side view the two sides overlap, so the
//   far thigh is where the near thigh is, not its mirror image;
// * a segment missing on **both** sides is dropped and the remaining masses
//   are renormalised to sum to one, with `complete` false on that frame;
// * the head's CoM *is* the nose, or — with no nose — 0.5 of the way from the
//   shoulder midpoint towards the body's axis extended past the shoulder,
//   which in a handstand is where the head hangs. An estimated head is not a
//   missing head: the frame stays complete;
// * a frame with nothing left has no CoM: `NaN`, not the mean of nothing;
// * `NaN` comparisons are false, as numpy's are, so an unmeasured sign is
//   never read as a direction.
// --------------------------------------------------------------------------- #

/// Which side of the base of support each frame's centre of mass is on —
/// `handstand.com.balance_zone`'s three answers, spelled exactly as Python
/// spells them, in `CentreOfMass.zoneNames` order.
public enum BalanceZone: String, Sendable, CaseIterable {
    /// Behind the heel of the hand: `com_forward < -CentreOfMass.baseBack`.
    case under
    /// Inside the base of support, both edges included.
    case ok
    /// In front of the fingertips: `com_forward > CentreOfMass.baseFront`.
    case over
}

/// One clip's centre of mass, and whether every segment was there —
/// `handstand.com.CentreOfMass`, with its `(frames, 2)` `com` column split
/// into the two body-frame axes.
///
/// `u` and `v` are `NaN` on a frame with nothing to measure, and `complete` is
/// `false` where a segment was missing on both sides and the masses were
/// renormalised without it — the number is still the CoM of the parts that
/// were seen, and `complete` says how much of the body that was.
public struct CentreOfMassResult: Sendable, Equatable {
    /// The CoM's horizontal position per frame, body lengths, `u` rightwards.
    public var u: [Double]
    /// The CoM's height per frame, body lengths, `v` upwards.
    public var v: [Double]
    /// Did every segment of the model have both its ends on this frame.
    public var complete: [Bool]

    public init(u: [Double], v: [Double], complete: [Bool]) {
        self.u = u
        self.v = v
        self.complete = complete
    }

    /// How many frames the answer covers — `CentreOfMass.frames`.
    public var frames: Int { u.count }
}

/// Port of the segment model and the balance geometry of
/// `pipeline/handstand/com.py`: `centre_of_mass`, `facing_sign`,
/// `majority_per_hold`, `com_forward` and `balance_zone`, in that order.
///
/// The helpers (`points`, `midpoint`, `along`, `headPoints`, `segmentPoints`,
/// `midlinePoints`, `sideMean`, `facingSign`, `majorityPerHold`,
/// `comForward`, `balanceZone`) are `internal` statics, so `@testable import`
/// reaches them the way the Python tests reach the module functions.
public enum CentreOfMass {
    /// `BASE_BACK` — how far behind the wrist midpoint the heel of the hand
    /// reaches, in body lengths: the back edge of the base of support.
    /// **Approximate**, a hand's own length read off a side view.
    public static let baseBack = 0.03

    /// `BASE_FRONT` — how far in front of the wrist midpoint the fingertips
    /// reach, in body lengths: the front edge of the base of support, longer
    /// than the back edge because a hand is. Also **approximate**.
    public static let baseFront = 0.06

    /// `ZONE_NAMES` — the three zones, in the order they run along
    /// `com_forward`.
    public static let zoneNames: [BalanceZone] = [.under, .ok, .over]

    /// `SEGMENT_SOURCE` — where every mass and fraction below comes from,
    /// named because a segment model is only as good as the table it was
    /// copied from.
    public static let segmentSource =
        "Winter, D. A. (2009). Biomechanics and Motor Control of Human Movement, "
        + "3rd edition, Wiley. Anthropometric tables: segment mass as a fraction of "
        + "body mass, segment centre of mass as a fraction of segment length from the "
        + "proximal end."

    // MARK: - Winter's constants, one per Python module constant

    /// `MASS_HEAD_NECK` — head + neck (midline), a fraction of the body.
    static let massHeadNeck = 0.081
    /// `MASS_TRUNK` — trunk (midline).
    static let massTrunk = 0.497
    /// `MASS_UPPER_ARM` — upper arm, each side.
    static let massUpperArm = 0.028
    /// `MASS_FOREARM_HAND` — forearm + hand, each side.
    static let massForearmHand = 0.022
    /// `MASS_THIGH` — thigh, each side.
    static let massThigh = 0.100
    /// `MASS_SHANK` — shank, each side.
    static let massShank = 0.0465
    /// `MASS_FOOT` — foot, each side.
    static let massFoot = 0.0145

    /// `COM_HEAD_NECK` — the head's CoM is at the nose, i.e. 1.0 along
    /// shoulder_mid → nose.
    static let comHeadNeck = 1.0
    /// `COM_TRUNK` — 0.5 along shoulder_mid → hip_mid.
    static let comTrunk = 0.5
    /// `COM_UPPER_ARM` — 0.436 along shoulder → elbow.
    static let comUpperArm = 0.436
    /// `COM_FOREARM_HAND` — 0.682 along elbow → wrist.
    static let comForearmHand = 0.682
    /// `COM_THIGH` — 0.433 along hip → knee.
    static let comThigh = 0.433
    /// `COM_SHANK` — 0.433 along knee → ankle.
    static let comShank = 0.433
    /// `COM_FOOT` — 0.5 along ankle → foot_index.
    static let comFoot = 0.5

    /// `HEAD_FALLBACK` — how far towards the nose the head's CoM goes when
    /// there is no nose to put it at: 0.5 of the way from the shoulder
    /// midpoint towards the body's axis extended past the shoulder.
    static let headFallback = 0.5

    /// `SIDES` — the two sides, as the schema names them.
    static let sides = ["left", "right"]

    /// One segment of the model: its mass, its two body parts, and where its
    /// CoM sits along them — `handstand.com.Segment`.
    ///
    /// `proximal` and `distal` are *part* names, looked up per side
    /// (`left_hip`, `right_hip`) unless `sides` is false, in which case both
    /// are midpoints of the two sides (`shoulder_mid`, `hip_mid`).
    /// `fallback` is the distal part to use when the schema reports no such
    /// joint at all, and `nil` when there is nothing to fall back to.
    struct Segment: Sendable {
        let name: String
        let mass: Double
        let proximal: String
        let distal: String
        let fraction: Double
        let sides: Bool
        let fallback: String?
    }

    /// `SEGMENTS` — every segment except the head, whose rule is its own
    /// (`headPoints`). Order is the order the body is read in, hands to feet;
    /// it does not affect the sum, and the paired ones contribute twice.
    static let segments: [Segment] = [
        Segment(
            name: "trunk", mass: CentreOfMass.massTrunk, proximal: "shoulder",
            distal: "hip", fraction: CentreOfMass.comTrunk, sides: false,
            fallback: nil),
        Segment(
            name: "upper_arm", mass: CentreOfMass.massUpperArm, proximal: "shoulder",
            distal: "elbow", fraction: CentreOfMass.comUpperArm, sides: true,
            fallback: nil),
        Segment(
            name: "forearm_hand", mass: CentreOfMass.massForearmHand,
            proximal: "elbow", distal: "wrist", fraction: CentreOfMass.comForearmHand,
            sides: true, fallback: nil),
        Segment(
            name: "thigh", mass: CentreOfMass.massThigh, proximal: "hip",
            distal: "knee", fraction: CentreOfMass.comThigh, sides: true,
            fallback: nil),
        Segment(
            name: "shank", mass: CentreOfMass.massShank, proximal: "knee",
            distal: "ankle", fraction: CentreOfMass.comShank, sides: true,
            fallback: nil),
        Segment(
            name: "foot", mass: CentreOfMass.massFoot, proximal: "ankle",
            distal: "foot_index", fraction: CentreOfMass.comFoot, sides: true,
            fallback: "ankle"),
    ]

    // MARK: - One clip

    /// What `centreOfMass` refuses: the shape Python's `ValueError` names.
    enum TrackError: Error, Equatable, CustomStringConvertible {
        /// The track's rows do not all hold one point per joint name.
        case shapeMismatch(joints: Int, frames: Int, columns: Int)

        var description: String {
            switch self {
            case let .shapeMismatch(joints, frames, columns):
                return
                    "uv must be (frames, \(joints), 2) for those joint names, "
                    + "got (\(frames), \(columns), 2)"
            }
        }
    }

    /// The whole clip's centre of mass from its body-frame joints —
    /// `centre_of_mass`.
    ///
    /// `uv` is `[frame][joint]` in body-frame units with `NaN` wherever a
    /// joint was not visible (which is how `Features.bodyFrameTrack` writes a
    /// frame the processed data has nothing for), and `joints` is the schema's
    /// names in column order. The answer is the mass-weighted mean of the
    /// twelve segment CoMs — the two midline ones and both sides of the five
    /// paired ones — with the missing ones handled as the module documents.
    ///
    /// Python raises `ValueError` on a shape mismatch; this throws the same
    /// message text, so `CentreOfMassTests` can pin the refusal down.
    static func centreOfMass(uv: [[BodyPoint]], joints: [Joint]) throws -> CentreOfMassResult {
        let frames = uv.count
        guard uv.allSatisfy({ $0.count == joints.count }) else {
            throw TrackError.shapeMismatch(
                joints: joints.count, frames: frames, columns: uv.first?.count ?? 0)
        }

        var perSegment: [[BodyPoint]] = []
        var masses: [Double] = []
        perSegment.append(headPoints(uv: uv, joints: joints))
        masses.append(CentreOfMass.massHeadNeck)
        for segment in CentreOfMass.segments {
            if segment.sides {
                // Both sides are separate entries of the same mass: a segment
                // missing on one side is already filled from the other inside
                // `segmentPoints`, so a side view contributes its mass twice
                // at the coordinates of the side that is there.
                for sidePoints in segmentPoints(uv: uv, joints: joints, segment: segment) {
                    perSegment.append(sidePoints)
                    masses.append(segment.mass)
                }
            } else {
                perSegment.append(midlinePoints(uv: uv, joints: joints, segment: segment))
                masses.append(segment.mass)
            }
        }

        var u = [Double](repeating: .nan, count: frames)
        var v = [Double](repeating: .nan, count: frames)
        var complete = [Bool](repeating: false, count: frames)
        for frame in 0..<frames {
            // The missing segments are skipped before the moment is summed:
            // `NaN * 0` is still NaN, and one absent segment must not take the
            // rest of the frame with it. The division renormalises what is
            // left onto the sum of the masses that are there.
            var total = 0.0
            var momentU = 0.0
            var momentV = 0.0
            var present = true
            for index in perSegment.indices {
                let point = perSegment[index][frame]
                guard point.isFinite else {
                    present = false
                    continue
                }
                total += masses[index]
                momentU += point.u * masses[index]
                momentV += point.v * masses[index]
            }
            complete[frame] = present
            if total > 0.0 {
                u[frame] = momentU / total
                v[frame] = momentV / total
            }
        }
        return CentreOfMassResult(u: u, v: v, complete: complete)
    }

    /// The `(frames)` track of one joint, `NaN` when the schema has no such
    /// joint — `_points`.
    ///
    /// A source that does not report a joint (Apple Vision has no
    /// `foot_index`) answers `NaN` rather than raising or inventing a
    /// position, which is what lets the same model run over every source.
    static func points(uv: [[BodyPoint]], joints: [Joint], _ name: Joint) -> [BodyPoint] {
        point(uv: uv, joints: joints, raw: name.rawValue)
    }

    /// The `(frames)` track of one joint *named as Python names it*, `NaN`
    /// when neither the name nor the joint exists — `_points` over the part
    /// names (`left_hip`, `foot_index`) the segment table is written in.
    static func point(uv: [[BodyPoint]], joints: [Joint], raw: String) -> [BodyPoint] {
        guard let name = Joint(rawValue: raw), let column = joints.firstIndex(of: name) else {
            return [BodyPoint](repeating: .nan, count: uv.count)
        }
        return uv.map { row in column < row.count ? row[column] : .nan }
    }

    /// Both sides of a point as one: the mean where both are, the one that is
    /// where only one is, `NaN` where neither is — `_side_mean`, the same rule
    /// every midpoint uses.
    static func sideMean(_ left: [BodyPoint], _ right: [BodyPoint]) -> [BodyPoint] {
        left.indices.map { index in
            let first = left[index]
            let second = right[index]
            let firstSeen = first.isFinite
            let secondSeen = second.isFinite
            if firstSeen && secondSeen {
                return BodyPoint(u: (first.u + second.u) / 2.0, v: (first.v + second.v) / 2.0)
            }
            if firstSeen || secondSeen {
                return firstSeen ? first : second
            }
            return .nan
        }
    }

    /// The `(frames)` midpoint of both sides of one body part, one side being
    /// enough — `_midpoint`.
    static func midpoint(uv: [[BodyPoint]], joints: [Joint], part: String) -> [BodyPoint] {
        sideMean(
            point(uv: uv, joints: joints, raw: "left_\(part)"),
            point(uv: uv, joints: joints, raw: "right_\(part)")
        )
    }

    /// The point `fraction` of the way from `proximal` to `distal`, `NaN`
    /// where either end is — `_along`.
    ///
    /// `NaN` propagates rather than being skipped: a segment whose end is not
    /// visible is a segment that is not measured, and the caller decides what
    /// to do about that (take the other side, or drop the mass).
    static func along(_ proximal: [BodyPoint], _ distal: [BodyPoint], _ fraction: Double)
        -> [BodyPoint]
    {
        proximal.indices.map { index in
            BodyPoint(
                u: proximal[index].u + fraction * (distal[index].u - proximal[index].u),
                v: proximal[index].v + fraction * (distal[index].v - proximal[index].v)
            )
        }
    }

    /// The head + neck segment's CoM: the nose, or the documented fallback
    /// without one — `_head_points`.
    static func headPoints(uv: [[BodyPoint]], joints: [Joint]) -> [BodyPoint] {
        let nose = points(uv: uv, joints: joints, .nose)
        let shoulder = midpoint(uv: uv, joints: joints, part: "shoulder")
        let hip = midpoint(uv: uv, joints: joints, part: "hip")
        return nose.indices.map { index in
            if nose[index].isFinite {
                return nose[index]
            }
            return BodyPoint(
                u: shoulder[index].u
                    + CentreOfMass.headFallback * (shoulder[index].u - hip[index].u),
                v: shoulder[index].v
                    + CentreOfMass.headFallback * (shoulder[index].v - hip[index].v)
            )
        }
    }

    /// Both sides of one paired segment's CoM, each side filled from the
    /// other — `_segment_points`.
    ///
    /// A side whose endpoints are not both visible takes the other side's
    /// **coordinates** — not its mirror image. A segment missing on both
    /// sides comes back `NaN` on both, and the caller drops its mass.
    static func segmentPoints(uv: [[BodyPoint]], joints: [Joint], segment: Segment)
        -> [[BodyPoint]]
    {
        var distal = segment.distal
        if let fallback = segment.fallback {
            // The schema has no such joint *at all* (Apple Vision has no
            // toes): the foot is still a foot, it is just zero-length.
            let hasDistal = CentreOfMass.sides.contains { side in
                guard let name = Joint(rawValue: "\(side)_\(segment.distal)") else { return false }
                return joints.contains(name)
            }
            if !hasDistal {
                distal = fallback
            }
        }
        let left = along(
            point(uv: uv, joints: joints, raw: "left_\(segment.proximal)"),
            point(uv: uv, joints: joints, raw: "left_\(distal)"),
            segment.fraction)
        let right = along(
            point(uv: uv, joints: joints, raw: "right_\(segment.proximal)"),
            point(uv: uv, joints: joints, raw: "right_\(distal)"),
            segment.fraction)
        let filledLeft = left.indices.map { index in left[index].isFinite ? left[index] : right[index] }
        let filledRight = left.indices.map { index in right[index].isFinite ? right[index] : left[index] }
        return [filledLeft, filledRight]
    }

    /// One midline segment's CoM — `_midline_points`: a midpoint is one side
    /// being enough.
    static func midlinePoints(uv: [[BodyPoint]], joints: [Joint], segment: Segment) -> [BodyPoint] {
        along(
            midpoint(uv: uv, joints: joints, part: segment.proximal),
            midpoint(uv: uv, joints: joints, part: segment.distal),
            segment.fraction)
    }

    // MARK: - Which way is forward

    /// `+1` when the athlete faces `+u`, `-1` when they face `-u`, `NaN`
    /// without a nose — `facing_sign`.
    ///
    /// The torso line is the shoulder midpoint to the hip midpoint, and the
    /// nose is on one side of it or the other; the side that is towards `+u`
    /// is `+1`. In a handstand the fingers point the way the chest faces, so
    /// this is the direction of the fingers — which is what makes
    /// `com_forward` mean "towards the fingers" rather than "towards the
    /// right of the image". The hold-level vote is `majorityPerHold`'s job.
    static func facingSign(
        shoulderMid: [BodyPoint], hipMid: [BodyPoint], nose: [BodyPoint]
    ) -> [Double] {
        shoulderMid.indices.map { index in
            let axisU = hipMid[index].u - shoulderMid[index].u
            let axisV = hipMid[index].v - shoulderMid[index].v
            // The normal that points to +u: for a handstand's upright torso
            // line that is (axis_v, -axis_u) already, and flipping it when it
            // points the other way keeps "+1 is the side the nose is on
            // towards +u" true however the torso is tilted.
            var acrossU = axisV
            var acrossV = -axisU
            if acrossU < 0.0 {
                acrossU = -acrossU
                acrossV = -acrossV
            }
            let front = (nose[index].u - shoulderMid[index].u) * acrossU
                + (nose[index].v - shoulderMid[index].v) * acrossV
            let span = hypot(acrossU, acrossV)
            if front > 0.0 && span > 0.0 {
                return 1.0
            }
            if front < 0.0 && span > 0.0 {
                return -1.0
            }
            return .nan
        }
    }

    /// One facing sign per hold, by majority: the sign most of the hold's
    /// frames had — `majority_per_hold`.
    ///
    /// The frames of a hold all take the sign most of them voted for (`NaN`
    /// votes do not count), frames outside a hold keep their own vote, and a
    /// hold with no majority — no finite votes, or an exact tie — keeps its
    /// frames as they were rather than being told a direction nothing agreed
    /// on.
    static func majorityPerHold(_ sign: [Double], holdId: [Int]) -> [Double] {
        precondition(
            sign.count == holdId.count,
            "sign must have \(holdId.count) entries, got \(sign.count)")
        var decided = sign
        let numbers = Set(holdId.filter { $0 >= 0 }).sorted()
        for number in numbers {
            let frames = holdId.indices.filter { holdId[$0] == number }
            let finite = frames.map { decided[$0] }.filter { $0.isFinite }
            if finite.isEmpty {
                continue
            }
            let forward = finite.filter { $0 > 0.0 }.count
            let backward = finite.filter { $0 < 0.0 }.count
            if forward > backward {
                for frame in frames { decided[frame] = 1.0 }
            } else if backward > forward {
                for frame in frames { decided[frame] = -1.0 }
            }
        }
        return decided
    }

    /// `com_u` in the athlete's own direction: positive towards the fingers —
    /// `com_forward`.
    ///
    /// The same multiplication in two places — once when the frame is
    /// measured and once when the hold has voted on its facing — so it is
    /// written once: a CoM to the right of the hands is *overbalanced* for an
    /// athlete facing right and *underbalanced* for one facing left.
    static func comForward(_ comU: [Double], _ facing: [Double]) -> [Double] {
        precondition(
            comU.count == facing.count, "com_u and facing must have the same shape")
        return zip(comU, facing).map { $0 * $1 }
    }

    /// Which side of the base of support each frame's CoM is on —
    /// `balance_zone`: `under` behind the heel of the hand, `over` in front of
    /// the fingertips, `ok` in between, and `nil` where `com_forward` is
    /// `NaN` — a frame with no CoM has no zone.
    static func balanceZone(_ forward: [Double]) -> [BalanceZone?] {
        forward.map { value in
            guard value.isFinite else { return nil }
            if value < -CentreOfMass.baseBack {
                return .under
            }
            if value > CentreOfMass.baseFront {
                return .over
            }
            return .ok
        }
    }
}
