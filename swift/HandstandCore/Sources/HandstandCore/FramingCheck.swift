/// The live framing guide: is one athlete wholly inside the frame, close
/// enough, and alone in it?
///
/// This is the pure half of the Record screen (chainlink #46): it takes what
/// the detector saw in **one frame** and says what the user should do about
/// it. Every rule it applies is a constant at the top of this file, so the
/// rules can be read and tested without a camera, a frame or Vision anywhere
/// near them — the AVFoundation half lives in `ios/HandstandApp/Capture/`.
///
/// Coordinates are **normalised image coordinates**: `0...1`, origin at the
/// top left, `y` growing downwards — the display frame the user sees, *not*
/// Vision's bottom-left origin (the app flips Vision's points into this space
/// before calling in). One person is one `[Joint: Keypoint]`, where
/// `Keypoint.x` / `.y` are those normalised coordinates and
/// `Keypoint.visibility` is the detector's confidence for that joint. A joint
/// the detector did not report is simply absent from the dictionary.
///
/// The guide assumes what `docs/recording_protocol.md` assumes: one athlete
/// alone in frame, phone on a tripod.
public enum FramingCheck {
    // MARK: - The rules

    /// A joint only *counts* when the detector is at least this sure of it.
    /// Below this a joint is treated exactly like a joint nobody reported:
    /// the body may be there, but the guide cannot vouch for it.
    public static let confidentThreshold: Double = 0.3

    /// How close to an edge — in normalised units, so `0.04` of the frame's
    /// width or height — a required joint may sit before that edge is treated
    /// as cutting the body off. Small enough that a person standing normally
    /// inside the frame never trips it, small enough that a foot already
    /// halfway out of the picture does.
    public static let margin: Double = 0.04

    /// The smallest body the guide will happily record: the vertical extent
    /// (max y − min y over the confident joints) must reach this fraction of
    /// the frame's height. `0.35` means the athlete fills about a third of
    /// the picture top to bottom — any less and the pixels are too few for
    /// the analysis later on.
    public static let minimumBodyHeight: Double = 0.35

    /// How many confident joints a body needs before a *second* body in the
    /// same frame counts as a second athlete rather than a passer-by's arm.
    /// Six is fewer than the eight required joints below but more than a
    /// stray detection: two real people always clear it.
    public static let significantJointCount: Int = 6

    /// The joints the guide watches — both wrists, ankles, shoulders and
    /// hips, in that order. They are the ends and the middle of the body: a
    /// wrist or an ankle near an edge is a hand or a foot about to leave the
    /// picture, a shoulder or a hip near one means the athlete is standing
    /// too close to the side of it. Each side is checked on its own, so a
    /// body whose left wrist alone is missing is reported for that wrist.
    ///
    /// Knees, elbows and the nose are deliberately absent from this list —
    /// they say nothing about the frame's edges — but *every* confident joint
    /// counts towards the vertical extent.
    public static let requiredJoints: [Joint] = [
        .leftWrist, .rightWrist,
        .leftAnkle, .rightAnkle,
        .leftShoulder, .rightShoulder,
        .leftHip, .rightHip,
    ]

    // MARK: - The verdict

    /// Evaluate one frame's detections.
    ///
    /// The order the statuses are tried in is a deliberate priority rather
    /// than the declaration order of ``FramingStatus``:
    ///
    /// 1. `.noPerson` — nothing confident was seen at all;
    /// 2. `.multiplePeople` — the app records *one* athlete, so two bodies
    ///    that both look real outrank everything else;
    /// 3. `.partlyOutOfFrame` — a body cut off at an edge is fixed by stepping
    ///    back whatever its size, and telling someone whose foot is already
    ///    cut off to "move closer" would make it worse, so the edge check
    ///    comes *before* the size check;
    /// 4. `.tooSmall` — the body is whole but too far away;
    /// 5. `.ok`.
    public static func status(people: [[Joint: Keypoint]]) -> FramingStatus {
        let counts = people.map(confidentCount(in:))
        // The person the verdict is about: whoever has the most confident
        // joints (first wins a tie). `nil` means nobody had even one, which
        // is `.noPerson` whether the detector reported people or not.
        guard let best = counts.indices.max(by: { counts[$0] < counts[$1] }), counts[best] > 0 else {
            return .noPerson
        }
        if counts.filter({ $0 >= significantJointCount }).count > 1 {
            return .multiplePeople
        }

        let primary = people[best]
        var edgeHits: Set<Edge> = []
        var missing: [Joint] = []
        for joint in requiredJoints {
            guard let point = primary[joint], point.visibility >= confidentThreshold else {
                missing.append(joint)
                continue
            }
            edgeHits.formUnion(edges(of: point))
        }
        if !edgeHits.isEmpty || !missing.isEmpty {
            return .partlyOutOfFrame(edges: edgeHits, missing: missing)
        }

        let heights = primary.values
            .filter { $0.visibility >= confidentThreshold }
            .map(\.y)
        if let lowest = heights.min(), let highest = heights.max(),
           highest - lowest < minimumBodyHeight {
            return .tooSmall
        }
        return .ok
    }

    /// What the user reads under the border for a verdict — one sentence per
    /// status, tested like the verdicts themselves so the guide can never
    /// show a stray `Optional(...)` or an untranslated enum name on screen.
    public static func message(for status: FramingStatus) -> String {
        switch status {
        case .noPerson:
            return "Step into the frame"
        case .multiplePeople:
            // The app assumes a single athlete (user decision 2026-09-27):
            // say so, rather than guessing which body is the one being
            // recorded.
            return "Only you in the frame please"
        case .tooSmall:
            return "Move closer"
        case .ok:
            return "Looks good"
        case let .partlyOutOfFrame(edges, missing):
            return partlyMessage(edges: edges, missing: missing)
        }
    }

    // MARK: - The pieces

    /// How many of a person's joints are confidently detected.
    private static func confidentCount(in person: [Joint: Keypoint]) -> Int {
        person.values.filter { $0.visibility >= confidentThreshold }.count
    }

    /// Which frame edges a point is standing on (or over: a coordinate
    /// outside `0...1` is beyond the edge on that side, and the comparisons
    /// below catch it too).
    private static func edges(of point: Keypoint) -> Set<Edge> {
        var found: Set<Edge> = []
        if point.y <= margin { found.insert(.top) }
        if point.y >= 1 - margin { found.insert(.bottom) }
        if point.x <= margin { found.insert(.left) }
        if point.x >= 1 - margin { found.insert(.right) }
        return found
    }

    /// The sentence for `.partlyOutOfFrame`: name the edges that are cutting
    /// the body off and the joints nobody could see. Both halves are optional
    /// (a status built by hand may carry neither), so the sentence degrades
    /// to a generic "step back" rather than reading "Step back: ".
    private static func partlyMessage(edges: Set<Edge>, missing: [Joint]) -> String {
        var clauses: [String] = []
        if !edges.isEmpty {
            // `Edge.allCases` order, not `Set` iteration order: the same
            // status must always read the same sentence.
            let names = Edge.allCases.filter { edges.contains($0) }.map(edgePhrase)
            clauses.append("you are cut off at \(list(names))")
        }
        var words: [String] = []
        for joint in requiredJoints where missing.contains(joint) {
            guard let word = bodyWord(for: joint), !words.contains(word) else { continue }
            words.append(word)
        }
        if !words.isEmpty {
            clauses.append("your \(list(words)) are out of frame")
        }
        guard !clauses.isEmpty else {
            return "Step back: part of your body is out of frame"
        }
        return "Step back: \(list(clauses))"
    }

    /// One edge as a phrase: "the top of the frame", "the left edge".
    private static func edgePhrase(_ edge: Edge) -> String {
        switch edge {
        case .top: return "the top of the frame"
        case .bottom: return "the bottom of the frame"
        case .left: return "the left edge"
        case .right: return "the right edge"
        }
    }

    /// What a required joint is called to the user. Always plural — every
    /// required joint has a left and a right — so the sentence always reads
    /// "are out of frame".
    private static func bodyWord(for joint: Joint) -> String? {
        switch joint {
        case .leftWrist, .rightWrist: return "hands"
        case .leftAnkle, .rightAnkle: return "feet"
        case .leftShoulder, .rightShoulder: return "shoulders"
        case .leftHip, .rightHip: return "hips"
        default: return nil
        }
    }

    /// "a", "a and b", "a, b and c" — the join the messages are built from.
    private static func list(_ items: [String]) -> String {
        switch items.count {
        case 0: return ""
        case 1: return items[0]
        case 2: return "\(items[0]) and \(items[1])"
        default:
            return items.dropLast().joined(separator: ", ") + " and " + items[items.count - 1]
        }
    }
}

/// One of the frame's four edges, in the order the sentences name them.
///
/// The app's border is drawn from these (via `FramingStatus`), which is why
/// they are display edges — top/bottom/left/right of the picture as the user
/// sees it, not of any buffer Vision happened to read.
public enum Edge: String, CaseIterable, Sendable {
    case top
    case bottom
    case left
    case right
}

/// What the framing guide decided about one frame, and therefore what colour
/// the border is and what the user is told to do.
public enum FramingStatus: Equatable, Sendable {
    /// No confidently detected body: the frame is empty, or whatever is in it
    /// could not be read well enough to judge.
    case noPerson
    /// Two or more bodies that each look like a real athlete. The app records
    /// one attempt by one person, so the frame is refused until the others
    /// leave it.
    case multiplePeople
    /// One body, wholly in frame, but too small in the frame to analyse well.
    case tooSmall
    /// One body that is not wholly in frame.
    ///
    /// - `edges`: the frame edges a required joint is standing on — the edges
    ///   that are cutting the body off.
    /// - `missing`: the required joints nobody could see at all (absent from
    ///   the detection, or below ``FramingCheck/confidentThreshold``). A joint
    ///   that *was* seen sitting on an edge is reported in `edges`, not here:
    ///   it is clipped, not missing.
    case partlyOutOfFrame(edges: Set<Edge>, missing: [Joint])
    /// One body, whole in the frame, big enough, alone in it.
    case ok
}
