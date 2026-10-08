import Foundation

// --------------------------------------------------------------------------- #
// The live "line" detector (chainlink #91, live mode stage 1): one camera frame
// in, zero or more ``LiveEvent``s out, with no camera, no audio and no model of
// its own — the app feeds it the joints MediaPipe found, and this file decides
// whether that frame *is* a line and how long it has been one.
//
// Three rules keep it honest against the analysis that runs afterwards:
//
// * **causal.** `update` sees only this frame and what earlier frames left
//   behind; nothing is smoothed backwards over a window, because the cue has to
//   be spoken while the moment is happening;
// * **timestamps, never frame counts.** The camera runs variable frame rate,
//   so onset and loss are measured in the milliseconds `tMs` carries — exactly
//   as `PhaseSegmenter` measures them offline;
// * **stricter than the hold gate, looser than the fault tolerance.** "Line"
//   asks for a 15° body angle and a 160° hip (the offline fault gates are 10°
//   and 165°, the offline hold gate is a much looser 35°), so a cue fires on a
//   shape the analysis will call a hold and never on one it will call a fault.
//
// The geometry is `BodyFrame`'s and `Features`' own — the wrist→ankle angle is
// `atan2(u, v)` in the body frame exactly as `PhaseSegmenter.frameSignals`
// computes it, and the hip angle is `Features.angleAt`, so the live cue and the
// offline report can never drift apart on the maths.
// --------------------------------------------------------------------------- #

/// What the detector needs to see before it calls a frame a line, and how long
/// the shape has to last before it says so.
public struct LiveLineConfig: Sendable, Equatable {
    /// The largest `|wrist→ankle angle from vertical|` (degrees) a line may
    /// have — looser than the 10° fault tolerance the report scores with,
    /// stricter than the 35° `PhaseConfig.holdMaxAngle` gate a hold needs, so
    /// "line" means *visibly* straight without demanding perfection mid-hold.
    public var bodyAngleMaxDeg = 15.0

    /// The smallest shoulder–hip–ankle angle (degrees) a line may have: 160°
    /// leaves no room for a pike or a tuck (the offline pike fault is 165°).
    public var hipAngleMinDeg = 160.0

    /// How long the shape must hold before `lineAchieved` fires — the hysteresis
    /// that keeps one noisy frame from being cheered.
    public var onsetS = 0.5

    /// How long a gap may last before an achieved line is reported lost — the
    /// same hysteresis the other way, so a blink in the detection does not end
    /// a line.
    public var lossS = 0.4

    /// The confidence a joint needs to count (`Keypoint.visibility`); below it a
    /// joint is treated exactly as one nobody reported.
    public var minVisibility = 0.5

    public init() {}
}

/// One thing the live detector noticed about the frames it has been fed.
///
/// Timestamps come from the frame's own `tMs` — the capture clock, variable
/// frame rate, never a fixed fps.
public enum LiveEvent: Equatable, Sendable {
    /// The shape became a line and has now held for
    /// ``LiveLineConfig/onsetS`` — once per line, the moment to speak
    /// *"Line. Hold it."*
    case lineAchieved(tMs: Int)

    /// An achieved line ended: `heldS` is how long it lasted (the first line
    /// frame to the last one), `tMs` the frame the loss was confirmed on.
    case lineLost(tMs: Int, heldS: Double)

    /// `seconds` of line time have passed while in line — the optional
    /// *"Five."* / *"Ten."* marks, one per `timeMarksEverySeconds`.
    case timeMark(tMs: Int, seconds: Int)
}

/// Causal line detection over a stream of single frames — chainlink #91.
///
/// Feed one frame per call with its capture timestamp and the joints in
/// **display pixels, display orientation** (the frame as the user sees it);
/// the events come back in the order they happened. The detector keeps only
/// what the next frame needs: when the current line began, whether it has
/// already been announced, and when a gap began.
public struct LiveLineDetector: Sendable {
    /// The rules this detector applies.
    public var config: LiveLineConfig

    /// Every how-many seconds of line time a `timeMark` fires while in line,
    /// or `nil` for no marks at all. Non-positive values read as `nil`.
    public let timeMarksEverySeconds: Int?

    /// When the current run of line frames began, `nil` outside one.
    private var lineSinceTMs: Int?

    /// The last frame that was a line — `heldS` in `lineLost` is measured to it.
    private var lastLineTMs: Int?

    /// Has `lineAchieved` already fired for the current run? Only a run that
    /// was announced can be announced as lost: a flicker too short to cue says
    /// nothing at all.
    private var announced = false

    /// When the current run of *non-line* frames began, `nil` while in line —
    /// the gap the `lossS` hysteresis times.
    private var breakSinceTMs: Int?

    /// The next time mark's second count, `nil` while marks are off or the
    /// current line has fired every one it earns.
    private var nextMarkSeconds: Int?

    /// The last frame's answer to "is the athlete upside down?" — the state
    /// the app's orientation rule and the cue scheduler read.
    private var invertedNow = false

    /// The last frame's answer to "are we in a *confirmed* line?" — true from
    /// `lineAchieved` until `lineLost`, so a one-frame blink inside the loss
    /// window keeps it true.
    private var inLineNow = false

    public init(config: LiveLineConfig = .init(), timeMarksEverySeconds: Int? = nil) {
        self.config = config
        self.timeMarksEverySeconds = timeMarksEverySeconds.flatMap { $0 > 0 ? $0 : nil }
    }

    /// Is a confirmed line in progress (announced, not yet lost)?
    public var isInLine: Bool { inLineNow }

    /// Was the athlete upside down in the frame just fed? Confident joints
    /// only — wrists below ankles, or below hips when the ankles are missing.
    public var isInverted: Bool { invertedNow }

    // MARK: - One frame

    /// Feed one frame; get the events it produced, in time order.
    ///
    /// `tMs` must not go backwards (the capture clock does not). One call is
    /// one frame: nothing here reaches for a frame it has not seen.
    public mutating func update(tMs: Int, joints: [Joint: Keypoint]) -> [LiveEvent] {
        invertedNow = Self.isInverted(joints: joints, minVisibility: config.minVisibility)
        var events = [LiveEvent]()

        if Self.isLine(joints: joints, config: config) {
            events.append(contentsOf: lineFrame(at: tMs))
        } else {
            events.append(contentsOf: nonLineFrame(at: tMs))
        }
        inLineNow = announced
        return events
    }

    /// A frame that is a line: open a run if there was none, then check the
    /// onset and the time marks against this frame's timestamp.
    private mutating func lineFrame(at tMs: Int) -> [LiveEvent] {
        breakSinceTMs = nil
        lastLineTMs = tMs
        if lineSinceTMs == nil {
            lineSinceTMs = tMs
            announced = false
            nextMarkSeconds = timeMarksEverySeconds
        }
        guard let since = lineSinceTMs else { return [] }

        var events = [LiveEvent]()
        if !announced, Double(tMs - since) >= config.onsetS * 1000 {
            announced = true
            events.append(.lineAchieved(tMs: tMs))
        }
        // Marks are line time, counted from the line's own first frame — and
        // only while the line is there: a mark spoken into a gap would be a
        // mark about nothing. A `while`, because one sparse (variable frame
        // rate) frame may be the first one past several marks at once.
        if announced {
            let every = timeMarksEverySeconds ?? 0
            while every > 0, let next = nextMarkSeconds,
                Double(tMs - since) >= Double(next) * 1000
            {
                events.append(.timeMark(tMs: tMs, seconds: next))
                nextMarkSeconds = next + every
            }
        }
        return events
    }

    /// A frame that is not a line: start the loss clock if the line had begun,
    /// and end it once the gap outlasts `lossS`.
    private mutating func nonLineFrame(at tMs: Int) -> [LiveEvent] {
        guard let since = lineSinceTMs else { return [] }
        if breakSinceTMs == nil {
            breakSinceTMs = tMs
        }
        guard let breakSince = breakSinceTMs,
            Double(tMs - breakSince) >= config.lossS * 1000
        else { return [] }

        var events = [LiveEvent]()
        if announced, let last = lastLineTMs {
            events.append(.lineLost(tMs: tMs, heldS: Double(last - since) / 1000.0))
        }
        lineSinceTMs = nil
        lastLineTMs = nil
        announced = false
        breakSinceTMs = nil
        nextMarkSeconds = nil
        return events
    }

    // MARK: - One frame's geometry

    /// Is the athlete upside down — wrists below ankles, or below hips when
    /// the ankles are missing?
    ///
    /// Confident joints only: a group whose members are missing or below
    /// `minVisibility` counts as not seen, and an unjudgeable frame is "not
    /// inverted" (the safe answer — it never cheers a standing athlete).
    static func isInverted(joints: [Joint: Keypoint], minVisibility: Double) -> Bool {
        guard let wrists = confident(joints, [.leftWrist, .rightWrist], minVisibility),
            let wristMid = midpoint(wrists)
        else { return false }
        if let ankles = confident(joints, [.leftAnkle, .rightAnkle], minVisibility),
            let ankleMid = midpoint(ankles) {
            return ankleMid.y < wristMid.y
        }
        if let hips = confident(joints, [.leftHip, .rightHip], minVisibility),
            let hipMid = midpoint(hips) {
            return hipMid.y < wristMid.y
        }
        return false
    }

    /// Is this frame a line? Inverted, straight wrist→ankle, open at the hip,
    /// with wrists, hips and ankles all confident — the whole definition in
    /// one place, in the order the checks are cheapest to fail.
    static func isLine(joints: [Joint: Keypoint], config: LiveLineConfig) -> Bool {
        guard isInverted(joints: joints, minVisibility: config.minVisibility) else { return false }
        guard let wrists = confident(joints, [.leftWrist, .rightWrist], config.minVisibility),
            let ankles = confident(joints, [.leftAnkle, .rightAnkle], config.minVisibility),
            confident(joints, [.leftHip, .rightHip], config.minVisibility) != nil,
            let wristMid = midpoint(wrists), let ankleMid = midpoint(ankles)
        else { return false }

        guard let bodyAngle = Self.bodyAngleDeg(wristMid: wristMid, ankleMid: ankleMid),
            abs(bodyAngle) <= config.bodyAngleMaxDeg
        else { return false }
        guard let hipAngle = Self.hipAngleDeg(joints: joints, minVisibility: config.minVisibility),
            hipAngle >= config.hipAngleMinDeg
        else { return false }
        return true
    }

    /// The wrist→ankle vector's angle from straight up, signed by the way it
    /// leans — `atan2(u, v)` of the body frame, exactly the number
    /// `PhaseSegmenter.frameSignals` calls `bodyAngleDeg`.
    ///
    /// The body length cancels in the ratio (`atan2(u/L, v/L)` is
    /// `atan2(u, v)`), so the live frame is measured in its own pixels with a
    /// body length of 1: no scale to estimate from a single frame.
    static func bodyAngleDeg(wristMid: Keypoint, ankleMid: Keypoint) -> Double? {
        guard let uv = try? BodyFrame.toBodyFrame(
            x: ankleMid.x,
            y: ankleMid.y,
            wristMidX: wristMid.x,
            wristMidY: wristMid.y,
            bodyLength: 1.0
        ) else { return nil }
        guard uv.u.isFinite, uv.v.isFinite, uv.u != 0 || uv.v != 0 else { return nil }
        return atan2(uv.u, uv.v) * (180.0 / Double.pi)
    }

    /// The shoulder–hip–ankle angle (degrees) of **the visible side**: the
    /// side whose three joints are all confident, preferring the one the
    /// detector is most sure of — `Features.angleAt`, the same measurement the
    /// `hip_angle` feature is built from. `nil` when no side can be measured.
    static func hipAngleDeg(joints: [Joint: Keypoint], minVisibility: Double) -> Double? {
        var best: Double?
        var bestScore = -Double.infinity
        for side in [(Joint.leftShoulder, Joint.leftHip, Joint.leftAnkle),
                     (Joint.rightShoulder, Joint.rightHip, Joint.rightAnkle)]
        {
            guard let shoulder = confident(joints, [side.0], minVisibility)?.first,
                let hip = confident(joints, [side.1], minVisibility)?.first,
                let ankle = confident(joints, [side.2], minVisibility)?.first
            else { continue }
            let score = (shoulder.visibility + hip.visibility + ankle.visibility) / 3.0
            guard score > bestScore else { continue }
            let angle = Features.angleAt(
                [BodyPoint(u: shoulder.x, v: -shoulder.y)],
                [BodyPoint(u: hip.x, v: -hip.y)],
                [BodyPoint(u: ankle.x, v: -ankle.y)]
            )[0]
            guard angle.isFinite else { continue }
            bestScore = score
            best = angle
        }
        return best
    }

    /// The two members of a joint group, or `nil` when any of them is missing
    /// or not confident enough.
    private static func confident(
        _ joints: [Joint: Keypoint],
        _ group: [Joint],
        _ minVisibility: Double
    ) -> [Keypoint]? {
        var points = [Keypoint]()
        points.reserveCapacity(group.count)
        for joint in group {
            guard let point = joints[joint], point.visibility >= minVisibility,
                point.x.isFinite, point.y.isFinite
            else { return nil }
            points.append(point)
        }
        return points
    }

    /// Both points' midpoint, or `nil` when there are not two of them —
    /// `BodyFrame.midpoint`, visibility and all.
    private static func midpoint(_ points: [Keypoint]) -> Keypoint? {
        guard points.count == 2 else { return nil }
        return BodyFrame.midpoint(points[0], points[1])
    }
}
