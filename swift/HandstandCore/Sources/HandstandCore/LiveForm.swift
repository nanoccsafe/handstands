import Foundation

// --------------------------------------------------------------------------- #
// The live "form" detector (chainlink #97): one camera frame in, zero or more
// ``LiveFormEvent``s out — the mid-hold correction the user asked for after a
// gym session (2026-10-14): *"I was standing in a banana / mexican handstand
// for a hold and didn't realize it until after I saw the video."*
//
// It is the line detector's sibling (``LiveLineDetector``) and lives by the
// same three rules:
//
// * **causal.** `update` sees only this frame and what earlier frames left
//   behind — nothing is smoothed backwards, because the phrase has to be
//   spoken while the shape is still happening;
// * **timestamps, never frame counts.** The camera runs variable frame rate,
//   so every duration is measured from the `tMs` the frames carry;
// * **the same maths as the shift report.** The banana is `Features.banana`'s
//   single-frame body (a shared helper, not a copy) in the live body length
//   `L_live`, and its severity is `StressDiagram.valueSeverity` — the very
//   function the post-analysis stress diagram colours with. "Red" here *is*
//   red there.
//
// The earlier rule (user, 2026-10-06) was "live mode never corrects mid-hold".
// The user changed it at the gym, keeping the spirit: **one short phrase, only
// for a sustained red fault, and optional in Settings.** The pacing is now
// 'coach mode' (user, 2026-10-15, chainlink #100): while a fault stays red the
// phrase repeats every couple of seconds rather than firing once, so the
// athlete is guided back to the line instead of corrected a single time. A red
// that merely flickers is still waited out.
// --------------------------------------------------------------------------- #

/// When the live form detector speaks, and about what — chainlink #97.
///
/// The numbers are the hysteresis the issue asks for. Every duration is in
/// seconds *of frame timestamps*: the camera is variable frame rate, so a
/// frame count would make the same shape speak at a different moment on a
/// different capture.
public struct LiveFormConfig: Sendable, Equatable {
    /// How long the banana must stay red before a phrase is earned — the
    /// hysteresis that keeps one noisy frame from being corrected.
    public var redHoldS = 1.0

    /// A clear frame this short is a blink, not a recovery: the red run
    /// continues across it. Longer than this, and the red run restarts.
    public var gapS = 0.3

    /// Coach-mode pacing: while a fault stays red, an event fires every this
    /// many seconds — measured from the previous form event of the attempt —
    /// the first one after ``redHoldS`` of red.
    public var repeatS = 2.0

    /// At least this long between any two form phrases, whatever the attempt
    /// and whatever the direction. The same two seconds as ``repeatS``: a
    /// correction about the same fault comes at coach pace.
    public var minSpacingS = 2.0

    /// How long the hold gate must have been true before a phrase may fire.
    public var gateS = 0.5

    /// The largest `|wrist→ankle body angle|` (degrees) that still counts as a
    /// hold-like shape — `PhaseConfig.holdMaxAngle`, the offline hold gate.
    public var gateMaxAngleDeg = 35.0

    /// The confidence a joint needs to count (`Keypoint.visibility`); below it
    /// a joint is treated exactly as one nobody reported.
    public var minVisibility = 0.5

    public init() {}
}

/// One thing the live form detector noticed about a frame it has been fed.
///
/// Timestamps come from the frame's own `tMs` — the capture clock, variable
/// frame rate, never a fixed fps.
public enum LiveFormEvent: Equatable, Sendable {
    /// The line has curved towards the athlete's belly (a positive banana)
    /// and stayed red — say *"Ribs in."*.
    case arch(tMs: Int)

    /// The line has curved away from the belly (a negative banana) and stayed
    /// red — say *"Open the hips."*.
    case hollow(tMs: Int)

    /// The line is red but the nose cannot say which way it curves — say
    /// *"Straighten."*.
    case bent(tMs: Int)

    /// The hips are piked — the shoulder–hip–ankle angle is red — say
    /// *"Hips over shoulders."* (chainlink #100).
    case hips(tMs: Int)

    /// The direction this event corrects.
    public var fault: FormFault {
        switch self {
        case .arch: return .arch
        case .hollow: return .hollow
        case .bent: return .bent
        case .hips: return .hips
        }
    }

    /// When the frame that earned this event was captured, milliseconds.
    public var tMs: Int {
        switch self {
        case .arch(let tMs), .hollow(let tMs), .bent(let tMs), .hips(let tMs): return tMs
        }
    }
}

/// One live frame's banana: the magnitude the red test is judged on and the
/// signed value whose sign picks the direction — chainlink #97.
struct LiveBanana: Equatable {
    /// `|hipMid − shoulderMid→ankleMid line|` in live body lengths — what
    /// `StressDiagram.valueSeverity` is asked about.
    var magnitude: Double
    /// The signed banana (`magnitude × facing`), `NaN` when the nose cannot
    /// say which way the athlete faces.
    var signed: Double
}

/// Causal form detection over a stream of single frames — chainlink #97.
///
/// Feed one frame per call with its capture timestamp and the joints in
/// **display pixels, display orientation**; the events come back in the order
/// they happened. The detector keeps only what the next frame needs: whether
/// a hold is in progress, when the current red run began, and when the last
/// phrase was said.
public struct LiveFormDetector: Sendable {
    /// The rules this detector applies.
    public var config: LiveFormConfig

    /// A hold attempt ends once the gate has been false this long — the same
    /// one-second hysteresis the issue asks for, and not a number the user
    /// tunes, so it stays a constant rather than a config var.
    public static let attemptEndS = 1.0

    /// When the current hold attempt began (its first gate-true frame), `nil`
    /// outside one. Survives a gate break shorter than ``attemptEndS``.
    private var attemptStartTMs: Int?

    /// When the current run of gate-false frames inside an attempt began,
    /// `nil` while the gate holds — what ``attemptEndS`` times.
    private var gateBreakSinceTMs: Int?

    /// When the current run of red frames began, `nil` when the shape is not
    /// red (or unmeasurable). Survives a clear gap shorter than
    /// ``LiveFormConfig/gapS``.
    private var redSinceTMs: Int?

    /// When the current clear gap inside a red run began, `nil` while red.
    private var redGapSinceTMs: Int?

    /// When the last form phrase was earned — the global
    /// ``LiveFormConfig/minSpacingS`` and ``LiveFormConfig/repeatS`` spacings.
    /// Survives the attempt, so two holds close together do not both get a
    /// phrase.
    private var lastEventTMs: Int?

    public init(config: LiveFormConfig = .init()) {
        self.config = config
    }

    /// Is a hold attempt in progress (the gate has held for the last frame)?
    public var isInAttempt: Bool { attemptStartTMs != nil }

    // MARK: - One frame

    /// Feed one frame; get the events it produced, in time order.
    ///
    /// `tMs` must not go backwards (the capture clock does not). One call is
    /// one frame: nothing here reaches for a frame it has not seen.
    public mutating func update(tMs: Int, joints: [Joint: Keypoint]) -> [LiveFormEvent] {
        let inShape = Self.isHoldLike(joints: joints, config: config)
        updateAttempt(inShape: inShape, at: tMs)

        // Only inside a hold-like shape is a fault worth saying: outside one
        // there is no line to break.
        guard let attemptStart = attemptStartTMs, inShape else { return [] }

        let banana = Self.banana(joints: joints, config: config)
        let hipAngle = LiveLineDetector.hipAngleDeg(
            joints: joints, minVisibility: config.minVisibility)
        let fault = Self.chosenFault(banana: banana, hipAngleDeg: hipAngle)

        if let fault {
            redGapSinceTMs = nil
            if redSinceTMs == nil { redSinceTMs = tMs }
            guard let since = redSinceTMs else { return [] }
            let heldLongEnough = Double(tMs - since) >= config.redHoldS * 1000
            let gatedLongEnough = Double(tMs - attemptStart) >= config.gateS * 1000
            let repeated = lastEventTMs.map {
                Double(tMs - $0) >= config.repeatS * 1000
            } ?? true
            let spacedOut = lastEventTMs.map {
                Double(tMs - $0) >= config.minSpacingS * 1000
            } ?? true
            guard heldLongEnough, gatedLongEnough, repeated, spacedOut else {
                return []
            }
            lastEventTMs = tMs
            return [Self.event(fault, tMs: tMs)]
        }

        // Not red: time the gap that keeps a red run alive. Once it outlasts
        // `gapS` the red run is over, so the next red earns a fresh hold.
        if redSinceTMs != nil {
            if redGapSinceTMs == nil { redGapSinceTMs = tMs }
            if let gap = redGapSinceTMs, Double(tMs - gap) >= config.gapS * 1000 {
                redSinceTMs = nil
                redGapSinceTMs = nil
            }
        }
        return []
    }

    // MARK: - The hold attempt

    /// Track the hold-like shape into an attempt: open one on the first frame
    /// the gate passes, and end it once the gate has been false for
    /// ``attemptEndS`` — a shorter break is a wobble, not the athlete coming
    /// down.
    private mutating func updateAttempt(inShape: Bool, at tMs: Int) {
        if inShape {
            gateBreakSinceTMs = nil
            if attemptStartTMs == nil {
                attemptStartTMs = tMs
                beginAttempt()
            }
            return
        }
        guard attemptStartTMs != nil else { return }
        if gateBreakSinceTMs == nil { gateBreakSinceTMs = tMs }
        if let broken = gateBreakSinceTMs, Double(tMs - broken) >= Self.attemptEndS * 1000 {
            endAttempt()
        }
    }

    /// A fresh attempt: no red run yet, and nothing already spoken.
    private mutating func beginAttempt() {
        redSinceTMs = nil
        redGapSinceTMs = nil
    }

    /// The attempt is over (the gate has been false for a second): drop the
    /// red state. The global event spacing is deliberately *not* reset.
    private mutating func endAttempt() {
        attemptStartTMs = nil
        gateBreakSinceTMs = nil
        beginAttempt()
    }

    // MARK: - One frame's shape

    /// Is this frame a hold-like shape — the gate a form phrase needs?
    ///
    /// Inverted by ``LiveLineDetector/isInverted(joints:minVisibility:)``'s
    /// rule and leaning no more than ``LiveFormConfig/gateMaxAngleDeg`` off
    /// vertical (the offline `PhaseConfig.holdMaxAngle`). It is deliberately
    /// looser than ``LiveLineDetector/isLine(joints:config:)``: a banana that
    /// is not a line is exactly what this detector exists to see.
    static func isHoldLike(joints: [Joint: Keypoint], config: LiveFormConfig) -> Bool {
        guard LiveLineDetector.isInverted(joints: joints, minVisibility: config.minVisibility)
        else { return false }
        guard let wristMid = confidentMidpoint(
            joints, .leftWrist, .rightWrist, config.minVisibility),
            let ankleMid = confidentMidpoint(joints, .leftAnkle, .rightAnkle, config.minVisibility),
            let bodyAngle = LiveLineDetector.bodyAngleDeg(wristMid: wristMid, ankleMid: ankleMid)
        else { return false }
        return abs(bodyAngle) <= config.gateMaxAngleDeg
    }

    /// The live frame's banana, measured with the offline maths (chainlink
    /// #97).
    ///
    /// The off-line report divides every offset by the clip's body length `L`;
    /// a single frame has no clip, so the live scale is
    /// `L_live = |shoulderMid→hipMid| + |hipMid→ankleMid|` of this same frame
    /// (the midpoints of the confident sides, as `Features` averages them).
    /// The sign comes from the nose exactly as `Features.banana` does; with no
    /// confident nose it is unknown (`signed` is `NaN`) and only the magnitude
    /// is meaningful.
    ///
    /// `nil` when the shoulder, hip or ankle midpoint is not there: a banana
    /// nobody measured is not a banana.
    static func banana(joints: [Joint: Keypoint], config: LiveFormConfig) -> LiveBanana? {
        guard let shoulder = confidentMidpoint(
            joints, .leftShoulder, .rightShoulder, config.minVisibility),
            let hip = confidentMidpoint(joints, .leftHip, .rightHip, config.minVisibility),
            let ankle = confidentMidpoint(joints, .leftAnkle, .rightAnkle, config.minVisibility)
        else { return nil }

        let length = hypot(shoulder.x - hip.x, shoulder.y - hip.y)
            + hypot(hip.x - ankle.x, hip.y - ankle.y)
        guard length.isFinite, length > Features.minSegmentL else { return nil }

        // The body frame's own map (`u` right, `v` up, divided by `L`), with
        // the origin left at the display frame's `(0, 0)`: the banana is a
        // difference of midpoints, so where the origin sits cancels out.
        func bodyPoint(_ point: Keypoint) -> BodyPoint {
            BodyPoint(u: point.x / length, v: -point.y / length)
        }
        let shoulderPoint = bodyPoint(shoulder)
        let hipPoint = bodyPoint(hip)
        let anklePoint = bodyPoint(ankle)
        let offset = Features.bananaOffset(
            shoulderMid: shoulderPoint, hipMid: hipPoint, ankleMid: anklePoint)
        var signed = Double.nan
        if let nose = confidentMidpoint(joints, .nose, .nose, config.minVisibility) {
            let facing = Features.facingSign(
                shoulderMid: shoulderPoint, ankleMid: anklePoint, nose: bodyPoint(nose))
            signed = offset * facing
        }
        return LiveBanana(magnitude: abs(offset), signed: signed)
    }

    /// Is `magnitude` a red banana? — the severity the stress diagram uses for
    /// "no reference", then `Severity.colourBand`.
    static func isRed(_ magnitude: Double) -> Bool {
        guard let severity = bananaSeverity(magnitude) else { return false }
        return Severity.colourBand(severity) == .bad
    }

    /// Is `deg` a red hip (a pike), by the very severity the stress diagram
    /// colours `hip_angle` with (chainlink #100)? No copied threshold: the
    /// tolerance and the band are `StressDiagram`'s.
    static func isHipRed(_ deg: Double) -> Bool {
        guard let severity = hipSeverity(deg) else { return false }
        return Severity.colourBand(severity) == .bad
    }

    /// The stress diagram's severity for a banana magnitude, `nil` when it has
    /// no opinion (an unmeasured value).
    static func bananaSeverity(_ magnitude: Double) -> Double? {
        guard magnitude.isFinite else { return nil }
        return StressDiagram.valueSeverity(magnitude, feature: "banana", reference: nil)
    }

    /// The stress diagram's severity for a hip angle, `nil` when it has no
    /// opinion (an unmeasured value).
    static func hipSeverity(_ deg: Double) -> Double? {
        guard deg.isFinite else { return nil }
        return StressDiagram.valueSeverity(deg, feature: "hip_angle", reference: nil)
    }

    /// The fault to say on a frame where something is red — the "biggest fault
    /// first" choice (chainlink #100). `nil` when neither the banana nor the
    /// hip is red.
    ///
    /// The two faults overlap: an arch lowers the unsigned hip angle, so its
    /// severity is a floor under the hip's. The banana's *sign* is what tells
    /// them apart: an arch (positive banana) is preferred only when its
    /// severity is at least the hip's; any other red shape is a hip, which is
    /// the correction that opens the line. A negative or unknown (hollow /
    /// bent) banana keeps its own #97 fault only while the hip is not red.
    static func chosenFault(banana: LiveBanana?, hipAngleDeg: Double?) -> FormFault? {
        let bananaRed = banana.map { isRed($0.magnitude) } ?? false
        let hipRed = hipAngleDeg.map(isHipRed) ?? false
        guard bananaRed || hipRed else { return nil }

        if bananaRed, hipRed {
            if let banana, banana.signed > 0,
                let hipDeg = hipAngleDeg,
                let bananaSev = bananaSeverity(banana.magnitude),
                let hipSev = hipSeverity(hipDeg),
                bananaSev >= hipSev
            {
                return .arch
            }
            return .hips
        }
        if bananaRed, let banana {
            return fault(forSignedBanana: banana.signed)
        }
        return .hips
    }

    /// The direction a signed banana corrects: positive is the belly side
    /// (arch), negative the back side (hollow), and an unknown sign is the
    /// directionless "bent".
    static func fault(forSignedBanana value: Double) -> FormFault {
        if value > 0 { return .arch }
        if value < 0 { return .hollow }
        return .bent
    }

    /// The event for a direction.
    static func event(_ fault: FormFault, tMs: Int) -> LiveFormEvent {
        switch fault {
        case .arch: return .arch(tMs: tMs)
        case .hollow: return .hollow(tMs: tMs)
        case .bent: return .bent(tMs: tMs)
        case .hips: return .hips(tMs: tMs)
        }
    }

    /// Is a frame whose banana is red — the guard
    /// ``LiveLineDetector/isLine(joints:config:)`` calls so "Line. Hold it."
    /// never fires on a banana.
    static func isBananaRed(joints: [Joint: Keypoint], minVisibility: Double) -> Bool {
        var config = LiveFormConfig()
        config.minVisibility = minVisibility
        guard let banana = banana(joints: joints, config: config) else { return false }
        return isRed(banana.magnitude)
    }

    // MARK: - Joints

    /// The midpoint of the confident members of a joint pair — `Features`'
    /// side mean, over points: both when both are there, the one that is when
    /// only one is, `nil` when neither clears `minVisibility`. `left == right`
    /// for a single joint (the nose), which then reads as itself.
    private static func confidentMidpoint(
        _ joints: [Joint: Keypoint], _ left: Joint, _ right: Joint, _ minVisibility: Double
    ) -> Keypoint? {
        var points = [Keypoint]()
        for joint in left == right ? [left] : [left, right] {
            guard let point = joints[joint], point.visibility >= minVisibility,
                point.x.isFinite, point.y.isFinite
            else { continue }
            points.append(point)
        }
        guard let first = points.first else { return nil }
        guard points.count == 2 else { return first }
        return Keypoint(
            x: (first.x + points[1].x) / 2.0,
            y: (first.y + points[1].y) / 2.0,
            visibility: Swift.min(first.visibility, points[1].visibility)
        )
    }
}
