import Foundation

// --------------------------------------------------------------------------- #
// Phase segmentation: pre, kick-up, hold, exit, post — labelled on every frame.
//
// A port of the phase half of `pipeline/handstand/phases.py` (chainlink #40),
// in the same order and with the same edge cases: the constants, the
// measurements, the hold runs and the state machine. The features, the centre
// of mass and the hold summary are chainlink #81; the file I/O, the segment
// tables, the `ClipStats` summary, the overlay and the CLI are not ported at
// all — this is the maths, nothing else.
//
// The rules the port lives by, taken straight from Python:
//
// * every duration is measured against `tMs` converted to seconds — these
//   clips are variable frame rate, so nothing assumes a fixed fps;
// * `NaN` comparisons are false, as numpy's are: an unmeasured signal is never
//   read as a threshold that passed;
// * `_TIME_EPS` is the slack on both sides of a measurement window, so a
//   window that lands exactly on a frame is not decided by the last bit of a
//   timestamp;
// * a clip with no body length comes back with every frame `unknown`, `hold_id`
//   -1, and the reason string Python writes.
// --------------------------------------------------------------------------- #

/// The six labels, one per frame of every clip, in the order a clip passes
/// through them — `pipeline/handstand/phases.PHASES`.
///
/// Declaration order *is* `PHASES`: `Phase.allCases.map(\.rawValue)` is the
/// Python tuple element for element, and `rawValue` is the name the golden
/// fixture rows carry. Python's state machine works on the ints
/// `PRE, KICKUP, HOLD, EXIT, POST, UNKNOWN`; this one works on the cases
/// directly, which is the same machine with the names left on.
public enum Phase: String, CaseIterable, Sendable {
    /// Standing, walking up: anything before the hands are down.
    case pre
    /// Hands down, the legs going up, not yet holding — the attempt.
    case kickup
    /// Inverted, straight, the hands planted and no hand step, sustained for
    /// at least `PhaseConfig.minHoldS`.
    case hold
    /// After a hold: the legs falling, no longer inverted, or the hands on the
    /// floor without holding — until the feet land or the hands leave the
    /// floor.
    case exit
    /// After the last exit. The body is no longer above its hands, so the clip
    /// is not about a handstand any more.
    case post
    /// A frame the model could not see into: trainer contact, or a wrist, an
    /// ankle or a hip with no position.
    case unknown
}

/// The tuning knobs of the phase segmenter: one property per constant of
/// `pipeline/handstand/phases.py` (each names its Python twin in a comment),
/// with the same defaults. `.init()` is exactly the Python pipeline's
/// configuration, which is what the golden fixtures were computed with, and
/// `PhasesTests.testTheConstantsAreThePythonOnes` pins every one of them down.
///
/// The one constant of the list that is *not* here is `PHASES`: it is
/// `Phase`'s declaration order rather than a number to tune.
public struct PhaseConfig: Sendable, Equatable {
    /// `NO_HOLD` — `hold_id` outside a hold. A negative number cannot be a
    /// hold's number, and a number keeps the column numeric.
    public var noHold = -1
    /// `INVERTED_MIN` — how far above the wrist midpoint the ankle midpoint
    /// has to be before the athlete counts as inverted, in body lengths. A
    /// margin, not a sign change: a body on the floor never gets there.
    public var invertedMin = 0.6
    /// `HOLD_MAX_ANGLE` — how far the wrist→ankle vector may lean from
    /// straight up inside a hold, in degrees.
    public var holdMaxAngle = 35.0
    /// `MIN_HOLD_S` — how long a run of hold frames has to last to count as a
    /// hold, in seconds.
    public var minHoldS = 0.3
    /// `HAND_STILL_L_PER_S` — how fast a wrist may move and still count as
    /// planted, in body lengths per second.
    public var handStillLPerS = 0.3
    /// `HAND_STILL_WINDOW_S` — the window both hand measurements are taken
    /// over, in seconds.
    public var handStillWindowS = 0.2
    /// `HAND_STEP_L` — how far a wrist may move over `handStillWindowS` and
    /// still be the same hold, in body lengths.
    public var handStepL = 0.1
    /// `HANDS_LOW_MIN_V` — how far above the wrist midpoint the hip midpoint
    /// has to be for the hands to count as the support, in body lengths.
    public var handsLowMinV = 0.1
    /// `LEG_VELOCITY_MIN_L_PER_S` — the deadband on the ankle midpoint's
    /// vertical velocity, in body lengths per second.
    public var legVelocityMinLPerS = 0.1
    /// `LEG_VELOCITY_WINDOW_S` — the window the vertical velocity is measured
    /// over, in seconds (centred).
    public var legVelocityWindowS = 0.1
    /// `HOLD_BREAK_MAX_S` — the longest stretch of frames that may fail to be
    /// a hold without ending one, in seconds. A hand step is never bridged.
    public var holdBreakMaxS = 0.3
    /// `REASON_TRAINER` — why a frame the athlete selection flagged is
    /// `unknown`.
    public var reasonTrainer = "trainer_contact"
    /// `REASON_WRISTS` — why a frame with no visible wrist is `unknown`.
    public var reasonWrists = "no_visible_wrist"
    /// `REASON_ANKLES` — why a frame with no visible ankle is `unknown`.
    public var reasonAnkles = "no_visible_ankle"
    /// `REASON_HIPS` — why a frame with no visible hip is `unknown`.
    public var reasonHips = "no_visible_hip"
    /// `WRIST_JOINTS` — the joints the origin and the hand measurements are
    /// read off, in order. One visible wrist is enough.
    public var wristJoints: [Joint] = [.leftWrist, .rightWrist]
    /// `ANKLE_JOINTS` — the joints the "upside down" test is read off.
    public var ankleJoints: [Joint] = [.leftAnkle, .rightAnkle]
    /// `HIP_JOINTS` — the joints `handsLow` is read off.
    public var hipJoints: [Joint] = [.leftHip, .rightHip]
    /// `_TIME_EPS` — slack, in seconds, when a measurement window is compared
    /// against a timestamp: it decides nothing about the data, it only stops a
    /// window that lands exactly on a frame from being decided by the last bit
    /// of a float.
    public var timeEps = 1e-9

    public init() {}
}

/// One hold: the frames it covers, inclusive, and the number it is —
/// `handstand.phases.HoldRun`.
public struct HoldRun: Sendable, Equatable {
    /// The first frame of the run.
    public var start: Int
    /// The last frame of the run, inclusive.
    public var end: Int
    /// The hold's number, `PhaseSegmenter.noHold` until `holdRuns` numbers the
    /// runs it keeps — Python's `hold_id`.
    public var holdId: Int = PhaseSegmenter.noHold

    /// How many frames the run covers, including any bridged unknown ones —
    /// `HoldRun.frames`.
    public var frames: Int { end - start + 1 }
}

/// Every per-frame measurement the state machine reads, one array per signal,
/// all `(frames,)` in frame order — `handstand.phases.FrameSignals`.
///
/// All of them are in the body frame's units: positions and lengths in body
/// lengths, angles in degrees, speeds in body lengths per second. Every
/// boolean is `false` on a frame the module cannot see into: `known` is the
/// array to filter on and `unknownReason` says which joint, or which person,
/// was in the way.
public struct FrameSignals: Sendable, Equatable {
    /// Can the module see this frame — wrists, ankles and hips visible and no
    /// trainer in front of the camera.
    public var known: [Bool]
    /// Why the frame is `unknown`: `""` on a known frame, otherwise one of
    /// `PhaseConfig`'s `reason*` strings.
    public var unknownReason: [String]
    /// The ankle midpoint's height above the wrist midpoint, body lengths.
    public var vAnkleMid: [Double]
    /// The ankle midpoint's horizontal offset from the wrist midpoint, body
    /// lengths (right positive).
    public var uAnkleMid: [Double]
    /// The hip midpoint's height above the wrist midpoint, body lengths.
    public var vHipMid: [Double]
    /// The wrist→ankle vector's lean from straight up, signed, in degrees.
    public var bodyAngleDeg: [Double]
    /// `known` and the ankle midpoint above `invertedMin`.
    public var inverted: [Bool]
    /// `known` and the hip midpoint above `handsLowMinV` — the hands are the
    /// support.
    public var handsLow: [Bool]
    /// How fast the fastest visible wrist moved over the window, body lengths
    /// per second; `NaN` where the window could not be asked.
    public var wristSpeedLPerS: [Double]
    /// How far the fastest visible wrist moved over the window, body lengths;
    /// `NaN` where the window could not be asked.
    public var wristStepL: [Double]
    /// `handsLow` and every measurable wrist slower than `handStillLPerS`
    /// (an unmeasurable speed is not evidence that the hands moved).
    public var handsDown: [Bool]
    /// `known` and a wrist that moved further than `handStepL` over the
    /// window: a hand step, which ends the current hold at once.
    public var handStep: [Bool]
    /// The ankle midpoint's vertical velocity over `legVelocityWindowS`,
    /// body lengths per second; `NaN` where it could not be measured.
    public var legsVelocityLPerS: [Double]
    /// `known` and the legs rising faster than the deadband.
    public var legsRising: [Bool]
    /// `known` and the legs falling faster than the deadband.
    public var legsFalling: [Bool]

    /// How many frames the clip has — `FrameSignals.frames`.
    public var frames: Int { known.count }

    /// The geometry of a hold on its own: known, inverted and straight —
    /// `FrameSignals.upside_down`.
    ///
    /// `holdCondition` adds the hands; this is what a later stage asks when it
    /// only wants to know whether the athlete was upside down at all.
    public func upsideDown(config: PhaseConfig = .init()) -> [Bool] {
        var out = [Bool](repeating: false, count: known.count)
        for index in known.indices {
            out[index] =
                known[index] && inverted[index] && abs(bodyAngleDeg[index]) <= config.holdMaxAngle
        }
        return out
    }
}

/// One clip's per-frame answer, with the numbers it was made from —
/// `handstand.phases.ClipPhases` minus the clip identity and the frame index
/// (there is no file to write here, so the frame's index is its array index).
public struct ClipPhases: Sendable, Equatable {
    /// The clip's frame timestamps, milliseconds — the only clock these clips
    /// have, and variable from frame to frame.
    public var tMs: [Int]
    /// The phase of each frame, in frame order.
    public var phase: [Phase]
    /// The number of the hold each frame is inside, `noHold` outside one.
    public var holdId: [Int]
    /// The measurements the answer was made of.
    public var signals: FrameSignals
    /// The holds that were long enough to count, numbered from 0 in time
    /// order.
    public var runs: [HoldRun] = []
    /// Does this clip have a body length, i.e. any usable scale at all.
    public var usable: Bool = true
    /// Why the clip is unusable; `""` when it is — Python's `reason`.
    public var unusableReason: String = ""

    /// How many holds the clip has — `ClipPhases.hold_count`.
    public var holdCount: Int { runs.count }

    /// The duration of every hold in seconds, in time order —
    /// `ClipPhases.hold_durations_s`.
    public func holdDurationsS() -> [Double] {
        let times = tMs.map { Double($0) / 1000.0 }
        return runs.map { times[$0.end] - times[$0.start] }
    }
}

/// Port of the phase segmenter of `pipeline/handstand/phases.py`: the
/// measurements of `frame_signals`, the runs of `hold_runs` and the state
/// machine of `assign_phases`, in that order — `classify_clip` is the entry
/// point.
///
/// The rules that make the labels match Python:
///
/// * time comes from each frame's `tMs` — variable frame rate, so a window is
///   found by timestamp with `np.searchsorted`, never by frame count;
/// * a `NaN` comparison is false, and a `NaN` speed is *not* evidence that the
///   hands moved (that is what lets a trainer's occlusion pass through a hold);
/// * an unknown stretch of at most `holdBreakMaxS` inside a hold does not
///   split it, but a hand step ends it wherever it happens;
/// * a clip with no body length comes back with every frame `unknown`,
///   `hold_id` -1 and Python's reason string.
///
/// The helpers (`midpoint`, `largest`, `bodyFrameUV`, `windowMotion`,
/// `velocityLPerS`, `frameSignals`, `holdCondition`, `holdRuns`,
/// `assignPhases`) are `internal` statics, so `@testable import` reaches them
/// the way the Python tests reach the module functions.
public enum PhaseSegmenter {
    /// `NO_HOLD` — `hold_id` outside a hold.
    public static let noHold = -1

    // MARK: - One clip

    /// Label one clip, from its processed trajectory to its phases —
    /// `classify_clip`.
    ///
    /// The input is exactly what Python passes: the frame clock, the
    /// post-processed `x`/`y`/`valid` of `ProcessedClip`, the **raw**
    /// `trainer_contact` (the post-process gates it away, a phase still has to
    /// say why the frame is unknown) and `bodyLength.totalPx` as the scale.
    ///
    /// A clip with no body length comes back with every frame `unknown` and
    /// `usable` set to `false`: without `L` there is no unit to measure
    /// inversion, angle or speed in, and a phase read in pixels is not a
    /// phase.
    public static func classify(
        tMs: [Int],
        processed: ProcessedClip,
        trainerContact: [Bool],
        config: PhaseConfig = .init()
    ) -> ClipPhases {
        let frames = tMs.count
        let bodyLength = processed.bodyLength.totalPx ?? .nan
        var reason = ""
        if !bodyLength.isFinite || bodyLength <= 0.0 {
            // Python's `f"no body length ({body_length!r} px)"`: Swift's
            // description of a `Double` is Python's `repr` for every value
            // this path can produce (`nan`, `inf`, `0.0`, …), which
            // `PhasesTests.testAnUnusableBodyLengthIsAllUnknown` pins down.
            reason = "no body length (\(bodyLength) px)"
        }
        if !reason.isEmpty {
            return ClipPhases(
                tMs: tMs,
                phase: [Phase](repeating: .unknown, count: frames),
                holdId: [Int](repeating: config.noHold, count: frames),
                signals: unusableSignals(frames: frames, reason: reason),
                runs: [],
                usable: false,
                unusableReason: reason
            )
        }

        // The processed trajectory as Python's arrays: one column per joint of
        // `Joint.allCases` (Python passes the clip's joint names, which for
        // every input this port sees is exactly this order), `NaN` wherever a
        // joint has no position.
        let order = Joint.allCases
        let rows = processed.frames.count
        var x = [[Double]](repeating: [Double](repeating: .nan, count: order.count), count: rows)
        var y = [[Double]](repeating: [Double](repeating: .nan, count: order.count), count: rows)
        var valid = [[Bool]](repeating: [Bool](repeating: false, count: order.count), count: rows)
        for (index, frame) in processed.frames.enumerated() {
            for (column, joint) in order.enumerated() {
                x[index][column] = frame.joints[joint]?.x ?? .nan
                y[index][column] = frame.joints[joint]?.y ?? .nan
                valid[index][column] = frame.valid.contains(joint)
            }
        }

        let signals = frameSignals(
            tMs: tMs, x: x, y: y, valid: valid, trainerContact: trainerContact,
            joints: order, bodyLength: bodyLength, config: config
        )
        let runs = holdRuns(
            tSeconds: tMs.map { Double($0) / 1000.0 },
            holding: holdCondition(signals, config: config),
            known: signals.known,
            event: signals.handStep,
            minHoldS: config.minHoldS,
            maxBreakS: config.holdBreakMaxS
        )
        let (phase, holdId) = assignPhases(signals: signals, runs: runs, config: config)
        return ClipPhases(
            tMs: tMs,
            phase: phase,
            holdId: holdId,
            signals: signals,
            runs: runs,
            usable: true,
            unusableReason: ""
        )
    }

    /// The signals of a clip with no scale: nothing measured, everything
    /// unknown — `_unusable_signals`.
    static func unusableSignals(frames: Int, reason: String) -> FrameSignals {
        FrameSignals(
            known: [Bool](repeating: false, count: frames),
            unknownReason: [String](repeating: reason, count: frames),
            vAnkleMid: [Double](repeating: .nan, count: frames),
            uAnkleMid: [Double](repeating: .nan, count: frames),
            vHipMid: [Double](repeating: .nan, count: frames),
            bodyAngleDeg: [Double](repeating: .nan, count: frames),
            inverted: [Bool](repeating: false, count: frames),
            handsLow: [Bool](repeating: false, count: frames),
            wristSpeedLPerS: [Double](repeating: .nan, count: frames),
            wristStepL: [Double](repeating: .nan, count: frames),
            handsDown: [Bool](repeating: false, count: frames),
            handStep: [Bool](repeating: false, count: frames),
            legsVelocityLPerS: [Double](repeating: .nan, count: frames),
            legsRising: [Bool](repeating: false, count: frames),
            legsFalling: [Bool](repeating: false, count: frames)
        )
    }

    // MARK: - Measurements

    /// The columns of `joints` this clip has, in `wanted` order —
    /// `_joint_columns`.
    ///
    /// A source that does not report one of them (Apple Vision has no
    /// `foot_index`) simply has no column for it, and every measurement over
    /// these names skips it rather than padding it with zeros.
    static func jointColumns(_ joints: [Joint], wanted: [Joint]) -> [Int] {
        wanted.compactMap { joints.firstIndex(of: $0) }
    }

    /// A joint group's per-frame midpoint over its visible members, and the
    /// count — `_midpoint`.
    ///
    /// One wrist is enough to place the origin and one ankle is enough to say
    /// which way up the body is, so a side-on clip with one wrist in shot is
    /// still classifiable; a frame where no member of the group is visible is
    /// left `NaN` rather than averaged over zeros (Python divides by
    /// `max(count, 1)` and then replaces the empty rows with `NaN`, so the
    /// empty sum never leaks out).
    static func midpoint(
        x: [[Double]],
        y: [[Double]],
        valid: [[Bool]],
        columns: [Int]
    ) -> (x: [Double], y: [Double], count: [Int]) {
        let frames = x.count
        guard !columns.isEmpty else {
            // Python returns zeros for both positions and a zero count: the
            // count is what makes `known` false on every frame, and the zero
            // origin still feeds the ungated signals exactly as Python's does.
            return (
                [Double](repeating: 0.0, count: frames),
                [Double](repeating: 0.0, count: frames),
                [Int](repeating: 0, count: frames)
            )
        }
        var midX = [Double](repeating: .nan, count: frames)
        var midY = [Double](repeating: .nan, count: frames)
        var count = [Int](repeating: 0, count: frames)
        for index in 0..<frames {
            // `np.where(known, x[:, columns], 0.0).sum(axis=1)`: invalid
            // members contribute zero, so summing in column order over the
            // visible ones is the same number.
            var sumX = 0.0
            var sumY = 0.0
            var visible = 0
            for column in columns where valid[index][column] {
                sumX += x[index][column]
                sumY += y[index][column]
                visible += 1
            }
            count[index] = visible
            if visible > 0 {
                midX[index] = sumX / Double(visible)
                midY[index] = sumY / Double(visible)
            }
        }
        return (midX, midY, count)
    }

    /// The largest finite value of each row; `NaN` where a row is all `NaN` —
    /// `_largest`.
    ///
    /// `np.nanmax` warns on an all-NaN slice, and a wrist that cannot be seen
    /// at both ends of a window is exactly that: the rows without a finite
    /// value are taken out before the maximum is taken.
    static func largest(_ values: [[Double]]) -> [Double] {
        guard let firstRow = values.first, !firstRow.isEmpty else {
            // No columns at all: every row is unmeasured (Python's
            // `_largest` returns `np.full(values.shape[0], nan)`).
            return [Double](repeating: .nan, count: values.count)
        }
        return values.map { row in
            var best = -Double.infinity
            var finite = false
            for value in row where value.isFinite {
                finite = true
                best = Swift.max(best, value)
            }
            return finite ? best : .nan
        }
    }

    /// `handstand.bodyframe.to_body_frame` over a whole clip, one frame at a
    /// time — `_body_frame_uv`.
    ///
    /// The origin is a per-frame quantity — the wrist midpoint of *that*
    /// frame — so this is the call `BodyFrame.toBodyFrame` is written for, and
    /// a `NaN` midpoint travels through it as a `NaN` position, which is how
    /// an unmeasurable frame stays unmeasurable. The scale is checked by the
    /// caller (`frameSignals`), exactly as Python checks it first.
    static func bodyFrameUV(
        x: [Double],
        y: [Double],
        originX: [Double],
        originY: [Double],
        bodyLength: Double
    ) -> (u: [Double], v: [Double]) {
        var u = [Double]()
        var v = [Double]()
        u.reserveCapacity(x.count)
        v.reserveCapacity(x.count)
        for index in x.indices {
            do {
                let point = try BodyFrame.toBodyFrame(
                    x: x[index], y: y[index],
                    wristMidX: originX[index], wristMidY: originY[index],
                    bodyLength: bodyLength
                )
                u.append(point.u)
                v.append(point.v)
            } catch {
                // Unreachable: `frameSignals` refuses a scale it cannot
                // measure in first. Python's `ValueError` text is kept.
                preconditionFailure(
                    "body_length must be a positive, finite number, got \(bodyLength)"
                )
            }
        }
        return (u, v)
    }

    /// How far, and how fast, a point moved over the `windowS` before each
    /// frame — `window_motion`.
    ///
    /// The window is the frames from the last one at or before `t - windowS`
    /// up to the frame itself, which is what "over the last fifth of a second"
    /// means on a variable frame rate clip — `np.searchsorted(...,
    /// side="right") - 1`, found by timestamp and never by frame count. The
    /// number is `NaN` whenever the question cannot be asked of the clip:
    /// when the window reaches back before its first frame, when no time has
    /// passed inside it, or when either end is a frame that cannot see the
    /// point. A half-measurement is never dressed up as a measurement.
    static func windowMotion(
        tSeconds: [Double],
        x: [Double],
        y: [Double],
        known: [Bool],
        bodyLength: Double,
        windowS: Double
    ) -> (stepL: [Double], speedLPerS: [Double]) {
        precondition(windowS > 0.0, "window_s must be > 0, got \(windowS)")
        let frames = tSeconds.count
        var stepL = [Double](repeating: .nan, count: frames)
        var speedLPerS = [Double](repeating: .nan, count: frames)
        for index in 0..<frames {
            let earlier = searchSortedRight(tSeconds, tSeconds[index] - windowS) - 1
            guard earlier >= 0, known[index], known[earlier] else { continue }
            let elapsed = tSeconds[index] - tSeconds[earlier]
            guard elapsed > 0.0 else { continue }
            stepL[index] = hypot(x[index] - x[earlier], y[index] - y[earlier]) / bodyLength
            speedLPerS[index] = stepL[index] / elapsed
        }
        return (stepL, speedLPerS)
    }

    /// The rate of change of `values`, over `spanS` centred on each frame —
    /// `velocity_l_per_s`.
    ///
    /// Centred, because a forward difference of a smoothed track reads the
    /// motion as happening *after* the frame it is attached to, and a phase
    /// boundary is where the motion is. The window is clipped to the frames
    /// the clip has — the first and last frames are measured against the one
    /// side they have — and is `NaN` whenever the frame itself or the one it
    /// is compared with is a frame the module cannot see into, so the edges of
    /// a gap are not velocities. `timeEps` is Python's `_TIME_EPS`, the slack
    /// that stops a window landing exactly on a frame from being decided by
    /// the last bit of a timestamp.
    static func velocityLPerS(
        tSeconds: [Double],
        values: [Double],
        known: [Bool],
        spanS: Double,
        timeEps: Double = PhaseConfig().timeEps
    ) -> [Double] {
        precondition(spanS > 0.0, "span_s must be > 0, got \(spanS)")
        let frames = tSeconds.count
        let half = spanS / 2.0
        var out = [Double](repeating: .nan, count: frames)
        for index in 0..<frames {
            // The nearest frame on each side of the window, clipped to the
            // clip: the first and last frames are measured against the one
            // side they have.
            var before = searchSortedRight(tSeconds, tSeconds[index] - half + timeEps) - 1
            var after = searchSortedLeft(tSeconds, tSeconds[index] + half - timeEps)
            before = Swift.max(Swift.min(before, index), 0)
            after = Swift.min(Swift.max(after, index), frames - 1)
            let elapsed = tSeconds[after] - tSeconds[before]
            guard elapsed > 0.0, known[index], known[before], known[after] else { continue }
            out[index] = (values[after] - values[before]) / elapsed
        }
        return out
    }

    /// Measure every signal of `FrameSignals` for one clip — `frame_signals`.
    ///
    /// `tMs` is the clip's frame timestamps in milliseconds — the only clock
    /// these clips have, and variable from frame to frame. `x`, `y` and
    /// `valid` are the *(frames, joints)* arrays of the **processed** track in
    /// display pixels, `trainerContact` says per frame whether the athlete
    /// selection flagged a trainer on the athlete (such a frame is two
    /// bodies' keypoints stitched together, and is `unknown` whatever the
    /// numbers say), `joints` is the clip's joint names in column order, and
    /// `bodyLength` is the clip's scale `L` in pixels — positive and finite,
    /// because a clip without one has no unit to measure in and is refused
    /// here rather than divided by later.
    static func frameSignals(
        tMs: [Int],
        x: [[Double]],
        y: [[Double]],
        valid: [[Bool]],
        trainerContact: [Bool],
        joints: [Joint],
        bodyLength: Double,
        config: PhaseConfig = .init()
    ) -> FrameSignals {
        precondition(
            bodyLength.isFinite && bodyLength > 0.0,
            "body_length must be a positive, finite number, got \(bodyLength)"
        )
        let times = tMs.map { Double($0) / 1000.0 }
        precondition(
            sameShape(x, y) && sameShape(x, valid),
            "x, y and valid must have the same (frames, joints) shape"
        )
        precondition(
            x.count == times.count,
            "x has \(x.count) frames, t_ms has \(times.count)"
        )
        precondition(
            trainerContact.count == times.count,
            "trainer_contact must have one entry per frame (\(times.count)), "
                + "got \(trainerContact.count)"
        )
        if let columns = x.first {
            // A clip with no frames has no column to check; Python's ndarray
            // would still have one, so the check runs whenever there is a row.
            precondition(
                columns.count == joints.count,
                "x has \(columns.count) joint columns but \(joints.count) joint names"
            )
        }

        let frames = times.count
        let wristColumns = jointColumns(joints, wanted: config.wristJoints)
        let ankleColumns = jointColumns(joints, wanted: config.ankleJoints)
        let hipColumns = jointColumns(joints, wanted: config.hipJoints)
        let wrist = midpoint(x: x, y: y, valid: valid, columns: wristColumns)
        let ankle = midpoint(x: x, y: y, valid: valid, columns: ankleColumns)
        let hip = midpoint(x: x, y: y, valid: valid, columns: hipColumns)

        var known = [Bool](repeating: false, count: frames)
        var unknownReason = [String](repeating: "", count: frames)
        for index in 0..<frames {
            known[index] =
                wrist.count[index] > 0 && ankle.count[index] > 0 && hip.count[index] > 0
                && !trainerContact[index]
            if trainerContact[index] {
                unknownReason[index] = config.reasonTrainer
            } else if wrist.count[index] == 0 {
                unknownReason[index] = config.reasonWrists
            } else if ankle.count[index] == 0 {
                unknownReason[index] = config.reasonAnkles
            } else if hip.count[index] == 0 {
                unknownReason[index] = config.reasonHips
            }
        }

        let (uAnkle, vAnkle) = bodyFrameUV(
            x: ankle.x, y: ankle.y, originX: wrist.x, originY: wrist.y, bodyLength: bodyLength
        )
        let (_, vHip) = bodyFrameUV(
            x: hip.x, y: hip.y, originX: wrist.x, originY: wrist.y, bodyLength: bodyLength
        )

        // atan2 of the horizontal against the vertical component: the angle of
        // the wrist->ankle vector from straight up, signed by which way it
        // leans. Measured on every frame, as Python does — an unmeasurable
        // frame is `atan2(NaN, NaN)`, which is `NaN`, and every comparison
        // against it is false.
        var bodyAngleDeg = [Double](repeating: .nan, count: frames)
        for index in 0..<frames {
            bodyAngleDeg[index] = atan2(uAnkle[index], vAnkle[index]) * (180.0 / Double.pi)
        }
        var inverted = [Bool](repeating: false, count: frames)
        var handsLow = [Bool](repeating: false, count: frames)
        for index in 0..<frames {
            inverted[index] = known[index] && vAnkle[index] > config.invertedMin
            handsLow[index] = known[index] && vHip[index] > config.handsLowMinV
        }

        // One window measurement per visible wrist, then the largest across
        // them: one wrist is enough when the source only reports one.
        var stepColumns = [[Double]]()
        var speedColumns = [[Double]]()
        for column in wristColumns {
            let motion = windowMotion(
                tSeconds: times,
                x: x.map { $0[column] },
                y: y.map { $0[column] },
                known: valid.map { $0[column] },
                bodyLength: bodyLength,
                windowS: config.handStillWindowS
            )
            stepColumns.append(motion.stepL)
            speedColumns.append(motion.speedLPerS)
        }
        // `(frames, wrists)` from the per-column results; with no wrist column
        // at all every row is empty and `largest` leaves every frame
        // unmeasured.
        let steps = (0..<frames).map { row in stepColumns.map { $0[row] } }
        let speeds = (0..<frames).map { row in speedColumns.map { $0[row] } }
        let wristStepL = largest(steps)
        let wristSpeedLPerS = largest(speeds)

        let legsVelocityLPerS = velocityLPerS(
            tSeconds: times, values: vAnkle, known: known,
            spanS: config.legVelocityWindowS, timeEps: config.timeEps
        )

        var handsDown = [Bool](repeating: false, count: frames)
        var handStep = [Bool](repeating: false, count: frames)
        var legsRising = [Bool](repeating: false, count: frames)
        var legsFalling = [Bool](repeating: false, count: frames)
        for index in 0..<frames {
            // A wrist whose speed could not be measured is not evidence that
            // the hands moved. That happens on the first fifth of a second
            // after a stretch of frames nobody could see, where there is no
            // earlier frame to compare with; reading it as "the hands are
            // moving" would end a hold every time a trainer walked in front of
            // the camera, which is the one thing a gap must not do.
            let still = wristSpeedLPerS[index].isFinite
                ? wristSpeedLPerS[index] < config.handStillLPerS
                : true
            handsDown[index] = handsLow[index] && still
            handStep[index] = known[index] && wristStepL[index] > config.handStepL
            legsRising[index] =
                known[index] && legsVelocityLPerS[index] > config.legVelocityMinLPerS
            legsFalling[index] =
                known[index] && legsVelocityLPerS[index] < -config.legVelocityMinLPerS
        }

        return FrameSignals(
            known: known,
            unknownReason: unknownReason,
            vAnkleMid: vAnkle,
            uAnkleMid: uAnkle,
            vHipMid: vHip,
            bodyAngleDeg: bodyAngleDeg,
            inverted: inverted,
            handsLow: handsLow,
            wristSpeedLPerS: wristSpeedLPerS,
            wristStepL: wristStepL,
            handsDown: handsDown,
            handStep: handStep,
            legsVelocityLPerS: legsVelocityLPerS,
            legsRising: legsRising,
            legsFalling: legsFalling
        )
    }

    // MARK: - Holds

    /// The tests a hold frame has to pass, all of them at once —
    /// `hold_condition`.
    ///
    /// Inverted and straight (`upsideDown`), the hands planted (`handsDown`)
    /// and no hand step (`handStep`) — the last of which is necessarily a
    /// failure of the third, since `handStepL` over `handStillWindowS` is
    /// faster than `handStillLPerS`. It stays its own term because the two
    /// answer different questions and #71 re-tunes the step.
    static func holdCondition(_ signals: FrameSignals, config: PhaseConfig = .init()) -> [Bool] {
        let upsideDown = signals.upsideDown(config: config)
        var out = [Bool](repeating: false, count: signals.frames)
        for index in 0..<signals.frames {
            out[index] = upsideDown[index] && signals.handsDown[index] && !signals.handStep[index]
        }
        return out
    }

    /// The stretches of frames a hold occupies, the long ones only, numbered
    /// from 0 — `hold_runs`.
    ///
    /// A run starts at the first frame that passes `holdCondition` and ends at
    /// the last one before a break that ends it. Three kinds of frame in
    /// between do not end a run:
    ///
    /// * an `event` frame — a hand step — ends the run at once, wherever it
    ///   is;
    /// * an unknown frame, which is a trainer in the way rather than a change
    ///   in what the athlete is doing;
    /// * a known frame the condition failed on, for as long as the whole break
    ///   lasts at most `maxBreakS`.
    ///
    /// The last one is the geometry wobbling: a held handstand sways, and its
    /// ankle midpoint crosses `invertedMin` now and then for a frame. A
    /// stretch of failures longer than `maxBreakS` is the athlete coming down,
    /// and it ends the hold. The frames a run bridges are inside it, so they
    /// carry the hold's phase and number — a hold with a hole in it is a thing
    /// every later stage would have to special-case.
    ///
    /// Runs at least `minHoldS` long come back numbered in time order; the
    /// shorter ones are not holds and are left to the state machine.
    static func holdRuns(
        tSeconds: [Double],
        holding: [Bool],
        known: [Bool],
        event: [Bool]? = nil,
        minHoldS: Double = PhaseConfig().minHoldS,
        maxBreakS: Double = PhaseConfig().holdBreakMaxS
    ) -> [HoldRun] {
        precondition(minHoldS >= 0.0, "min_hold_s must be >= 0, got \(minHoldS)")
        precondition(maxBreakS >= 0.0, "max_break_s must be >= 0, got \(maxBreakS)")
        precondition(
            holding.count == known.count && holding.count == tSeconds.count,
            "holding, known and t_seconds must have the same shape"
        )
        if let event {
            precondition(
                event.count == holding.count,
                "event must have the same shape as holding"
            )
        }

        var runs = [(start: Int, end: Int)]()
        var start: Int?
        var breakStart: Int?
        for index in tSeconds.indices {
            if holding[index] {
                if start == nil { start = index }
                breakStart = nil
                continue
            }
            guard let runStart = start else { continue }
            if event?[index] == true {
                runs.append((start: runStart, end: index - 1))
                start = nil
                breakStart = nil
                continue
            }
            if breakStart == nil {
                breakStart = index
            } else if tSeconds[index] - tSeconds[breakStart!] > maxBreakS {
                runs.append((start: runStart, end: breakStart! - 1))
                start = nil
                breakStart = nil
            }
        }
        if let runStart = start {
            let end = breakStart.map { $0 - 1 } ?? (tSeconds.count - 1)
            runs.append((start: runStart, end: end))
        }

        var out = [HoldRun]()
        for run in runs where tSeconds[run.end] - tSeconds[run.start] >= minHoldS {
            out.append(HoldRun(start: run.start, end: run.end, holdId: out.count))
        }
        return out
    }

    // MARK: - The state machine

    /// Label every frame with its phase and, inside a hold, its hold number —
    /// `assign_phases`.
    ///
    /// A hold's own frames are labelled first, so the walk below never has to
    /// know whether a run qualified. The rest is four rules, in this order,
    /// and they are the whole of it:
    ///
    /// 1. a frame the module cannot see into is `unknown` and does not move
    ///    the state on — a gap is not an event;
    /// 2. a hold frame puts the state in `hold`, and the first known frame
    ///    that is not holding takes it to `exit`: a hold that stops holding is
    ///    an exit, whatever the athlete does next;
    /// 3. hands down with the legs rising is `kickup`, whenever it happens: a
    ///    new attempt, whether it is the first, the third, or the one after a
    ///    fall;
    /// 4. the body no longer above the hands ends the attempt: `pre` stays
    ///    `pre` and every other state becomes `post`.
    ///
    /// Anything else stands, so `kickup` and `exit` last until one of those
    /// four says otherwise. Note that rule 4 asks about `handsLow` and not
    /// about `handsDown`: a hand step is the hands *moving* while they stay on
    /// the floor, which ends the hold (rule 2) and is an exit, not the end of
    /// the attempt.
    static func assignPhases(
        signals: FrameSignals,
        runs: [HoldRun],
        config: PhaseConfig = .init()
    ) -> (phase: [Phase], holdId: [Int]) {
        let frames = signals.frames
        var phase = [Phase](repeating: .unknown, count: frames)
        var holdId = [Int](repeating: config.noHold, count: frames)
        var holding = [Bool](repeating: false, count: frames)
        for run in runs {
            for index in run.start...run.end {
                holding[index] = true
                holdId[index] = run.holdId
                phase[index] = .hold
            }
        }

        var state = Phase.pre
        for index in 0..<frames {
            if !signals.known[index] { continue }
            if holding[index] {
                state = .hold
                continue
            }
            if state == .hold {
                state = .exit
            } else if signals.handsDown[index] && signals.legsRising[index] {
                state = .kickup
            } else if !signals.handsLow[index] {
                state = state == .pre ? .pre : .post
            }
            phase[index] = state
        }
        return (phase, holdId)
    }

    // MARK: - Timestamps

    /// `np.searchsorted(sorted, value, side: "left")` — the first index whose
    /// entry is `>= value`. Times are non-decreasing and never `NaN`, as
    /// Python's `t_ms` derived seconds are.
    private static func searchSortedLeft(_ sorted: [Double], _ value: Double) -> Int {
        var low = 0
        var high = sorted.count
        while low < high {
            let middle = (low + high) / 2
            if sorted[middle] < value {
                low = middle + 1
            } else {
                high = middle
            }
        }
        return low
    }

    /// `np.searchsorted(sorted, value, side: "right")` — the first index whose
    /// entry is strictly `> value`.
    private static func searchSortedRight(_ sorted: [Double], _ value: Double) -> Int {
        var low = 0
        var high = sorted.count
        while low < high {
            let middle = (low + high) / 2
            if sorted[middle] <= value {
                low = middle + 1
            } else {
                high = middle
            }
        }
        return low
    }

    // MARK: - Shared shape checks (Python's ValueError messages, kept)

    private static func sameShape(_ a: [[Double]], _ b: [[Double]]) -> Bool {
        guard a.count == b.count else { return false }
        guard let columns = a.first else { return true }
        return a.indices.allSatisfy { a[$0].count == columns.count && b[$0].count == columns.count }
    }

    private static func sameShape(_ a: [[Double]], _ b: [[Bool]]) -> Bool {
        guard a.count == b.count else { return false }
        guard let columns = a.first else { return true }
        return a.indices.allSatisfy { a[$0].count == columns.count && b[$0].count == columns.count }
    }
}
