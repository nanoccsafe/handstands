import Foundation

// --------------------------------------------------------------------------- #
// What the phone says out loud while the camera is up (chainlink #91, live mode
// stage 1) — the pure half of it: which cue, if any, this tick earns.
//
// The scheduler is a *policymaker*, not a speaker: `CaptureService` feeds it one
// tick at a time (the live detector's events, the framing verdict, whether the
// athlete is upside down, whether the red light is on) and it answers with at
// most one ``Cue``, which the app then says and shows as a banner. Every rule
// the issue asks for lives here as a constant at the top of the file, so the
// pacing can be tuned — and tested — without a phone, a gym or a voice.
//
// The philosophy (user, 2026-10-06) is *sparse and encouraging*: one cue when a
// line is achieved, optional time marks, framing hints before the attempt, and
// **no corrections during a hold** — the critique belongs to the analysis that
// runs after the recording, which stays authoritative. The user changed that
// last rule at the gym (2026-10-14, chainlink #97): a *sustained red fault*
// mid-hold now earns a short form cue. It now has two shapes — the #97 banana
// faults and the #100 piked hip — and repeats at *coach pace* (every couple of
// seconds while the fault stays red) rather than firing once. It still has its
// own toggle, on by default because the user asked for it.
// --------------------------------------------------------------------------- #

/// The toggles behind the gear on the home screen (chainlink #91, extended by
/// #97), the defaults a fresh install shows.
public struct CueSettings: Equatable, Sendable {
    /// Speak the line cue, the time marks, "Ready." and the form cues — the
    /// master switch for everything the *attempt* earns. Framing hints have
    /// their own toggle.
    public var voiceCues = true

    /// Speak the optional "Five." / "Ten." marks while a line is held.
    public var timeMarks = false

    /// Speak (and show) the framing hints *before* the attempt.
    public var framingHints = true

    /// Speak one short phrase when the line goes red during a hold (chainlink
    /// #97) — the user asked for it at the gym (2026-10-14), so a fresh
    /// install has it **on**.
    public var formCues = true

    public init() {}

    public init(voiceCues: Bool, timeMarks: Bool, framingHints: Bool, formCues: Bool = true) {
        self.voiceCues = voiceCues
        self.timeMarks = timeMarks
        self.framingHints = framingHints
        self.formCues = formCues
    }
}

/// Which way the line broke during a hold (chainlink #97), with the same
/// spellings the `banana` coaching cues use: positive banana is the belly
/// side, negative the back.
public enum FormFault: String, Sendable, CaseIterable {
    /// The line curved towards the belly — "Ribs in.".
    case arch
    /// The line curved away from the belly — "Open the hips.".
    case hollow
    /// The line is red but the nose cannot say which way it curves —
    /// "Straighten.".
    case bent
    /// The hips are piked — the shoulder–hip–ankle angle is red — "Hips over
    /// shoulders." (chainlink #100).
    case hips
}

/// What a framing hint tells the user to do about the picture.
public enum FramingHint: String, Sendable {
    /// The body is cut off at the top or bottom (or a required joint nobody
    /// can see): the phone is too close.
    case stepBack
    /// One body, but too small in the frame to analyse.
    case comeCloser
    /// The body is cut off at the left or right edge: move sideways.
    case moveToMiddle
    /// Two bodies in the picture; only one may be recorded.
    case onlyYou
    /// Cut off at the top *while standing* — the phone sits too low.
    case raisePhone
}

/// One thing worth saying (and showing as a banner) — chainlink #91, #97.
public enum Cue: Equatable, Sendable {
    /// "Up." — the athlete is up (palms down, legs above the hips) and it has
    /// held for the onset (chainlink #100).
    case up
    /// "Line. Hold it." — a line has been achieved, once per line.
    case lineHold
    /// "Five." / "Ten." — `seconds` of line time have passed.
    case timeMark(Int)
    /// A framing hint, before the attempt.
    case framing(FramingHint)
    /// "Ready." — the framing went from wrong to right while upright.
    case ready
    /// "Ribs in." / "Open the hips." / "Straighten." / "Hips over shoulders."
    /// — the line broke during a hold (chainlink #97, #100).
    case form(FormFault)
}

/// Which cue to speak now — the whole of live mode's policy (chainlink #91).
///
/// One `decide` call per tick: the live frame's events in, at most one cue out.
/// The rules, as constants below:
///
/// * `up` once per up stretch (on ``LiveEvent/up``), superseded by a pending
///   `lineHold`;
/// * `lineHold` once per line (on ``LiveEvent/lineAchieved``);
/// * `timeMark` only when ``CueSettings/timeMarks`` is on;
/// * framing hints only while upright and not in line, never within
///   ``afterInversionGapS`` of the athlete coming down;
/// * at most one framing hint every ``hintGapS``, never the same hint twice in
///   a row within ``hintRepeatGapS``;
/// * `ready` once per non-ok → ok transition while upright;
/// * form cues at coach pace — every ``formGapS`` while a fault stays red,
///   but only while the athlete is *up* and outside an achieved line;
/// * nothing at all when ``CueSettings/voiceCues`` is off, except framing hints,
///   which have their own toggle;
/// * globally at least ``debounceGapS`` between any two cues — except
///   `lineHold` and `up`, which may follow anything after ``lineHoldGapS``, and
///   form cues, which follow anything after ``formGapS``.
public struct CueScheduler: Sendable {
    /// The toggles in force. Assignable, so the gear's changes apply to the
    /// scheduler already running (state such as the last hint is kept).
    public var settings: CueSettings

    // MARK: - The pacing rules

    /// The global debounce: at least this many seconds between any two cues.
    public static let debounceGapS = 3.0

    /// What `lineHold` and `up` may follow anything by — the cues that matter
    /// most must never be delayed a whole debounce by a hint the user barely
    /// noticed.
    public static let lineHoldGapS = 1.0

    /// What form cues may follow anything by — coach-mode pacing (chainlink
    /// #100): roughly a phrase every two seconds while a fault stays red.
    public static let formGapS = 2.0

    /// No framing hint within this many seconds of the athlete coming down out
    /// of inversion (the picture is still "upside-down business" for a moment).
    public static let afterInversionGapS = 1.5

    /// At most one framing hint every this many seconds.
    public static let hintGapS = 4.0

    /// The same hint is never repeated within this many seconds.
    public static let hintRepeatGapS = 8.0

    // MARK: - State

    /// Is the athlete in an announced line? Set on `lineAchieved`, cleared on
    /// `lineLost` — the detector's own hysteresis, mirrored here so hints stay
    /// quiet through the whole line.
    private var inLine = false

    /// A `lineAchieved` waiting for its turn to be spoken (it is never dropped:
    /// the line cue is the point of the whole feature).
    private var pendingLineHold = false

    /// An `.up` waiting for its turn (chainlink #100). Queued like the line cue
    /// — never dropped — but discarded the moment a line cue is pending, since
    /// "Line. Hold it." supersedes the acknowledgement.
    private var pendingUp = false

    /// A `timeMark` waiting for its turn; the newest mark replaces an older one.
    private var pendingTimeMark: Int?

    /// A form cue (chainlink #97) waiting for its turn; the newest fault
    /// replaces an older one. It is queued like the line cue and the marks —
    /// the fault is a real correction, not a stale edge — but it is only
    /// spoken when framing hints have nothing to say (they outrank it), only
    /// with ``CueSettings/formCues`` on, only while the athlete is *up*
    /// (``decide``'s `up`) and not inside an achieved line (chainlink #100
    /// review).
    private var pendingForm: FormFault?

    /// When the last cue was spoken, and which — the global debounce reads it.
    private var lastSpokenTMs: Int?

    /// The last cue spoken (`nil` before the first) — what the debounce spent
    /// itself on; tests and the tuning stats read it.
    public private(set) var lastCue: Cue?

    /// When the last framing hint was spoken, and which — the hint spacing rules.
    private var lastHintTMs: Int?
    private var lastHint: FramingHint?

    /// The athlete *was* upside down last tick?
    private var wasInverted = false

    /// When inversion ended (the tick `inverted` went false), `nil` while
    /// upright or while inverted continuously — the 1.5 s cooldown reads it.
    private var inversionEndedTMs: Int?

    /// The framing was non-ok on the previous tick with a verdict in it; the
    /// `ready` edge reads it (`nil` until the first verdict).
    private var framingWasNonOk: Bool?

    public init(settings: CueSettings = .init()) {
        self.settings = settings
    }

    // MARK: - One tick

    /// Decide the cue for this tick — one frame's events in, at most one cue
    /// out. Pure with respect to its inputs' *order*: call it once per live
    /// frame, in time order, with the framing verdict in force at that moment.
    ///
    /// `up` is ``LiveLineDetector/isUpNow`` for this frame: form cues are only
    /// for an athlete who has already been acknowledged ("Up."), so while it
    /// is false incoming form events are ignored and a waiting one is dropped
    /// — a correction must never arrive before the acknowledgement of the
    /// same up stretch (chainlink #100 review). A `lineAchieved` clears any
    /// waiting form cue too: "Line. Hold it." supersedes a stale correction,
    /// and none is queued while the scheduler is in a line.
    ///
    /// A cue that is *waiting* (a pending line hold or time mark, blocked only
    /// by the debounce) is held for a later tick rather than dropped; the
    /// edge-triggered cues (`ready`, framing hints) are dropped when the
    /// debounce is not yet spent, because saying them late is worse than not
    /// saying them.
    public mutating func decide(
        tMs: Int,
        events: [LiveEvent],
        formEvents: [LiveFormEvent] = [],
        framing: FramingStatus?,
        inverted: Bool,
        up: Bool = true,
        recording: Bool
    ) -> Cue? {
        // A toggle switched off while something was pending must not leave a
        // cue behind to fire after the user asked for silence. `up` is the
        // same: a form cue waiting while the athlete comes back down is stale.
        if !settings.voiceCues {
            pendingLineHold = false
            pendingUp = false
            pendingTimeMark = nil
        }
        if !settings.voiceCues || !settings.formCues || !up {
            pendingForm = nil
        }

        // The inversion edge: cooldown starts the tick the athlete comes down.
        if inverted {
            inversionEndedTMs = nil
        } else if wasInverted {
            inversionEndedTMs = tMs
        }
        wasInverted = inverted

        // What this frame's detector said.
        for event in events {
            switch event {
            case .up:
                if settings.voiceCues { pendingUp = true }
            case .lineAchieved:
                inLine = true
                if settings.voiceCues {
                    pendingLineHold = true
                    // The line cue supersedes the acknowledgement: if a line
                    // is pending, "Up." is not worth saying any more.
                    pendingUp = false
                    // …and it supersedes a correction too: a form cue still
                    // waiting is about a shape the athlete is no longer in
                    // (chainlink #100 review).
                    pendingForm = nil
                }
            case .lineLost:
                inLine = false
            case .timeMark(_, let seconds):
                if settings.voiceCues, settings.timeMarks { pendingTimeMark = seconds }
            }
        }
        // Form events are ignored unless the athlete is *up* and outside a
        // line: a correction before "Up." is out of order, and one inside an
        // achieved line is stale the moment "Line. Hold it." is due.
        for event in formEvents where settings.voiceCues && settings.formCues && up && !inLine {
            pendingForm = event.fault
        }

        // 1. The line cue — highest priority, shortest debounce.
        if pendingLineHold {
            guard allowed(tMs: tMs, gapS: Self.lineHoldGapS) else { return nil }
            pendingLineHold = false
            pendingUp = false
            return spoken(.lineHold, tMs: tMs)
        }

        // 1b. The "Up." acknowledgement — right after the line cue, on the
        //     same short gap, queued (never dropped) but superseded above.
        if pendingUp {
            guard allowed(tMs: tMs, gapS: Self.lineHoldGapS) else { return nil }
            pendingUp = false
            return spoken(.up, tMs: tMs)
        }

        // 2. A waiting time mark.
        if let pendingTimeMark {
            if settings.timeMarks, allowed(tMs: tMs, gapS: Self.debounceGapS) {
                self.pendingTimeMark = nil
                return spoken(.timeMark(pendingTimeMark), tMs: tMs)
            }
            if !settings.timeMarks { self.pendingTimeMark = nil }
            return nil
        }

        // 3. "Ready." — the edge from wrong framing to right, upright.
        if let ready = readyCue(tMs: tMs, framing: framing, inverted: inverted, recording: recording)
        {
            return ready
        }

        // 4. A framing hint — pre-attempt only, throttled three ways. It
        //    outranks the form cue below, so a hint is never held back for a
        //    correction (in practice a form cue only fires upside down, where
        //    the hints are already silent).
        if let hint = hintCue(tMs: tMs, framing: framing, inverted: inverted, recording: recording)
        {
            return hint
        }

        // 5. A waiting form cue — the last of the cues, and only after the
        //    hints had nothing to say.
        return formCue(tMs: tMs)
    }

    // MARK: - The edge-triggered cues

    /// "Ready." when the framing goes non-ok → ok while upright and not
    /// recording — once per transition, whatever the debounce says (an edge
    /// nobody caught is not repeated later).
    private mutating func readyCue(
        tMs: Int,
        framing: FramingStatus?,
        inverted: Bool,
        recording: Bool
    ) -> Cue? {
        guard let framing else { return nil }
        let isOk = framing == .ok
        let wasNonOk = framingWasNonOk == true
        framingWasNonOk = !isOk
        guard settings.voiceCues, !recording, !inverted, isOk, wasNonOk,
            allowed(tMs: tMs, gapS: Self.debounceGapS)
        else { return nil }
        return spoken(.ready, tMs: tMs)
    }

    /// A framing hint while upright, out of line and pre-attempt — suppressed
    /// within `afterInversionGapS` of coming down, at most one per
    /// `hintGapS`, and never the same one twice inside `hintRepeatGapS`.
    private mutating func hintCue(
        tMs: Int,
        framing: FramingStatus?,
        inverted: Bool,
        recording: Bool
    ) -> Cue? {
        guard settings.framingHints, !recording, !inverted, !inLine, let framing else { return nil }
        guard let hint = Self.hint(for: framing, upright: !inverted) else { return nil }
        if let ended = inversionEndedTMs, Double(tMs - ended) < Self.afterInversionGapS * 1000 {
            return nil
        }
        guard allowed(tMs: tMs, gapS: Self.debounceGapS) else { return nil }
        if let lastHintTMs, Double(tMs - lastHintTMs) < Self.hintGapS * 1000 { return nil }
        if hint == lastHint, let lastHintTMs, Double(tMs - lastHintTMs) < Self.hintRepeatGapS * 1000
        {
            return nil
        }
        lastHint = hint
        lastHintTMs = tMs
        return spoken(.framing(hint), tMs: tMs)
    }

    /// A form cue (chainlink #97) waiting for its turn: queued (the correction
    /// is the point, not a stale edge), held back by the coach-mode
    /// ``formGapS``, dropped the moment its toggle goes off. It is checked
    /// after the framing hints, so a hint always outranks it. A line in
    /// progress outranks it too — "Line. Hold it." is not followed by a
    /// correction to the shape the athlete has left (chainlink #100 review).
    private mutating func formCue(tMs: Int) -> Cue? {
        guard let fault = pendingForm, !inLine else { return nil }
        guard settings.voiceCues, settings.formCues else {
            pendingForm = nil
            return nil
        }
        guard allowed(tMs: tMs, gapS: Self.formGapS) else { return nil }
        pendingForm = nil
        return spoken(.form(fault), tMs: tMs)
    }

    // MARK: - The mapping

    /// What a framing verdict is worth saying, and `nil` for the verdicts that
    /// say nothing (`.ok` is `ready`'s job, `.noPerson` is "step into the
    /// frame" on the border, which the user is already looking at).
    ///
    /// - Parameter upright: the athlete is not upside down. The
    ///   `.partlyOutOfFrame` rules ask it: an upright body clipped at the top
    ///   only means the phone is on the floor looking up at them — *raise* it;
    ///   the same clip upside down means *step back*.
    public static func hint(for status: FramingStatus, upright: Bool) -> FramingHint? {
        switch status {
        case .ok, .noPerson:
            return nil
        case .tooSmall:
            return .comeCloser
        case .multiplePeople:
            return .onlyYou
        case .partlyOutOfFrame(let edges, _):
            if edges.contains(.left) || edges.contains(.right) {
                return .moveToMiddle
            }
            if edges == [.top], upright {
                return .raisePhone
            }
            // Top, bottom, or a required joint nobody can see: the border's own
            // message for all of them is a variant of "step back".
            return .stepBack
        }
    }

    // MARK: - The bookkeeping

    /// Has the global debounce been spent since the last cue?
    private func allowed(tMs: Int, gapS: Double) -> Bool {
        guard let last = lastSpokenTMs else { return true }
        return Double(tMs - last) >= gapS * 1000
    }

    /// Record a cue as spoken and hand it back.
    private mutating func spoken(_ cue: Cue, tMs: Int) -> Cue {
        lastSpokenTMs = tMs
        lastCue = cue
        return cue
    }
}
