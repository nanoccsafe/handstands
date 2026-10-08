import CoreVideo
import Foundation
import HandstandCore

// --------------------------------------------------------------------------- #
// Live pose for the cues (chainlink #91): the camera frames in, one ``LiveFrame``
// per analysed frame out — the joints MediaPipe saw, the detector's events and
// what the frame cost.
//
// The pipeline is the *throttled* half of the Record screen. The framing guide
// already runs on the engine's analysis queue at 5 fps; this runs **on the same
// queue**, at the live rate (10 fps by default), with three rules that protect
// the recording, which must never drop a frame because of live mode:
//
// 1. **At most one inference in flight.** While MediaPipe is thinking, new
//    frames are *skipped*, not queued — live analysis drops its own frames
//    first, by skipping, and the capture queue never waits for it.
// 2. **Offered, not pulled.** `offer` runs on the capture queue and only hops
//    over when the throttle says a frame is due, so the delegate's work stays
//    the append it has always been.
// 3. **The guard.** The average inference over a 2 s window above 80 ms — or
//    any frame dropped from the *recording* — halves the live rate, to a floor
//    of 4 fps, and says so in the log.
//
// Orientation: one pass per frame, never both (two passes are too expensive
// live). The rule is `VisionPoseCore.AutoRotation`'s, made causal — the frame
// just seen decides whether the *next* one is fed turned 180°:
// `mean y of both wrists > mean y of both ankles` means "upside down, rotate
// next". VIDEO-mode tracking then sees a consistent orientation through a hold
// and flips only when the athlete comes down.
// --------------------------------------------------------------------------- #

/// One frame's hand-off from the capture queue to the analysis queue
/// (chainlink #91): the buffer and the timestamp the throttle compared.
///
/// A struct for the same reason `CaptureEngine`'s `FramingFrame` is one —
/// `@unchecked Sendable` is honest because the buffer is immutable once the
/// camera has delivered it and is read only by the analysis queue from here
/// on. `DispatchQueue.async` wants a `Sendable` payload, and a bare
/// `CVPixelBuffer` is not one.
private struct LiveOffer: @unchecked Sendable {
    let buffer: CVPixelBuffer
    let tMs: Int

    init(buffer: CVPixelBuffer, tMs: Int) {
        self.buffer = buffer
        self.tMs = tMs
    }
}

/// One analysed live frame — what `CaptureService` needs to decide (and to
/// account) in one value.
struct LiveFrame: Equatable, Sendable {
    /// The capture timestamp of the frame, milliseconds — the same monotonic
    /// clock `LiveLineDetector` measures its hysteresis in.
    let tMs: Int
    /// What the detector noticed about this frame (possibly nothing).
    let events: [LiveEvent]
    /// Was the athlete upside down in this frame?
    let inverted: Bool
    /// How long this frame's inference took, milliseconds — the guard's input
    /// and the sidecar's `avg_inference_ms`.
    let inferenceMs: Double
    /// The live rate in force when this frame was offered, fps.
    let fpsTarget: Double
}

/// How often live frames are offered to MediaPipe — and the lever the
/// performance guard pulls.
///
/// The interval and the rate are the same number seen from two sides; both are
/// read on the capture queue and written from the analysis queue, so both live
/// behind one lock.
final class LiveRateController: @unchecked Sendable {
    /// The rate live mode starts at (chainlink #91: "at most 10 fps").
    static let initialFps = 10.0

    /// Where a halved rate stops: below 4 fps the cues arrive too late to be
    /// worth speaking.
    static let minimumFps = 4.0

    /// The average inference budget, milliseconds — over
    /// ``InferenceGuard/windowS``, going over halves the rate.
    static let inferenceBudgetMs = 80.0

    /// The window the average is taken over, seconds.
    static let inferenceWindowS = 2.0

    private let lock = NSLock()
    private var fps: Double

    init(fps: Double = LiveRateController.initialFps) {
        self.fps = Swift.max(Self.minimumFps, fps)
    }

    /// The live rate right now, fps — what `LiveFrame.fpsTarget` carries.
    var targetFps: Double {
        lock.withLock { fps }
    }

    /// Seconds between two live frames at the current rate.
    var intervalS: TimeInterval {
        lock.withLock { 1.0 / fps }
    }

    /// Halve the rate (never below ``minimumFps``) and log it. Returns whether
    /// the rate actually changed, so a caller does not log twice for one
    /// slow stretch.
    @discardableResult
    func halve(reason: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard fps > Self.minimumFps else { return false }
        fps = Swift.max(Self.minimumFps, fps / 2)
        NSLog("Handstand: live mode %@ — rate halved to %.1f fps", reason, fps)
        return true
    }
}

/// The performance guard's arithmetic, on its own: does the last
/// `windowS` seconds of inference average above `budgetMs`?
///
/// Pure (timestamps in, one average out) so the rule that protects the
/// recording is testable without a camera, a model or a phone. Two details
/// are the whole trick: the verdict needs a **full window** of evidence
/// first — the opening seconds of a session include the cold start, and one
/// slow frame is not a trend — and the window is emptied when it does
/// verdict, so one slow stretch halves the rate once rather than all the way
/// to the floor.
struct InferenceGuard {
    struct Sample: Equatable {
        var tMs: Int
        var ms: Double
    }

    /// The window, seconds.
    let windowS: Double

    /// The average to compare against, milliseconds.
    let budgetMs: Double

    /// The samples inside the window, newest last.
    private(set) var samples: [Sample] = []

    /// When the current window's first sample arrived, `nil` between
    /// verdicts — nothing is judged until `windowS` has been measured.
    private var windowStartTMs: Int?

    init(windowS: Double, budgetMs: Double) {
        self.windowS = windowS
        self.budgetMs = budgetMs
    }

    /// Record one inference; `nil` while the window is missing, short or
    /// within budget, and the over-budget **average** once it is a full
    /// window over budget (which empties it: the next verdict needs a fresh
    /// `windowS` of evidence).
    mutating func record(tMs: Int, ms: Double) -> Double? {
        if windowStartTMs == nil { windowStartTMs = tMs }
        samples.append(Sample(tMs: tMs, ms: ms))
        let cutoff = Int(windowS * 1000)
        samples.removeAll { tMs - $0.tMs > cutoff }

        guard let start = windowStartTMs, tMs - start >= cutoff else { return nil }
        guard samples.count > 1 else { return nil }
        let average = samples.reduce(0.0) { $0 + $1.ms } / Double(samples.count)
        guard average > budgetMs else { return nil }
        samples.removeAll(keepingCapacity: true)
        windowStartTMs = nil
        return average
    }
}

/// The live pose pass — one landmarker, one detector, one throttle
/// (chainlink #91).
///
/// Queue confinement: `offer`, `noteRecordingDrop` and the `stopped` /
/// `inFlight` flags are the capture queue's; everything else is the analysis
/// queue's (the engine passes it in). The two meet only through the lock and
/// through hops.
final class LivePosePipeline: @unchecked Sendable {
    /// Every how-many seconds of line time the optional time marks fire.
    static let timeMarkEverySeconds = 5

    private let modelURL: URL
    /// The engine's analysis queue: the same one the framing guide runs on, so
    /// the two never read a pixel buffer concurrently.
    private let queue: DispatchQueue
    private let rate: LiveRateController
    private let onFrame: @Sendable (LiveFrame) -> Void

    // Capture-queue side.
    private let lock = NSLock()
    private var lastOfferTMs: Int?
    private var inFlight = false
    private var stopped = false

    // Analysis-queue side.
    private let downscaler = FramingAnalyzer()
    private var landmarker: MediaPipeLandmarker?
    private var detector: LiveLineDetector
    private var inferenceGuard: InferenceGuard
    private var rotateNextFrame = false

    init(
        modelURL: URL,
        queue: DispatchQueue,
        rate: LiveRateController,
        onFrame: @escaping @Sendable (LiveFrame) -> Void
    ) {
        self.modelURL = modelURL
        self.queue = queue
        self.rate = rate
        self.onFrame = onFrame
        self.detector = LiveLineDetector(
            config: .init(), timeMarksEverySeconds: Self.timeMarkEverySeconds)
        self.inferenceGuard = InferenceGuard(
            windowS: LiveRateController.inferenceWindowS,
            budgetMs: LiveRateController.inferenceBudgetMs
        )
    }

    // MARK: - The capture queue's side

    /// Offer one camera frame. Cheap and synchronous: it either starts an
    /// analysis (when the throttle is due and nothing is in flight) or drops
    /// this frame, which is exactly what "live analysis drops its frames
    /// first" means.
    ///
    /// - Parameter tMs: the capture timestamp in milliseconds, increasing.
    func offer(_ pixelBuffer: CVPixelBuffer, tMs: Int) {
        let offer = LiveOffer(buffer: pixelBuffer, tMs: tMs)
        lock.lock()
        guard !stopped, !inFlight else {
            lock.unlock()
            return
        }
        let intervalMs = Int((rate.intervalS * 1000).rounded())
        if let lastOfferTMs, tMs - lastOfferTMs < intervalMs {
            lock.unlock()
            return
        }
        lastOfferTMs = tMs
        inFlight = true
        lock.unlock()
        queue.async { [weak self] in
            self?.analyse(offer)
        }
    }

    /// A frame was dropped from the *recording*: the guard's second trigger.
    func noteRecordingDrop() {
        let rate = self.rate
        queue.async {
            rate.halve(reason: "a recording frame was dropped")
        }
    }

    // MARK: - The analysis queue's side

    private func analyse(_ offer: LiveOffer) {
        let pixelBuffer = offer.buffer
        let tMs = offer.tMs
        defer { lock.withLock { inFlight = false } }

        guard let landmarker = landmarker ?? makeLandmarker() else { return }
        guard let frame = downscaler.downscaled(
            pixelBuffer, maxDimension: FramingAnalyzer.maxDimension)
        else { return }

        // One pass only: this frame turned 180° when the last one said so.
        let rotated = rotateNextFrame
        let input: CVPixelBuffer
        if rotated {
            guard let turned = try? MediaPipeFrameRotation.turn180(frame) else { return }
            input = turned
        } else {
            input = frame
        }

        let started = DispatchTime.now()
        let pose: [MediaPipeLandmark]?
        do {
            pose = try landmarker.detectForVideo(input, tMs: tMs)
        } catch {
            NSLog("Handstand: live pose failed on a frame: %@", "\(error)")
            return
        }
        let inferenceMs =
            Double(DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds) / 1_000_000

        // Display pixels of the frame MediaPipe was shown, turned back from the
        // 180° pass when it was — the space `LiveLineDetector` documents.
        let joints = Self.displayJoints(
            from: pose,
            rotated: rotated,
            width: CVPixelBufferGetWidth(frame),
            height: CVPixelBufferGetHeight(frame)
        )
        rotateNextFrame = Self.rotateRule(joints, keeping: rotateNextFrame)
        let events = detector.update(tMs: tMs, joints: joints)

        if let overBudget = inferenceGuard.record(tMs: tMs, ms: inferenceMs) {
            rate.halve(
                reason: String(
                    format: "averaged %.0f ms of inference over %.0f s",
                    overBudget, LiveRateController.inferenceWindowS))
        }
        onFrame(
            LiveFrame(
                tMs: tMs,
                events: events,
                inverted: detector.isInverted,
                inferenceMs: inferenceMs,
                fpsTarget: rate.targetFps
            ))
    }

    /// The landmarker for live mode: its own VIDEO-mode instance — never the
    /// analysis one, which is built per extraction — created on first use
    /// (loading the model is not something the capture queue may wait for).
    /// A build without the model (it is never committed) simply has no live
    /// cues: the pipeline stops offering frames, and the log says why.
    private func makeLandmarker() -> MediaPipeLandmarker? {
        do {
            let made = try MediaPipeLandmarker(modelURL: modelURL)
            landmarker = made
            return made
        } catch {
            lock.withLock { stopped = true }
            NSLog("Handstand: live cues disabled (the pose model could not be opened): %@", "\(error)")
            return nil
        }
    }

    // MARK: - The rules, on their own

    /// One pose's keypoints in display pixels: MediaPipe's normalised
    /// coordinates, mapped back through the 180° turn when the rotated pass
    /// saw them — the same `x = 1 − x` / `y = 1 − y` map-back
    /// `MediaPipeClipExtractor` does, then scaled by the frame's size exactly
    /// as `_display_pixels` scales them.
    static func displayJoints(
        from pose: [MediaPipeLandmark]?,
        rotated: Bool,
        width: Int,
        height: Int
    ) -> [Joint: Keypoint] {
        guard let pose, width > 1, height > 1 else { return [:] }
        var joints: [Joint: Keypoint] = [:]
        for (joint, landmarkIndex) in mediaPipeLandmarkIndex {
            guard landmarkIndex < pose.count else { continue }
            let landmark = pose[landmarkIndex]
            guard let visibility = landmark.visibility, visibility.isFinite,
                landmark.x.isFinite, landmark.y.isFinite
            else { continue }
            let x = rotated ? 1 - landmark.x : landmark.x
            let y = rotated ? 1 - landmark.y : landmark.y
            joints[joint] = Keypoint(
                x: x * Double(width - 1),
                y: y * Double(height - 1),
                visibility: visibility
            )
        }
        return joints
    }

    /// `VisionPoseCore.AutoRotation`'s rule, in the same form: the next frame
    /// is turned 180° when the body just seen looks upside down — mean y of
    /// both wrists **below** (greater than) mean y of both ankles.
    ///
    /// A frame with nobody in it keeps the previous decision (the model said
    /// nothing); a pose missing a wrist or an ankle is judged "not inverted",
    /// exactly as `AutoRotation.isInverted` judges it.
    static func rotateRule(_ joints: [Joint: Keypoint], keeping previous: Bool) -> Bool {
        guard !joints.isEmpty else { return previous }
        guard let leftWrist = joints[.leftWrist], let rightWrist = joints[.rightWrist],
            let leftAnkle = joints[.leftAnkle], let rightAnkle = joints[.rightAnkle],
            leftWrist.y.isFinite, rightWrist.y.isFinite,
            leftAnkle.y.isFinite, rightAnkle.y.isFinite
        else { return false }
        return (leftWrist.y + rightWrist.y) / 2 > (leftAnkle.y + rightAnkle.y) / 2
    }
}
