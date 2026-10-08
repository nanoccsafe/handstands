import AVFoundation
import Foundation
import HandstandCore
import Observation
import UIKit

/// What the Record screen shows and when: camera permission, the live
/// framing verdict, the record/stop toggle, the running clock, the file
/// that came out of the last take — and the live cues (chainlink #91): the
/// scheduler's answer each tick, spoken and shown as a banner.
///
/// This is the main-actor half of capture (chainlink #46). The AVFoundation
/// half lives in `CaptureEngine`, off the main actor entirely; everything
/// here reaches it through methods that hop onto its queue, and everything
/// it reports comes back through `@Sendable` closures that hop here. No
/// networking, no microphone: recordings are written to
/// `Application Support/Recordings/` and stay on the phone.
@MainActor
@Observable
final class CaptureService {
    /// Where the screen is in its life cycle.
    enum Phase: Equatable, Sendable {
        /// Asking iOS for the camera, or the session is coming up.
        case starting
        /// There is no camera to ask for (the simulator), or the session
        /// could not be configured.
        case unavailable
        /// Camera access was refused: show why and offer Settings.
        case denied
        /// Live camera, framing guide, record button.
        case ready
    }

    private(set) var phase: Phase = .starting
    /// What the "no camera" screen says — the required sentence when the
    /// device simply has none, the configuration error when it has one that
    /// will not come up.
    private(set) var unavailableMessage = "Camera not available on this device"
    /// The framing guide's latest verdict; `.noPerson` (red) until the first
    /// check of the first frame.
    private(set) var framingStatus: FramingStatus = .noPerson
    private(set) var isRecording = false
    /// Seconds since the record button went down, updated a few times a
    /// second; the finished file's real duration replaces it on the done
    /// screen.
    private(set) var elapsed: TimeInterval = 0
    /// The last finished recording, ready for `RecordingDoneView`.
    private(set) var finished: VideoInfo?
    /// The hold the take on the done screen was recorded for (chainlink
    /// #66): the one that was selected when *that* recording started, not
    /// whatever the picker says now.
    private(set) var finishedHoldType: HoldType = .default
    private(set) var errorMessage: String?
    /// The session the preview layer displays (set once the engine exists).
    private(set) var session: AVCaptureSession?

    /// Where a finished take is recorded into History (chainlink #51).
    /// Injected by `RecordView` from its environment's `modelContext`; when
    /// it is nil (a screen without a container) nothing here fails — the
    /// video is still written, and `reconcile()` adds it to History the
    /// next time that screen opens.
    var sessionStore: SessionStore?

    /// The hold the picker shows. The Record screen keeps it in step with
    /// `@AppStorage("holdType")`; it is copied into `activeTake` the moment
    /// recording starts, so a tap on the picker afterwards cannot rewrite
    /// what the sidecar says. Hold type is the only setup the user does —
    /// wall, steps, camera angle and phases are detected (#72).
    var holdType: HoldType = .default

    // MARK: - Live cues (chainlink #91): the state

    /// The cue toggles in force, pushed here from the Record screen's
    /// `@AppStorage` (the gear on Home writes the same keys). Copied into the
    /// scheduler on every tick, so a change mid-session applies at once
    /// without throwing away the debounce history.
    var cueSettings: CueSettings = .init()

    /// The voice. `nil` until the first cue — the synthesizer and the audio
    /// session are made on demand — and a fake the wiring test installs.
    var speaker: CueSpeaking?

    /// The cue the banner is showing (`nil` once its 2 s are up): what
    /// `RecordView` draws under the border, silent mode's half of the cue.
    private(set) var cueBanner: Cue?

    /// What the take in flight has seen of live mode — the source of the
    /// sidecar's five optional fields. Internal rather than private so the
    /// wiring test can read what a fake take accumulated; reset when
    /// recording starts.
    var liveStats = LiveTakeStats()

    /// The cue policy: one `decide` per tick, all the pacing rules.
    private var scheduler = CueScheduler()

    /// When the banner goes away again.
    private var bannerTask: Task<Void, Never>?

    private var engine: CaptureEngine?
    private var didStart = false
    private var isShutdown = false
    private var recordStart: Date?
    /// The take in flight: which hold was selected when the record button
    /// went down, and when — the sidecar's `hold_type` and `recorded_at`
    /// come from here (both captured at START).
    private var activeTake: (hold: HoldType, startedAt: Date)?
    private var ticker: Task<Void, Never>?

    /// The sentence under the border — `HandstandCore` decides it.
    var framingMessage: String { FramingCheck.message(for: framingStatus) }
    /// The border colour for that verdict.
    var borderTone: FramingTone { FramingDisplay.tone(for: framingStatus) }

    // MARK: - Live cues (chainlink #91): the ticks

    /// One tick of the live pipeline: the detector's events, the framing
    /// verdict in force, and whether the red light is on.
    ///
    /// Internal, with every input spelled out rather than read from the
    /// service's own state, so the wiring test can drive it with a fake
    /// detector and no camera anywhere near it. The engine's callback arrives
    /// through `DispatchQueue.main`, which is FIFO from the single analysis
    /// queue — the cue order is the frame order.
    func liveUpdate(_ frame: LiveFrame, framing: FramingStatus?, recording: Bool) {
        if recording {
            liveStats.record(frame: frame)
        }
        scheduler.settings = cueSettings
        guard let cue = scheduler.decide(
            tMs: frame.tMs,
            events: frame.events,
            framing: framing,
            inverted: frame.inverted,
            recording: recording
        ) else { return }

        // Everything the scheduler hands back is spoken: the toggles are its
        // own — with voice cues off it returns framing hints and nothing
        // else. Never interrupted: `SpeechCueSpeaker` queues rather than stops.
        if speaker == nil { speaker = SpeechCueSpeaker() }
        speaker?.speak(cue)
        showBanner(cue)

        if recording, let firstLiveTMs = liveStats.firstLiveTMs {
            liveStats.cues.append(
                SpokenCue(
                    tMs: Swift.max(0, frame.tMs - firstLiveTMs),
                    cue: CueText.identifier(for: cue)
                )
            )
        }
    }

    /// A frame was dropped from the recording (`CaptureEngine`'s `didDrop`):
    /// one of the guard's triggers and one of the sidecar's numbers.
    func recordingFrameDropped() {
        liveStats.droppedFrames += 1
    }

    /// The banner: the same cue, as words, for 2 s — the half of every cue
    /// that works with the sound off. A newer cue replaces it and restarts
    /// the 2 s.
    private func showBanner(_ cue: Cue) {
        cueBanner = cue
        bannerTask?.cancel()
        bannerTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(2))
            guard !Task.isCancelled else { return }
            self?.cueBanner = nil
        }
    }

    /// Called when the Record screen opens: find the camera, ask for it if
    /// this is the first time, then bring the session up.
    func start() async {
        if didStart {
            engine?.start()
            return
        }
        didStart = true
        phase = .starting

        // The simulator — and any device without a back wide camera — has no
        // capture device at all. Say so instead of asking for a permission
        // with nothing behind it (and instead of crashing on a nil device).
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            unavailableMessage = "Camera not available on this device"
            phase = .unavailable
            return
        }

        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            break
        case .notDetermined:
            // Camera access is requested when Record opens, as the task asks.
            let granted = await AVCaptureDevice.requestAccess(for: .video)
            guard granted else {
                phase = .denied
                return
            }
        case .denied, .restricted:
            phase = .denied
            return
        @unknown default:
            phase = .denied
            return
        }

        // The screen can go away while iOS is still asking (or while the
        // answer is coming back): bringing the session up behind a dismissed
        // view would leave the camera running with nobody looking at it, so
        // `shutdown` wins the race by setting `isShutdown` first — both run
        // on the main actor, so the flag is exact.
        guard !isShutdown, !Task.isCancelled else { return }

        let engine = CaptureEngine(
            device: device,
            onFraming: { [weak self] status in
                Task { @MainActor in self?.framingStatus = status }
            },
            onLive: { [weak self] frame in
                // One producer (the analysis queue) and a FIFO main queue, so
                // the ticks reach the scheduler in frame order — which is
                // what the cue order depends on. `assumeIsolated`: the main
                // queue *is* the main actor.
                DispatchQueue.main.async {
                    guard let self else { return }
                    MainActor.assumeIsolated {
                        self.liveUpdate(
                            frame, framing: self.framingStatus, recording: self.isRecording)
                    }
                }
            },
            onRecordingFrameDropped: { [weak self] in
                DispatchQueue.main.async {
                    guard let self else { return }
                    MainActor.assumeIsolated { self.recordingFrameDropped() }
                }
            },
            onRecordingFinished: { [weak self] result in
                Task { @MainActor in self?.recordingFinished(result) }
            },
            onFailure: { [weak self] message in
                Task { @MainActor in self?.configurationFailed(message) }
            }
        )
        self.engine = engine
        session = engine.session
        engine.start()
        phase = .ready
    }

    /// The record/stop button.
    func toggleRecording() {
        if isRecording {
            stopRecording()
        } else {
            startRecording()
        }
    }

    /// Back to the live camera from the done screen; the file is already on
    /// disk, so nothing here touches it.
    func recordAgain() {
        finished = nil
        errorMessage = nil
        elapsed = 0
        recordStart = nil
    }

    /// The Record screen went away: stop the clock, finish an open file on
    /// the engine's queue (the file completes even if this screen is never
    /// seen again) and stop the session.
    func shutdown() {
        isShutdown = true
        ticker?.cancel()
        ticker = nil
        if isRecording {
            isRecording = false
            recordStart = nil
            engine?.stopRecording()
        }
        engine?.stop()
    }

    // MARK: - Private

    private func startRecording() {
        guard phase == .ready, let engine, !isRecording else { return }
        do {
            let url = try RecordingFile.url()
            errorMessage = nil
            finished = nil
            elapsed = 0
            // Captured at START: the sidecar records the hold (and the time)
            // this take began with, whatever the picker says while it runs.
            let startedAt = Date()
            recordStart = startedAt
            activeTake = (hold: holdType, startedAt: startedAt)
            // The sidecar's live numbers (#91) count this take only.
            liveStats = LiveTakeStats()
            isRecording = true
            startTicker()
            engine.startRecording(url: url)
        } catch {
            errorMessage = "Could not start recording: \(error.localizedDescription)"
        }
    }

    private func stopRecording() {
        guard isRecording else { return }
        isRecording = false
        ticker?.cancel()
        ticker = nil
        if let recordStart {
            elapsed = Date().timeIntervalSince(recordStart)
        }
        recordStart = nil
        engine?.stopRecording()
    }

    /// The elapsed clock: one hop of the clock every 200 ms, cancelled by
    /// `stopRecording`/`shutdown`.
    private func startTicker() {
        ticker?.cancel()
        ticker = Task { [weak self, recordStart] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(200))
                guard !Task.isCancelled, let self, let recordStart else { return }
                self.elapsed = Date().timeIntervalSince(recordStart)
            }
        }
    }

    private func recordingFinished(_ result: Result<URL, any Error>) {
        switch result {
        case let .success(url):
            // The sidecar rides next to the movie (`20260928-143059.json`),
            // written for every successful recording: the hold captured when
            // this take started plus its start time. If it cannot be
            // written, the video is kept — never deleted — and errorMessage
            // explains the problem.
            let take = activeTake ?? (hold: HoldType.default, startedAt: Date())
            activeTake = nil
            finishedHoldType = take.hold
            // The sidecar's live fields (chainlink #91): the rate in force at
            // the end of the take, what it actually managed, what an inference
            // cost on average, how many frames the *recording* lost, and what
            // was said. `nil` for anything the take never saw — a build
            // without the pose model writes the same sidecar it always did.
            let stats = liveStats
            let achieved: Double? =
                stats.fpsTarget != nil && elapsed > 0
                ? Double(stats.liveFrames) / elapsed
                : nil
            let inference: Double? =
                stats.liveFrames > 0 ? stats.inferenceMsTotal / Double(stats.liveFrames) : nil
            let metadata = RecordingMetadata(
                holdType: take.hold,
                recordedAt: take.startedAt,
                appVersion: RecordingMetadata.currentAppVersion,
                liveFpsTarget: stats.fpsTarget.map(Self.tenth),
                liveFpsAchieved: achieved.map(Self.tenth),
                avgInferenceMs: inference.map(Self.tenth),
                recordingFramesDropped: stats.droppedFrames,
                cuesSpoken: stats.cues.isEmpty ? nil : stats.cues
            )
            do {
                try metadata.write(for: url)
            } catch {
                errorMessage = "The recording was kept, but its sidecar could not be written: "
                    + error.localizedDescription
            }
            // The done screen shows what the *file* says, not what the
            // stopwatch said — same VideoInfo the picker screen reads. The
            // same read feeds History (chainlink #51): one Session row per
            // recording, right after the sidecar that describes it.
            Task { [weak self] in
                let info: VideoInfo
                do {
                    info = try await VideoInfoReader.read(url: url)
                } catch {
                    self?.errorMessage = "The recording could not be read back: \(error.localizedDescription)"
                    return
                }
                guard let self else { return }
                self.finished = info
                do {
                    _ = try self.sessionStore?.add(movie: url, metadata: metadata, info: info)
                } catch {
                    // Never at the cost of the video: the take stays on
                    // disk and `reconcile()` (History's `.task`) adds it
                    // the next time that screen opens.
                    self.errorMessage = "The recording was kept, but it is missing from History for now: "
                        + error.localizedDescription
                }
            }
        case let .failure(error):
            activeTake = nil
            if let writerError = error as? RecordingWriter.WriterError,
               case .noFrames = writerError {
                errorMessage = "That take was too short to save — hold it a moment longer."
            } else {
                errorMessage = "Recording failed: \(error.localizedDescription)"
            }
        }
    }

    private func configurationFailed(_ message: String) {
        unavailableMessage = message
        phase = .unavailable
    }

    /// One decimal place — `9.7264…` → `9.7`, `10.0` → `10.0` — the rounding
    /// the sidecar's live numbers are written with (a tuning table read with
    /// `jq` does not need more, and must not carry a page of digits).
    private static func tenth(_ value: Double) -> Double {
        (value * 10).rounded() / 10
    }
}

/// What one take has seen of live mode (chainlink #91) — the source of the
/// sidecar's five optional fields.
///
/// Counted only while the red light is on: `live_fps_target` is the rate in
/// force at the take's most recent frame (the performance guard may have
/// halved it mid-take), `live_fps_achieved` the frames actually analysed over
/// the take's length, `avg_inference_ms` what they cost, and
/// `recording_frames_dropped` what the *recording* lost. `cues` is what was
/// said, each with its `t_ms` on the capture clock relative to the take's
/// first live frame — the numbers "how much feedback is too much" is tuned by.
struct LiveTakeStats {
    /// Live frames analysed during the take.
    var liveFrames = 0
    /// Their inference time summed — `/ liveFrames` is the average.
    var inferenceMsTotal = 0.0
    /// The live rate at the most recent frame of the take, fps.
    var fpsTarget: Double?
    /// Frames the capture output dropped from the recording.
    var droppedFrames = 0
    /// The capture timestamp of the take's first live frame: the zero every
    /// spoken cue's `t_ms` is measured from.
    var firstLiveTMs: Int?
    /// The cues spoken during the take, in time order.
    var cues: [SpokenCue] = []

    /// Count one live frame of this take.
    mutating func record(frame: LiveFrame) {
        liveFrames += 1
        inferenceMsTotal += frame.inferenceMs
        fpsTarget = frame.fpsTarget
        if firstLiveTMs == nil { firstLiveTMs = frame.tMs }
    }
}
