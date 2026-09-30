import AVFoundation
import Foundation
import HandstandCore
import Observation
import UIKit

/// What the Record screen shows and when: camera permission, the live
/// framing verdict, the record/stop toggle, the running clock and the file
/// that came out of the last take.
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

    /// The hold the picker shows. The Record screen keeps it in step with
    /// `@AppStorage("holdType")`; it is copied into `activeTake` the moment
    /// recording starts, so a tap on the picker afterwards cannot rewrite
    /// what the sidecar says. Hold type is the only setup the user does —
    /// wall, steps, camera angle and phases are detected (#72).
    var holdType: HoldType = .default

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
            do {
                try RecordingMetadata(
                    holdType: take.hold,
                    recordedAt: take.startedAt,
                    appVersion: RecordingMetadata.currentAppVersion
                ).write(for: url)
            } catch {
                errorMessage = "The recording was kept, but its sidecar could not be written: "
                    + error.localizedDescription
            }
            // The done screen shows what the *file* says, not what the
            // stopwatch said — same VideoInfo the picker screen reads.
            Task { [weak self] in
                do {
                    let info = try await VideoInfoReader.read(url: url)
                    self?.finished = info
                } catch {
                    self?.errorMessage = "The recording could not be read back: \(error.localizedDescription)"
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
}
