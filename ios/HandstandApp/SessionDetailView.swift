import AVFoundation
import Foundation
import HandstandCore
import SwiftUI
import UIKit

/// One recording, played (chainlink #51) with its stress diagram over it
/// (chainlink #48): the movie the row points at, exactly the facts the list
/// shows beside it, the analysis of chainlink #47 (read the frames, run
/// Apple Vision + the pipeline, save the result back to this row), the
/// overlay drawn from the pose cache — a skeleton coloured by how far the
/// form is off, the stack line and the centre of mass — the summary under
/// the player (chainlink #49): a heat strip of the whole clip that seeks
/// where you tap, the worst moment of the hold as a tappable card, and up
/// to three coaching cues — the export of that same picture as a video
/// (chainlink #50): diagram and strip burned into an `.mp4` in Caches, with
/// progress, cancel and a share sheet, deleted when this screen goes away —
/// and the one destructive action in the app: deleting takes the video off
/// the phone too, after the same confirmation the list uses.
@MainActor
struct SessionDetailView: View {
    let session: Session
    /// The store the row came from: it owns the Recordings folder, so it is
    /// what `delete` and `movieURL(for:)` need.
    let store: SessionStore

    @Environment(\.dismiss) private var dismiss
    @State private var confirmingDelete = false
    @State private var deletionError: String?
    /// The player for this row's movie, made once — body would otherwise
    /// build a fresh `AVPlayer` (and restart the video) on every redraw.
    @State private var player: AVPlayer?
    /// The player's periodic time observer (every 1/30 s), removed when the
    /// screen goes away — the player outlives no callback that outlives it.
    @State private var timeObserver: Any?
    /// The analysis of this row: button, progress, results. Owned here, so
    /// leaving the screen stops a run in flight (see `onDisappear`).
    @State private var service = AnalysisService()
    /// Where playback is, in the clip's own milliseconds — the clock the
    /// diagram's frames are stamped with (see `PlaybackClock`).
    @State private var clock = PlaybackClock()
    /// Is the movie playing (what the play/pause button shows)?
    @State private var isPlaying = false
    /// Is the scrubber being dragged, and where it was dragged to? While it
    /// is, the slider shows its own value and the time observer stays out
    /// of the way (the movie is paused for the drag).
    @State private var scrubbing = false
    @State private var scrubSeconds = 0.0
    /// The analysis the overlay is drawn from — from the pose cache when the
    /// screen opens on an analysed take, from the run that just finished
    /// otherwise. `nil` means no overlay: an un-analysed take shows the
    /// plain picture until "Analyse" runs.
    @State private var diagramAnalysis: Analysis?
    /// The reference that analysis was scored/severed against, for the
    /// legend's note line.
    @State private var diagramReference: ScoreReference?
    /// Is the overlay showing? On by default whenever there is one.
    @State private var showDiagram = true
    /// The summary (chainlink #49), computed once per analysis in
    /// `loadDiagram` — never per playback frame. The strip's bins, the
    /// worst moment (and its diagram, so the card's overlay is not
    /// rebuilt either) and the coaching cues.
    @State private var heatBins: [HeatBin] = []
    @State private var worstMoment: WorstMoment?
    @State private var worstMomentDiagram: DiagramFrame?
    @State private var cues: [CoachingCue] = []
    /// The annotated video export (chainlink #50), owned here so leaving
    /// the screen stops a run in flight and takes its cached file with it
    /// (see `onDisappear`). Starts `.idle`, like the analysis service.
    @State private var exporter = ExportService()

    /// Is a run in flight — the state the screen has to stay awake for.
    private var isAnalysing: Bool {
        if case .running = service.state { return true }
        return false
    }

    /// The video's display size in pixels — the space the analysis's
    /// points are written in, and what the overlay's aspect-fit maths
    /// starts from.
    private var videoSize: CGSize {
        CGSize(width: session.width, height: session.height)
    }

    /// The diagram frame to draw right now: `nil` while the overlay is off,
    /// until an analysis is loaded, or while the playhead is before the
    /// clip's first frame (nothing has been seen yet).
    private var currentDiagram: DiagramFrame? {
        guard showDiagram, let analysis = diagramAnalysis else { return nil }
        guard
            let index = StressDiagram.frameIndex(
                atMs: clock.playheadMs, in: analysis.features.tMs)
        else {
            return nil
        }
        return StressDiagram.frame(
            index,
            analysis: analysis,
            reference: diagramReference,
            height: Double(session.height)
        )
    }

    var body: some View {
        List {
            Section {
                if let player {
                    VStack(spacing: 8) {
                        ZStack {
                            PlayerLayerView(player: player)
                            if let diagram = currentDiagram {
                                StressDiagramOverlay(diagram: diagram, videoSize: videoSize)
                            }
                        }
                        .frame(height: 260)
                        .background(Color.black)

                        if currentDiagram != nil {
                            StressDiagramLegend(hasReference: diagramReference != nil)
                        }

                        playbackControls(player)

                        // The summary (chainlink #49): everything below is
                        // computed once per analysis in `loadDiagram`, so
                        // only the playhead moves while the movie plays.
                        if let analysis = diagramAnalysis,
                            let firstMs = analysis.features.tMs.first,
                            let lastMs = analysis.features.tMs.last
                        {
                            HeatStripView(
                                bins: heatBins,
                                firstMs: firstMs,
                                lastMs: lastMs,
                                playheadMs: clock.playheadMs,
                                onSeek: { tMs in
                                    seek(player, to: Double(tMs) / 1000.0)
                                }
                            )
                            if let moment = worstMoment, let diagram = worstMomentDiagram {
                                WorstMomentCard(
                                    movieURL: store.movieURL(for: session),
                                    moment: moment,
                                    diagram: diagram,
                                    videoSize: videoSize,
                                    onTap: { seekToWorst(player, moment) }
                                )
                            }
                        }
                    }
                    .listRowInsets(EdgeInsets())
                } else {
                    ProgressView()
                        .frame(height: 260)
                        .listRowInsets(EdgeInsets())
                }
            }

            // What to work on: up to three plain-language cues about the
            // longest hold (chainlink #49), one line each. Shown with the
            // analysis, like everything else of its; an empty list (no
            // hold, unmeasurable clip) shows no section at all.
            if diagramAnalysis != nil, !cues.isEmpty {
                Section("What to work on") {
                    ForEach(Array(cues.enumerated()), id: \.offset) { index, cue in
                        Text("\(index + 1). \(cue.text)")
                            .font(.subheadline)
                    }
                }
            }

            Section("Recording") {
                LabeledContent("Recorded", value: SessionFormatter.dateTime(session.recordedAt))
                LabeledContent("Hold", value: session.holdType.displayName)
                LabeledContent("Duration", value: VideoInfoFormatter.duration(session.durationS))
                LabeledContent(
                    "Frame",
                    value: VideoInfoFormatter.pixelSize(width: session.width, height: session.height)
                )
                if session.analysisNote != nil {
                    // Analysed but unmeasurable (no body length): say so
                    // rather than showing a score that is not there or a
                    // hold count of 0, which would read as "did not hold".
                    Text("Couldn't measure")
                        .foregroundStyle(.secondary)
                } else if let score = session.clipScore {
                    LabeledContent("Score", value: SessionFormatter.score(score))
                } else if session.analyzedAt == nil {
                    Text("Not analysed yet")
                        .foregroundStyle(.secondary)
                } else {
                    // Analysed but scoreless (no reference in the build, or
                    // no hold that could be scored): analysed, no number —
                    // "Not analysed yet" would be wrong.
                    Text("No score yet")
                        .foregroundStyle(.secondary)
                }
            }

            Section("Analysis") {
                // What the last run *saved* (chainlink #51's columns), while
                // the panel has nothing of its own to show: the panel starts
                // from `.idle` on every visit, so these rows are how a
                // re-opened session shows the numbers the run reported.
                if service.state == .idle, session.analyzedAt != nil {
                    if session.analysisNote != nil {
                        // The run could not measure this take: no holds to
                        // list, and a "0" would read as "did not hold".
                        Text("Couldn't measure")
                            .foregroundStyle(.secondary)
                    } else {
                        if let holds = session.holdCount {
                            LabeledContent("Holds", value: "\(holds)")
                        }
                        if let longest = session.longestHoldS {
                            LabeledContent(
                                "Longest hold",
                                value: String(format: "%.1f s", longest)
                            )
                        }
                    }
                }
                // The first end-to-end run in the app (chainlink #47):
                // frames -> Apple Vision -> Analyzer -> this row.
                AnalysisPanel(
                    service: service,
                    startLabel: session.analyzedAt == nil ? "Analyse" : "Analyse again",
                    onStart: analyse
                )
            }

            // The annotated video (chainlink #50): the picture with the
            // diagram drawn on every frame and the heat strip along the
            // bottom, written to the app's Caches — out of it only through
            // the share sheet, and deleted when this screen goes away.
            // Shown exactly when there is an analysis to draw (the same
            // condition as the overlay).
            if diagramAnalysis != nil {
                Section("Export") {
                    switch exporter.state {
                    case .idle:
                        Button("Export video") { exportVideo() }
                    case .running(let progress):
                        ProgressView(value: progress)
                        Button("Cancel", role: .cancel) { exporter.cancel() }
                    case .finished(let url):
                        ShareLink(item: url) {
                            Label("Share video", systemImage: "square.and.arrow.up")
                        }
                        Button("Export again") { exportVideo() }
                    case .failed(let message):
                        Text(message)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        Button("Try again") { exportVideo() }
                    }
                }
            }

            Section {
                Button("Delete recording", role: .destructive) {
                    confirmingDelete = true
                }
            }
        }
        .navigationTitle(SessionFormatter.dateTime(session.recordedAt))
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            // One player per screen, for this screen's file — and one time
            // observer over it, feeding the diagram's clock at the clip's
            // own 30 fps rather than at the screen's refresh rate.
            if player == nil {
                player = AVPlayer(url: store.movieURL(for: session))
            }
            if let player, timeObserver == nil {
                let clock = self.clock
                timeObserver = player.addPeriodicTimeObserver(
                    forInterval: CMTime(value: 1, timescale: 30), queue: .main
                ) { time in
                    // Installed on the main queue, so this *is* the main
                    // actor — `PlaybackClock` is `@Observable`, which is
                    // what repaints the overlay from here.
                    MainActor.assumeIsolated {
                        clock.tick(time.seconds)
                    }
                }
            }
            // An already-analysed take: the pose cache has the frames, the
            // pipeline over them is milliseconds, and the overlay is on.
            loadDiagram()
        }
        .onDisappear {
            // Leaving the screen takes the sound with it, the analysis with
            // it (a run nobody is watching must not keep the CPU hot — and
            // Delete may be what sent this screen away), the time observer
            // with it, and the "stay awake" with it. The annotated export
            // (chainlink #50) goes too: it is a cache of *this* screen, so
            // a run in flight is cancelled (the export deletes its own
            // partial file) and the finished movie is deleted here — it is
            // always one button away, remade on demand.
            player?.pause()
            isPlaying = false
            if let timeObserver, let player {
                player.removeTimeObserver(timeObserver)
            }
            timeObserver = nil
            service.cancel()
            let exported = exporter.finishedURL
            exporter.cancel()
            if let exported {
                try? FileManager.default.removeItem(at: exported)
            }
            UIApplication.shared.isIdleTimerDisabled = false
        }
        .onChange(of: isAnalysing) { _, analysing in
            // The run takes seconds and the screen dims in ten: keep it
            // awake while it runs, and only while it runs — restored the
            // moment it finishes, fails or is cancelled.
            UIApplication.shared.isIdleTimerDisabled = analysing
        }
        .onChange(of: service.state) { _, state in
            // A finished run has just written its pose cache (chainlink
            // #48): re-read it so the overlay appears without leaving the
            // screen. Forced, because a diagram loaded earlier is exactly
            // what a re-run must replace.
            if case .finished = state {
                loadDiagram(force: true)
            }
        }
        .confirmationDialog(
            "Delete this recording? The video is removed from the phone.",
            isPresented: $confirmingDelete,
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) { delete() }
            Button("Cancel", role: .cancel) {}
        }
        .alert(
            "Could not delete the recording",
            isPresented: Binding(
                get: { deletionError != nil },
                set: { if !$0 { deletionError = nil } }
            )
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(deletionError ?? "")
        }
    }

    // MARK: - Playback controls

    /// Play/pause, the scrubber, and the overlay's switch — kept simple on
    /// purpose: the picture and the diagram are the show, these only move
    /// the moment in time the diagram is drawn for.
    private func playbackControls(_ player: AVPlayer) -> some View {
        HStack(spacing: 12) {
            Button {
                togglePlayback(player)
            } label: {
                Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                    .frame(width: 24)
            }
            .accessibilityLabel(isPlaying ? "Pause" : "Play")

            Slider(value: scrubBinding, in: scrubRange, onEditingChanged: { editing in
                scrubbing = editing
                if editing {
                    // Dragging seeks where you drop it, so the picture
                    // must hold still for the drag.
                    player.pause()
                    isPlaying = false
                    scrubSeconds = clock.seconds
                } else {
                    seek(player, to: scrubSeconds)
                }
            })

            if diagramAnalysis != nil {
                Button {
                    showDiagram.toggle()
                } label: {
                    Label("Diagram", systemImage: showDiagram ? "eye" : "eye.slash")
                }
                .accessibilityLabel(showDiagram ? "Hide diagram" : "Show diagram")
            }
        }
        .buttonStyle(.borderless)
        .frame(maxWidth: .infinity)
    }

    /// What the scrubber shows and writes: its own value while dragged,
    /// the clock's while the movie plays.
    private var scrubBinding: Binding<Double> {
        Binding(
            get: { scrubbing ? scrubSeconds : clock.seconds },
            set: { scrubSeconds = $0 }
        )
    }

    /// The scrubber's range: the movie's length, never `NaN` (a movie with
    /// unreadable metadata still gets a slider that answers).
    private var scrubRange: ClosedRange<Double> {
        let duration = session.durationS
        return 0...(duration.isFinite && duration > 0 ? duration : 0.1)
    }

    /// Play or pause — and start over when the movie has run out, so the
    /// play button is never a button that does nothing.
    private func togglePlayback(_ player: AVPlayer) {
        if isPlaying {
            player.pause()
            isPlaying = false
            return
        }
        let duration = session.durationS
        if duration.isFinite, duration > 0, clock.seconds >= duration - 0.05 {
            seek(player, to: 0)
        }
        player.play()
        isPlaying = true
    }

    /// Jump the movie (and the diagram with it) to `seconds`.
    private func seek(_ player: AVPlayer, to seconds: Double) {
        let duration = session.durationS
        let upper = duration.isFinite && duration > 0 ? duration : .greatestFiniteMagnitude
        let target = Swift.min(Swift.max(0, seconds.isFinite ? seconds : 0), upper)
        player.seek(
            to: CMTime(seconds: target, preferredTimescale: 600),
            toleranceBefore: .zero, toleranceAfter: .zero
        )
        clock.tick(target)
    }

    // MARK: - The diagram

    /// Loads the overlay's analysis from the take's pose cache — on appear,
    /// and (forced) after a run has just written one. Nothing to do when
    /// the take was never analysed, or when there is no cache yet (a take
    /// analysed before this feature): the overlay then appears after
    /// "Analyse", which is what writes it.
    private func loadDiagram(force: Bool = false) {
        guard force || diagramAnalysis == nil else { return }
        guard session.analyzedAt != nil else { return }
        guard let frames = PoseCache.read(for: store.movieURL(for: session)) else { return }
        // The same reference the run itself used: with one, the colours
        // compare against your own good holds; without one, against the
        // built-in thresholds (`FeatureTolerances`). Re-analysing here (it
        // is milliseconds) rather than caching the answer is what keeps the
        // overlay from going stale when a new reference lands.
        let reference = ReferenceLoader.load(for: session.holdType)
        let analysis = Analyzer.analyze(
            frames, reference: reference, config: PoseBackend.preferred.postProcessConfig)
        diagramReference = reference
        diagramAnalysis = analysis
        showDiagram = true

        // The summary under the player (chainlink #49): the strip's bins,
        // the worst moment with the diagram drawn over its thumbnail, and
        // the coaching cues — all computed *here*, once per analysis (and
        // so again after "Analyse again"), never per playback frame.
        heatBins = SessionSummary.heatStrip(analysis: analysis, reference: reference)
        let moment = SessionSummary.worstMoment(analysis: analysis, reference: reference)
        worstMoment = moment
        worstMomentDiagram = moment.map {
            StressDiagram.frame(
                $0.frameIndex, analysis: analysis, reference: reference,
                height: Double(session.height))
        }
        cues = CoachingCues.cues(analysis: analysis, reference: reference)
    }

    /// The worst-moment card's tap: seek there and **pause** — a jump to
    /// the moment you asked about should hold still while you look at it.
    private func seekToWorst(_ player: AVPlayer, _ moment: WorstMoment) {
        player.pause()
        isPlaying = false
        seek(player, to: Double(moment.tMs) / 1000.0)
    }

    private func delete() {
        // A run in flight must not write its result to a row that is about
        // to go: cancelling bumps the generation, so its save is refused
        // no matter when it wakes up. The export goes for the same reason —
        // `store.delete` takes the session's cached export with it, and a
        // writer still going would only put a file back.
        service.cancel()
        exporter.cancel()
        do {
            try store.delete(session)
            dismiss()
        } catch {
            // The row and the video are still there (delete only removes
            // the record once the files are gone), so say what went wrong.
            deletionError = error.localizedDescription
        }
    }

    /// The "Export video" button: this row's movie and the analysis on
    /// screen, into the app's Caches as `<basename>-annotated.mp4`
    /// (`SessionStore`'s own spelling, so `delete` removes exactly this
    /// file). Anything a previous export left behind is deleted first, so
    /// two exports never share a file.
    private func exportVideo() {
        guard let analysis = diagramAnalysis else { return }
        let output = store.annotatedExportURL(for: session)
        try? FileManager.default.removeItem(at: output)
        let reference = diagramReference
        let movie = store.movieURL(for: session)
        Task {
            await exporter.export(
                movie: movie, analysis: analysis, reference: reference, to: output)
        }
    }

    /// The "Analyse" button: this row's movie, this row's hold, saved back
    /// to this row. `AnalysisService` stops anything already running, so
    /// every tap means one run.
    private func analyse() {
        Task {
            await service.analyse(
                movie: store.movieURL(for: session),
                holdType: session.holdType,
                session: session,
                store: store
            )
        }
    }
}
