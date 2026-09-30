import PhotosUI
import SwiftUI
import UIKit

/// "Analyse a video": pick a video, read its duration and pixel size, show
/// them — and analyse it (chainlink #47) with the same run a session gets:
/// frames → Apple Vision → `Analyzer` → the same result panel.
///
/// A video picked from Photos is not a recording of this app, so nothing
/// here touches History: `analyse` is called with no session and no store,
/// and the result lives on this screen only. The picked file is a private
/// copy in tmp (`MovieFile`), kept for as long as this screen may analyse
/// it and removed when a new pick replaces it or the screen goes.
struct AnalyseVideoView: View {
    @State private var picked: PhotosPickerItem?
    @State private var info: VideoInfo?
    /// Where the picked copy lives while it may still be analysed.
    @State private var movieURL: URL?
    @State private var message: String?
    @State private var reading = false
    @State private var service = AnalysisService()

    /// Is a run in flight — the state the screen has to stay awake for.
    private var isAnalysing: Bool {
        if case .running = service.state { return true }
        return false
    }

    var body: some View {
        VStack(spacing: 16) {
            PhotosPicker(selection: $picked, matching: .videos) {
                Label("Choose a video", systemImage: "photo.on.rectangle")
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)

            if reading {
                ProgressView("Reading the video…")
            }

            if let info {
                LabeledContent(
                    "Duration",
                    value: VideoInfoFormatter.duration(info.duration)
                )
                LabeledContent(
                    "Frame",
                    value: VideoInfoFormatter.pixelSize(width: info.width, height: info.height)
                )
            }

            if let movieURL {
                // The same panel the session screen shows: the button, the
                // progress with Cancel, and the results.
                AnalysisPanel(service: service, startLabel: "Analyse") {
                    analyse(movie: movieURL)
                }
            }

            if let message {
                Text(message)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }

            Spacer()
        }
        .padding(.horizontal, 24)
        .navigationTitle("Analyse a video")
        .onChange(of: picked) { _, item in
            Task {
                await readVideo(item)
            }
        }
        .onChange(of: isAnalysing) { _, analysing in
            // Same as the session screen: awake while it runs, only then.
            UIApplication.shared.isIdleTimerDisabled = analysing
        }
        .onDisappear {
            UIApplication.shared.isIdleTimerDisabled = false
            service.cancel()
            removeMovie()
        }
    }

    // MARK: - The picked video

    /// Loads the picked item as a movie file and reads it with AVFoundation.
    @MainActor
    private func readVideo(_ item: PhotosPickerItem?) async {
        info = nil
        message = nil
        // A new pick (or none): the old run stops and its temp copy goes —
        // the file it was reading is about to be deleted.
        service.cancel()
        removeMovie()
        guard let item else { return }
        reading = true
        defer { reading = false }
        do {
            guard let movie = try await item.loadTransferable(type: MovieFile.self) else {
                message = "That item is not a video file."
                return
            }
            do {
                info = try await VideoInfoReader.read(url: movie.url)
                // Kept for the analysis; removed on the next pick or when
                // this screen goes (previously it was deleted right after
                // reading the info — there was nothing to analyse yet).
                movieURL = movie.url
            } catch {
                try? FileManager.default.removeItem(at: movie.url)
                throw error
            }
        } catch {
            message = "Could not read that video: \(error.localizedDescription)"
        }
    }

    private func removeMovie() {
        guard let movieURL else { return }
        self.movieURL = nil
        info = nil
        try? FileManager.default.removeItem(at: movieURL)
    }

    /// The "Analyse" button: the same run a session gets, with the MVP's
    /// Line hold (there is no recorded sidecar to read a hold from) and
    /// nothing to save it to — a picked video is not in History.
    private func analyse(movie: URL) {
        Task {
            await service.analyse(movie: movie, holdType: .line, session: nil, store: nil)
        }
    }
}
