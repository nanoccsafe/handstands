import PhotosUI
import SwiftUI

/// "Analyse a video": pick a video, read its duration and pixel size, show
/// them. That is all this version does — the pose analysis itself is
/// chainlink #47 — but the pick → read → display path is the real one the
/// analysis will hang off.
struct AnalyseVideoView: View {
    @State private var picked: PhotosPickerItem?
    @State private var info: VideoInfo?
    @State private var message: String?
    @State private var reading = false

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
    }

    /// Loads the picked item as a movie file and reads it with AVFoundation.
    @MainActor
    private func readVideo(_ item: PhotosPickerItem?) async {
        info = nil
        message = nil
        guard let item else { return }
        reading = true
        defer { reading = false }
        do {
            guard let movie = try await item.loadTransferable(type: MovieFile.self) else {
                message = "That item is not a video file."
                return
            }
            defer { try? FileManager.default.removeItem(at: movie.url) }
            info = try await VideoInfoReader.read(url: movie.url)
        } catch {
            message = "Could not read that video: \(error.localizedDescription)"
        }
    }
}
