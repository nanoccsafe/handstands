import AVFoundation
import HandstandCore
import SwiftUI
import UIKit

// --------------------------------------------------------------------------- #
// The "worst moment" card (chainlink #49): a thumbnail of the worst frame
// of the hold with the stress diagram drawn over it, when it was, what was
// off — and a tap that seeks the player there.
//
// The frame comes from `AVAssetImageGenerator` at the moment's exact
// `t_ms` (zero tolerance either side, so the picture is the frame the
// analysis judged, not a neighbouring one), drawn once per moment
// (`.task(id:)`) — playing the movie changes nothing about this card. The
// overlay is the *existing* `StressDiagramOverlay`: it aspect-fits into
// whatever size it is given (`OverlayGeometry`), so it lands on the
// thumbnail exactly where it lands on the player.
// --------------------------------------------------------------------------- #

/// The card: thumbnail on the left, "Worst moment · 0:04.2" and the
/// features in plain words on the right; tapping anywhere seeks.
struct WorstMomentCard: View {
    /// The movie the thumbnail is read from.
    let movieURL: URL
    /// The worst frame to show and seek to.
    let moment: WorstMoment
    /// That frame's diagram, drawn over the thumbnail.
    let diagram: DiagramFrame
    /// The video's display size in pixels — the thumbnail's aspect ratio
    /// and the overlay's coordinate space.
    let videoSize: CGSize
    /// Seek the player to the worst moment (the screen pauses it).
    let onTap: () -> Void

    /// The decoded frame; `nil` until it arrives, and afterwards if the
    /// movie could not be read (the card still shows the diagram).
    @State private var thumbnail: UIImage?

    /// The thumbnail's width in points; its height follows the video's
    /// aspect ratio so the overlay lands on the picture exactly.
    private static let thumbnailWidth: CGFloat = 96

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 12) {
                ZStack {
                    if let thumbnail {
                        Image(uiImage: thumbnail)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                    } else {
                        Color.black.opacity(0.5)
                    }
                    StressDiagramOverlay(diagram: diagram, videoSize: videoSize)
                }
                .frame(width: Self.thumbnailWidth, height: thumbnailHeight)
                .background(Color.black)
                .clipShape(RoundedRectangle(cornerRadius: 6))

                VStack(alignment: .leading, spacing: 3) {
                    Text("Worst moment · \(SessionFormatter.position(moment.tMs))")
                        .font(.subheadline.weight(.semibold))
                    if !moment.features.isEmpty {
                        Text(
                            moment.features
                                .map(FaultLabel.text(for:))
                                .joined(separator: ", ")
                        )
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    }
                }
                Spacer(minLength: 0)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Worst moment at \(SessionFormatter.position(moment.tMs))")
        .task(id: moment.tMs) {
            thumbnail = await Self.frameImage(url: movieURL, tMs: moment.tMs)
        }
    }

    // MARK: - Private

    /// The thumbnail's height: the video's own aspect, so a portrait take
    /// gets a portrait thumb (and the overlay's aspect-fit fills it).
    private var thumbnailHeight: CGFloat {
        guard videoSize.width > 0, videoSize.height > 0 else {
            return Self.thumbnailWidth
        }
        return Self.thumbnailWidth * videoSize.height / videoSize.width
    }

    /// The movie's frame at `tMs`, exactly there (zero time tolerance).
    private static func frameImage(url: URL, tMs: Int) async -> UIImage? {
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        let time = CMTime(value: CMTimeValue(Swift.max(tMs, 0)), timescale: 1000)
        guard let generated = try? await generator.image(at: time) else { return nil }
        return UIImage(cgImage: generated.image)
    }
}
