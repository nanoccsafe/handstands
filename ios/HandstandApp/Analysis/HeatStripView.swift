import HandstandCore
import SwiftUI

// --------------------------------------------------------------------------- #
// The timeline heat strip (chainlink #49): the clip's timeline under the
// player, one coloured bin per slice of it, a playhead, and a tap or drag
// that seeks.
//
// The bins come from `SessionSummary.heatStrip`, already computed when the
// analysis is loaded — nothing here is recomputed per playback frame, only
// the playhead moves (`PlaybackClock`). The colours are the diagram's own
// (`StressDiagramOverlay.colour`), so the strip and the overlay over the
// video cannot drift apart, and the x↔time maths lives in
// `HeatStripGeometry`, where it is unit-tested without a view.
// --------------------------------------------------------------------------- #

/// The heat strip: a horizontal bar of the clip's bins, the playhead over
/// it, and the caption below.
struct HeatStripView: View {
    /// The bins, in time order — `SessionSummary.heatStrip`'s answer.
    let bins: [HeatBin]
    /// The clip's first and last `t_ms` — the span the strip maps over.
    let firstMs: Int
    let lastMs: Int
    /// Where playback is right now, in the clip's milliseconds: the only
    /// thing about this view that changes during playback.
    let playheadMs: Int
    /// Seek the player to a moment of the clip (milliseconds on the
    /// clip's clock).
    let onSeek: (Int) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            GeometryReader { geometry in
                let width = geometry.size.width
                Canvas { context, size in
                    drawBins(context: context, width: size.width, height: size.height)
                    drawPlayhead(context: context, width: size.width, height: size.height)
                }
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { value in
                            guard width > 0 else { return }
                            let fraction = value.location.x / width
                            onSeek(
                                HeatStripGeometry.tMs(
                                    fraction: fraction, firstMs: firstMs, lastMs: lastMs))
                        }
                )
            }
            .frame(height: 16)

            Text("Form over time (holds coloured)")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Drawing

    /// Every bin, from its own timestamps (not from its index), so the
    /// colours line up exactly with the playhead's mapping.
    private func drawBins(context: GraphicsContext, width: CGFloat, height: CGFloat) {
        guard width > 0 else { return }
        for bin in bins {
            let start = HeatStripGeometry.fraction(
                tMs: bin.startMs, firstMs: firstMs, lastMs: lastMs)
            let end = HeatStripGeometry.fraction(tMs: bin.endMs, firstMs: firstMs, lastMs: lastMs)
            let x0 = start * width
            let rect = CGRect(x: x0, y: 0, width: Swift.max((end - start) * width, 0), height: height)
            guard rect.width > 0 else { continue }
            context.fill(Path(rect), with: .color(StressDiagramOverlay.colour(bin.band)))
        }
    }

    /// The playhead: a thin white line at the clock's current time.
    private func drawPlayhead(context: GraphicsContext, width: CGFloat, height: CGFloat) {
        guard width > 0 else { return }
        let fraction = HeatStripGeometry.fraction(
            tMs: playheadMs, firstMs: firstMs, lastMs: lastMs)
        let x = fraction * width
        let rect = CGRect(x: Swift.max(x - 0.75, 0), y: 0, width: 1.5, height: height)
        context.fill(Path(rect), with: .color(.white))
    }
}
