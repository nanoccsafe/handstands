import HandstandCore
import SwiftUI

// --------------------------------------------------------------------------- #
// The stress diagram on screen (#48): one `DiagramFrame`, drawn.
//
// The model (`HandstandCore.StressDiagram`) says *what* to draw and the
// geometry (`OverlayGeometry`) says *where*; this file only turns those two
// answers into strokes. It carries no severity rules of its own — every
// colour here comes from the frame's bands, so what the tests pinned in
// HandstandCore is exactly what the phone shows.
//
// The drawing is a `Canvas`, layered straight over `PlayerLayerView`'s
// picture: bones first (4 pt, coloured by band), then the joints as circles
// sized by their own severity, then the stack line, then the centre of mass.
// --------------------------------------------------------------------------- #

/// The overlay itself: the picture goes underneath, this draws on top.
struct StressDiagramOverlay: View {
    /// The frame to draw — one playback moment's worth of diagram.
    let diagram: DiagramFrame
    /// The video's display size in pixels (`session.width × session.height`),
    /// which is the space `diagram`'s points are written in.
    let videoSize: CGSize

    var body: some View {
        Canvas { context, size in
            let rect = OverlayGeometry.videoRect(videoSize: videoSize, in: size)
            // No usable video size (or a view still laying out): nothing to
            // draw on, and NaN points would draw on it anyway.
            guard rect.width > 0, rect.height > 0 else { return }

            func point(_ point: Point2) -> CGPoint {
                OverlayGeometry.viewPoint(point, videoSize: videoSize, rect: rect)
            }

            // 1. The stack line — the column through the hands, dashed, so
            //    it reads as a reference line rather than a bone.
            if let line = diagram.stackLine {
                var path = Path()
                path.move(to: point(line.top))
                path.addLine(to: point(line.bottom))
                context.stroke(
                    path,
                    with: .color(Self.colour(diagram.stackBand)),
                    style: StrokeStyle(lineWidth: 4, dash: [7, 5])
                )
            }

            // 2. The bones: 4 pt lines, each its worse joint's band. The
            //    frame names bones by joint, so the positions come from the
            //    joints it drew.
            let positions = Dictionary(
                uniqueKeysWithValues: diagram.joints.map { ($0.joint, $0.point) })
            for bone in diagram.bones {
                guard let start = positions[bone.from], let end = positions[bone.to] else {
                    continue
                }
                var path = Path()
                path.move(to: point(start))
                path.addLine(to: point(end))
                context.stroke(path, with: .color(Self.colour(bone.band)), lineWidth: 4)
            }

            // 3. The ideal skeleton: `diagram.ideal` is always `nil` until
            //    chainlink #28 provides the user's reference skeleton — when
            //    it does, its ghost goes here, drawn between the bones and
            //    the joints. Nothing else in this file changes.

            // 4. The joints: circles sized by severity — 4 + 8 × severity
            //    points, and a flat 4 for the neutral ones — so "how bad"
            //    is readable at a glance as well as by colour.
            for joint in diagram.joints {
                let radius = joint.severity.map { 4.0 + 8.0 * Swift.min(Swift.max($0, 0), 1) }
                    ?? 4.0
                let centre = point(joint.point)
                let dot = Path(
                    ellipseIn: CGRect(
                        x: centre.x - radius, y: centre.y - radius,
                        width: radius * 2, height: radius * 2)
                )
                context.fill(dot, with: .color(Self.colour(joint.band)))
            }

            // 5. The centre of mass: a dot where it is, and an arrow from
            //    the wrist midpoint to where it projects onto the hand line
            //    — the distance the balance is being held over, in the
            //    colour of its balance zone.
            let zone = Self.colour(diagram.balanceZone)
            if let com = diagram.com {
                let centre = point(com)
                let dot = Path(
                    ellipseIn: CGRect(x: centre.x - 5, y: centre.y - 5, width: 10, height: 10))
                context.fill(dot, with: .color(zone))
            }
            if let floor = diagram.comFloor {
                // The wrist midpoint is where the stack line runs, at the
                // hand line's height (which is exactly `comFloor`'s y).
                let wrist = Point2(x: diagram.stackLine?.top.x ?? floor.x, y: floor.y)
                Self.drawArrow(
                    from: point(wrist), to: point(floor), context: context, colour: zone)
            }
        }
        .allowsHitTesting(false)
    }

    // MARK: - The palette

    /// A band as the colour it is drawn in: grey when nothing is measured,
    /// then green / amber / red by severity.
    static func colour(_ band: SeverityBand) -> Color {
        switch band {
        case .neutral: Color.gray
        case .ok: Color.green
        case .warn: Color.orange
        case .bad: Color.red
        }
    }

    /// A balance zone as its colour: inside the base of support is good
    /// (green), in front of the fingertips is over (red), behind the heel is
    /// under (amber), and no CoM at all is grey.
    static func colour(_ zone: BalanceZone?) -> Color {
        switch zone {
        case .ok: Color.green
        case .over: Color.red
        case .under: Color.orange
        case nil: Color.gray
        }
    }

    // MARK: - Private

    /// A line from `start` to `end` with a small arrowhead at `end` — the
    /// CoM's projection arrow.
    private static func drawArrow(
        from start: CGPoint, to end: CGPoint, context: GraphicsContext, colour: Color
    ) {
        guard start != end else { return }
        var path = Path()
        path.move(to: start)
        path.addLine(to: end)
        context.stroke(path, with: .color(colour), lineWidth: 3)

        let angle = atan2(end.y - start.y, end.x - start.x)
        let size: CGFloat = 9
        var head = Path()
        head.move(to: end)
        head.addLine(
            to: CGPoint(
                x: end.x - size * cos(angle - 0.4), y: end.y - size * sin(angle - 0.4)))
        head.move(to: end)
        head.addLine(
            to: CGPoint(
                x: end.x - size * cos(angle + 0.4), y: end.y - size * sin(angle + 0.4)))
        context.stroke(head, with: .color(colour), lineWidth: 3)
    }
}

/// The legend under the player: what the colours mean, and where they come
/// from — the user's own reference when the build has one (#28), the
/// built-in form thresholds until then.
struct StressDiagramLegend: View {
    /// Is the colour scale comparing against a scoring reference?
    let hasReference: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 12) {
                swatch(StressDiagramOverlay.colour(SeverityBand.ok), "on form")
                swatch(StressDiagramOverlay.colour(SeverityBand.warn), "off")
                swatch(StressDiagramOverlay.colour(SeverityBand.bad), "far off")
                swatch(StressDiagramOverlay.colour(SeverityBand.neutral), "not held")
            }
            Text(note)
                .foregroundStyle(.secondary)
        }
        .font(.caption2)
        .frame(maxWidth: .infinity, alignment: .center)
    }

    /// The note line, spelled exactly as chainlink #48 asks for it.
    private var note: String {
        hasReference
            ? "Colours compare with your reference"
            : "Colours use built-in form thresholds (no reference yet)"
    }

    private func swatch(_ colour: Color, _ label: String) -> some View {
        HStack(spacing: 4) {
            Circle()
                .fill(colour)
                .frame(width: 8, height: 8)
            Text(label)
        }
    }
}
