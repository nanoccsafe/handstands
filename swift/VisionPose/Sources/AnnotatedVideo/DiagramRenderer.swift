import CoreGraphics
import Foundation
import HandstandCore

// --------------------------------------------------------------------------- #
// Drawing one frame of the export (chainlink #50).
//
// The screen draws the same model in SwiftUI (`StressDiagramOverlay`) — this
// is the CoreGraphics twin: same order (stack line, bones, joints, centre of
// mass), same palette and sizes through `DiagramStyle`, no rules of its own.
// Everything the model (`DiagramFrame`) does not say — how thick, which
// colour, how big — comes from the style, so the two drawings cannot drift.
//
// The coordinate contract: the context handed to `draw` is in **display
// pixels with the origin top-left** (display `y` grows downwards, like the
// view's), and the *caller* has already flipped its CTM. Sizes are multiplied
// by `scale` = `videoHeight / 1000` — a 1920-high export draws 1.92× the
// on-screen point sizes, a 320-high test clip 0.32×.
// --------------------------------------------------------------------------- #

/// Draws `DiagramFrame`s and the timeline heat strip into a `CGContext`.
public enum DiagramRenderer {
    /// Draws one frame of the diagram: the dashed stack line, the bones, the
    /// joints (sized by their severity), and the centre of mass with its
    /// projection arrow — in that order, exactly as the overlay layers them.
    ///
    /// - Parameters:
    ///   - frame: what to draw, in display pixels.
    ///   - ctx: a context in display pixels, origin **top-left** (the
    ///     caller flips its CTM before calling, and this function does not
    ///     touch it beyond one save/restore pair).
    ///   - scale: `videoHeight / 1000`, the multiplier from the overlay's
    ///     point sizes to this video's pixels.
    ///   - style: the palette and sizes; `.standard` is the overlay's.
    public static func draw(
        _ frame: DiagramFrame, in ctx: CGContext, scale: Double,
        style: DiagramStyle = .standard
    ) {
        let scale = (scale.isFinite && scale > 0) ? scale : 1
        ctx.saveGState()
        defer { ctx.restoreGState() }

        // 1. The stack line — the column through the hands, dashed, so it
        //    reads as a reference line rather than a bone.
        if let line = frame.stackLine {
            ctx.saveGState()
            ctx.setLineDash(
                phase: 0, lengths: style.stackDash.map { $0 * scale })
            ctx.setStrokeColor(style.colour(for: frame.stackBand).cgColor)
            ctx.setLineWidth(style.stackLineWidth * scale)
            ctx.move(to: CGPoint(x: line.top.x, y: line.top.y))
            ctx.addLine(to: CGPoint(x: line.bottom.x, y: line.bottom.y))
            ctx.strokePath()
            ctx.restoreGState()
        }

        // 2. The bones: 4 pt lines, each its worse joint's band. The frame
        //    names bones by joint, so the positions come from the joints it
        //    drew — the overlay's own dictionary.
        let positions = Dictionary(
            uniqueKeysWithValues: frame.joints.map { ($0.joint, $0.point) })
        for bone in frame.bones {
            guard let start = positions[bone.from], let end = positions[bone.to] else {
                continue
            }
            ctx.setStrokeColor(style.colour(for: bone.band).cgColor)
            ctx.setLineWidth(style.boneWidth * scale)
            ctx.move(to: CGPoint(x: start.x, y: start.y))
            ctx.addLine(to: CGPoint(x: end.x, y: end.y))
            ctx.strokePath()
        }

        // 3. The ideal skeleton: `frame.ideal` is always `nil` until chainlink
        //    #28 provides the user's reference skeleton — out of scope here,
        //    and deliberately not drawn rather than drawn as something else.

        // 4. The joints: circles sized by severity — `4 + 8 × severity`
        //    points, a flat 4 without one — so "how bad" is readable at a
        //    glance as well as by colour.
        for joint in frame.joints {
            let radius = style.jointRadius(severity: joint.severity) * scale
            let centre = CGPoint(x: joint.point.x, y: joint.point.y)
            ctx.setFillColor(style.colour(for: joint.band).cgColor)
            ctx.fillEllipse(
                in: CGRect(
                    x: centre.x - radius, y: centre.y - radius,
                    width: radius * 2, height: radius * 2)
            )
        }

        // 5. The centre of mass: a dot where it is, and an arrow from the
        //    wrist midpoint to where it projects onto the hand line — the
        //    distance the balance is being held over, in its zone's colour.
        let zone = style.colour(for: frame.balanceZone)
        if let com = frame.com {
            let radius = style.comRadius * scale
            ctx.setFillColor(zone.cgColor)
            ctx.fillEllipse(
                in: CGRect(
                    x: com.x - radius, y: com.y - radius,
                    width: radius * 2, height: radius * 2)
            )
        }
        if let floor = frame.comFloor {
            // The wrist midpoint is where the stack line runs, at the hand
            // line's height (which is exactly `floor.y`).
            let wrist = CGPoint(
                x: frame.stackLine?.top.x ?? floor.x, y: floor.y)
            drawArrow(
                from: wrist,
                to: CGPoint(x: floor.x, y: floor.y),
                in: ctx, colour: zone, style: style, scale: scale)
        }
    }

    /// Draws the timeline heat strip inside `rect`: one coloured band per
    /// `HeatBin`, tiled across the clip's span, and a white playhead at
    /// `playheadMs`. An **empty** bin list is a *neutral* strip — a solid
    /// grey bar — which is what an unusable analysis exports (chainlink #50
    /// says the video still goes out, with the strip saying nothing).
    ///
    /// The x↔time mapping is `HeatStripGeometry`'s, spelled again here so
    /// this target stays a library of its own: 0 is the first bin's start, 1
    /// its last bin's end, and a time outside the clip clamps into it.
    ///
    /// - Parameters:
    ///   - bins: `SessionSummary.heatStrip`'s answer, in time order.
    ///   - playheadMs: this frame's time on the clip's clock.
    ///   - rect: where the strip sits, display pixels (origin top-left).
    ///   - ctx: the same context `draw(_:…)` takes.
    ///   - style: the palette; `.standard` is the overlay's.
    public static func drawHeatStrip(
        _ bins: [HeatBin], playheadMs: Int, in rect: CGRect, ctx: CGContext,
        style: DiagramStyle = .standard
    ) {
        guard rect.width > 0, rect.height > 0 else { return }
        ctx.saveGState()
        defer { ctx.restoreGState() }

        guard let first = bins.first?.startMs, let last = bins.last?.endMs else {
            // Nothing analysed (or nothing usable): the neutral strip — grey
            // across the bottom, saying "no form to show here" rather than
            // showing nothing at all.
            ctx.setFillColor(style.colour(for: .neutral).cgColor)
            ctx.fill(rect)
            return
        }

        for bin in bins {
            let x0 = rect.minX + fraction(bin.startMs, first: first, last: last) * rect.width
            let x1 = rect.minX + fraction(bin.endMs, first: first, last: last) * rect.width
            let width = Swift.max(x1 - x0, 0)
            guard width > 0 else { continue }
            ctx.setFillColor(style.colour(for: bin.band).cgColor)
            ctx.fill(CGRect(x: x0, y: rect.minY, width: width, height: rect.height))
        }

        // The playhead: a thin white line at this frame's time, the same one
        // the screen's strip shows (slightly thicker, because it is pixels
        // here rather than points).
        let x = rect.minX + fraction(playheadMs, first: first, last: last) * rect.width
        let width = Swift.max(style.playheadWidth, 1)
        ctx.setFillColor(DiagramStyle.Colour.white.cgColor)
        ctx.fill(
            CGRect(x: x - width / 2, y: rect.minY, width: width, height: rect.height))
    }

    // MARK: - Private

    /// Where a time sits on the strip, as a fraction of its width — the
    /// screen's `HeatStripGeometry.fraction(tMs:firstMs:lastMs:)`: 0 at the
    /// first bin's start, 1 at the last bin's end, clamped into the clip,
    /// and 0 for a clip with no span.
    private static func fraction(_ tMs: Int, first: Int, last: Int) -> Double {
        guard last > first else { return 0 }
        let clamped = Swift.min(Swift.max(Double(tMs), Double(first)), Double(last))
        return (clamped - Double(first)) / Double(last - first)
    }

    /// A line from `start` to `end` with a small arrowhead at `end` — the
    /// centre of mass's projection arrow, the overlay's own shape.
    private static func drawArrow(
        from start: CGPoint, to end: CGPoint, in ctx: CGContext,
        colour: DiagramStyle.Colour, style: DiagramStyle, scale: Double
    ) {
        guard start != end else { return }
        ctx.setStrokeColor(colour.cgColor)
        ctx.setLineWidth(style.arrowWidth * scale)
        ctx.move(to: start)
        ctx.addLine(to: end)
        ctx.strokePath()

        let angle = atan2(end.y - start.y, end.x - start.x)
        let size = style.arrowHeadSize * scale
        for spread in [0.4, -0.4] {
            ctx.move(to: end)
            ctx.addLine(
                to: CGPoint(
                    x: end.x - size * cos(angle - spread),
                    y: end.y - size * sin(angle - spread)
                ))
        }
        ctx.strokePath()
    }
}
