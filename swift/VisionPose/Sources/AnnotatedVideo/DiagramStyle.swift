import CoreGraphics
import Foundation
import HandstandCore

// --------------------------------------------------------------------------- #
// What the annotated video draws with (chainlink #50).
//
// The screen has its palette already — `StressDiagramOverlay`'s colours and
// sizes — and the export has to look the same, so this is that palette
// spelled in sRGB components instead of SwiftUI `Color`s (which are dynamic
// system colours a `CGContext` cannot resolve). The values are the light-mode
// system colours the overlay resolves to: green `#34C759`, orange `#FF9500`,
// red `#FF3B30`, grey `#8E8E93`.
//
// Sizes keep the overlay's *point* numbers (4 pt bones, 4 + 8 × severity
// joint radii, the 7/5 dash, the 3 pt arrow with its 9 pt head) and are
// multiplied by `scale` — `videoHeight / 1000` — wherever they are drawn, so
// a 1920-high export draws 1.92× the on-screen sizes and the diagram reads
// at the same weight over the picture.
// --------------------------------------------------------------------------- #

/// The colours and sizes the annotated export draws the stress diagram, the
/// heat strip and the footer with. All values are in *points* of the
/// on-screen overlay; the renderer scales them.
public struct DiagramStyle: Sendable {
    /// One colour as sRGB components in `0...1`, built into a `CGColor` on
    /// demand (a `CGColor` is not `Sendable`, so the style stores numbers).
    public struct Colour: Sendable, Equatable {
        public var red: Double
        public var green: Double
        public var blue: Double
        public var alpha: Double

        public init(red: Double, green: Double, blue: Double, alpha: Double = 1) {
            self.red = red
            self.green = green
            self.blue = blue
            self.alpha = alpha
        }

        /// The colour in the sRGB space the export's contexts are created
        /// in — one fresh `CGColor` per call, never shared across threads.
        public var cgColor: CGColor {
            CGColor(
                colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
                components: [CGFloat(red), CGFloat(green), CGFloat(blue), CGFloat(alpha)]
            )!
        }

        /// The overlay's grey — "nobody is measuring this".
        public static let grey = Colour(red: 0.557, green: 0.557, blue: 0.576)
        /// The overlay's green — on form.
        public static let green = Colour(red: 0.204, green: 0.780, blue: 0.349)
        /// The overlay's orange — off, but not badly.
        public static let orange = Colour(red: 1.0, green: 0.584, blue: 0.0)
        /// The overlay's red — far off.
        public static let red = Colour(red: 1.0, green: 0.231, blue: 0.188)
        /// The heat strip's playhead, and the footer's text.
        public static let white = Colour(red: 1.0, green: 1.0, blue: 1.0)
    }

    /// The colour each `SeverityBand` is drawn in — the overlay's own map.
    public var bandColours: [SeverityBand: Colour] = [
        .neutral: .grey,
        .ok: .green,
        .warn: .orange,
        .bad: .red,
    ]
    /// The colour each `BalanceZone`'s centre-of-mass dot and arrow are
    /// drawn in — the overlay's own map.
    public var zoneColours: [BalanceZone: Colour] = [
        .ok: .green,
        .over: .red,
        .under: .orange,
    ]
    /// The colour when there is no centre of mass to judge (grey, like a
    /// `.neutral` band).
    public var noZoneColour: Colour = .grey
    /// Bones and the stack line are 4 pt on screen.
    public var boneWidth: Double = 4
    public var stackLineWidth: Double = 4
    /// The stack line's dash, `7/5` points on screen, so it reads as a
    /// reference line rather than a bone.
    public var stackDash: [Double] = [7, 5]
    /// Joint radius: `jointBaseRadius + jointSeverityRadius × severity`,
    /// clamped to `0...1`, and a flat base radius with no severity — the
    /// overlay's `4 + 8 × severity`.
    public var jointBaseRadius: Double = 4
    public var jointSeverityRadius: Double = 8
    /// The centre-of-mass dot's radius (5 pt) and its projection arrow's
    /// line width (3 pt) and head size (9 pt).
    public var comRadius: Double = 5
    public var arrowWidth: Double = 3
    public var arrowHeadSize: Double = 9
    /// The heat strip's height as a fraction of the video's height: 3 %.
    public var stripFraction: Double = 0.03
    /// The playhead's width **in pixels** (the strip is laid out in pixels
    /// directly, so this one is not scaled).
    public var playheadWidth: Double = 2
    /// The footer's font size and left margin, in points (scaled like the
    /// diagram's sizes).
    public var footerFontSize: Double = 12
    public var footerMargin: Double = 6

    public init() {}

    /// The overlay's palette and sizes — what every export uses unless a
    /// caller says otherwise (custom styling is out of scope, chainlink #50).
    public static let standard = DiagramStyle()

    /// A band's colour (`SeverityBand.neutral` → grey, then green / amber /
    /// red), exactly the map `StressDiagramOverlay.colour(_:)` shows.
    public func colour(for band: SeverityBand) -> Colour {
        bandColours[band] ?? .grey
    }

    /// A balance zone's colour: green inside the base of support, red over
    /// the fingers, amber behind the heel, grey with no centre of mass —
    /// `StressDiagramOverlay.colour(_:)`'s map.
    public func colour(for zone: BalanceZone?) -> Colour {
        guard let zone else { return noZoneColour }
        return zoneColours[zone] ?? noZoneColour
    }

    /// The joint-radius rule: `4 + 8 × severity` clamped to `0...1`, and a
    /// flat `4` when nothing measured the joint.
    public func jointRadius(severity: Double?) -> Double {
        guard let severity, severity.isFinite else { return jointBaseRadius }
        let clamped = Swift.min(Swift.max(severity, 0), 1)
        return jointBaseRadius + jointSeverityRadius * clamped
    }
}
