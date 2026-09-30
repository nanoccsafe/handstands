import Foundation

// --------------------------------------------------------------------------- #
// Where a time sits on the heat strip, and where a tap on the strip is in
// time (chainlink #49): two pure functions, the only place an x fraction of
// the strip ever meets a `t_ms`.
//
// The strip is drawn left-to-right from the clip's first to its last
// timestamp (`SessionSummary.heatStrip`'s bins tile exactly that span), the
// playhead is placed with `fraction(tMs:…)`, and a tap or drag is read back
// with `tMs(fraction:…)` — so the line you tap and the line you see cannot
// disagree, and both directions are unit-tested without a view
// (`HeatStripGeometryTests`).
// --------------------------------------------------------------------------- #

/// The heat strip's x ↔ time mapping.
enum HeatStripGeometry {
    /// The timestamp a fraction `x` of the strip means: 0 is the clip's
    /// first `t_ms`, 1 its last, 0.5 the middle. Out-of-range fractions
    /// (a finger that slid off the strip) clamp to the ends, and a
    /// fraction that is not a number answers the first frame rather than
    /// NaN.
    static func tMs(fraction: Double, firstMs: Int, lastMs: Int) -> Int {
        let clamped = fraction.isFinite ? Swift.min(Swift.max(fraction, 0), 1) : 0
        guard lastMs > firstMs else { return firstMs }
        let span = Double(lastMs - firstMs)
        return firstMs + Int((span * clamped).rounded())
    }

    /// Where `tMs` sits on the strip, as the fraction `tMs(fraction:…)`
    /// reads back: 0 at the clip's first timestamp, 1 at its last, and a
    /// time outside the clip clamped into it. A clip with no span (one
    /// frame) answers 0 — there is nowhere to go.
    static func fraction(tMs: Int, firstMs: Int, lastMs: Int) -> Double {
        guard lastMs > firstMs else { return 0 }
        let clamped = Swift.min(Swift.max(Double(tMs), Double(firstMs)), Double(lastMs))
        return (clamped - Double(firstMs)) / Double(lastMs - firstMs)
    }
}
