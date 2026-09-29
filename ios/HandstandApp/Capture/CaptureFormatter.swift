import Foundation

/// The strings the Record screen shows that are not the framing message:
/// just the running clock while the red light is on. Kept out of SwiftUI so
/// the tests can pin the format down without a view in the way.
enum CaptureFormatter {
    /// `7.24` → `"0:07.2"`, `10.0` → `"0:10.0"`, `65.4` → `"1:05.4"` —
    /// minutes, seconds and tenths, because a ten-second attempt needs to
    /// show that it is counting. Anything that is not a real, non-negative,
    /// finite number of seconds reads `"0:00.0"` instead of trapping.
    ///
    /// The finished recording's length comes from the file itself and is
    /// shown with `VideoInfoFormatter.duration`; this is only the live
    /// estimate between tap and tap.
    static func elapsed(_ seconds: TimeInterval) -> String {
        guard seconds.isFinite, seconds >= 0, seconds < 100_000_000 else {
            return "0:00.0"
        }
        let tenthsTotal = Int((seconds * 10).rounded())
        let minutes = tenthsTotal / 600
        let wholeSeconds = (tenthsTotal / 10) % 60
        let tenth = tenthsTotal % 10
        return "\(minutes):\(padded(wholeSeconds)).\(tenth)"
    }

    private static func padded(_ value: Int) -> String {
        value < 10 ? "0\(value)" : "\(value)"
    }
}
