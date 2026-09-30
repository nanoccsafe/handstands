import Foundation

/// The strings the History screens show that are neither durations nor
/// pixel sizes (`VideoInfoFormatter` owns those): when a take happened and
/// what it scored. Kept out of the views so the tests can pin the formats
/// down without a view in the way.
enum SessionFormatter {
    /// `28 Sep 2026, 14:30` — a list of takes is read by date first, so
    /// date and time always appear together, in a fixed POSIX format that
    /// does not shift with the device language. The time zone is injected
    /// for the tests; the app uses the phone's.
    static func dateTime(_ date: Date, timeZone: TimeZone = .current) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = timeZone
        formatter.dateFormat = "d MMM yyyy, HH:mm"
        return formatter.string(from: date)
    }

    /// A clip score to two decimals. Anything that is not a real number
    /// (and the `nil` of a take that has not been analysed) reads "—"
    /// rather than "nan" or "-0.00"; the views say "Not analysed yet" for
    /// the `nil` case themselves.
    static func score(_ value: Double?) -> String {
        guard let value, value.isFinite else { return "—" }
        return String(format: "%.2f", value)
    }
}
