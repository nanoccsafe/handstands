import HandstandCore
import SwiftUI

/// About: the app version, and two lines computed from `HandstandCore` at
/// render time — the package is a real dependency of this target (see
/// ios/project.yml), and these numbers are the proof: the 15-joint schema and
/// a body-frame conversion the package's own tests pin down.
struct AboutView: View {
    var body: some View {
        List {
            Section("Version") {
                Text(AboutText.version)
            }
            Section("HandstandCore") {
                Text(AboutText.jointCountLine)
                Text(AboutText.bodyFrameLine)
            }
            Section {
                Text("No network, no analytics: recordings stay on this iPhone.")
            }
        }
        .navigationTitle("About")
    }
}

/// The About screen's strings, computed from the app's own build and the
/// linked package rather than typed in.
enum AboutText {
    /// e.g. `"0.1.0 (1)"`; falls back to whatever the bundle does have.
    static var version: String {
        let short = (Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String) ?? ""
        let build = (Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String) ?? ""
        guard !short.isEmpty else {
            return build.isEmpty ? "unknown" : build
        }
        guard !build.isEmpty, build != short else { return short }
        return "\(short) (\(build))"
    }

    /// `"15 joints tracked, as in the shared schema."`
    static var jointCountLine: String {
        "\(Joint.allCases.count) joints tracked, as in the shared schema."
    }

    /// A fixed sample point one body length above the wrist midpoint:
    /// `"Body frame of the sample point: u 0.00, v 1.00."`
    static var bodyFrameLine: String {
        guard let uv = try? BodyFrame.toBodyFrame(
            x: 120, y: 100, wristMidX: 120, wristMidY: 300, bodyLength: 200
        ) else {
            return "Body frame: unavailable"
        }
        return "Body frame of the sample point: u \(decimal(uv.u)), v \(decimal(uv.v))."
    }

    private static func decimal(_ value: Double) -> String {
        String(format: "%.2f", value)
    }
}
