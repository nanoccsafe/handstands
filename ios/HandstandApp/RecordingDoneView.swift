import HandstandCore
import SwiftUI

/// What comes after Stop: the file that was just written, as the same two
/// numbers the video-picker screen shows — duration and pixel size, read
/// from the file itself by `VideoInfoReader` — plus the hold this take was
/// recorded for (chainlink #66), and the two ways out: another take, or
/// back home.
struct RecordingDoneView: View {
    let info: VideoInfo
    /// The hold selected when this recording started — what its sidecar says.
    let holdType: HoldType
    let onRecordAgain: () -> Void
    let onDone: () -> Void

    var body: some View {
        VStack(spacing: 24) {
            Label("Recording saved", systemImage: "checkmark.circle.fill")
                .font(.title2.weight(.semibold))
                .foregroundStyle(.green)

            VStack(spacing: 8) {
                LabeledContent(
                    "Duration",
                    value: VideoInfoFormatter.duration(info.duration)
                )
                LabeledContent(
                    "Frame",
                    value: VideoInfoFormatter.pixelSize(width: info.width, height: info.height)
                )
                LabeledContent("Hold", value: holdType.displayName)
            }
            .frame(maxWidth: 300)

            HStack(spacing: 16) {
                Button("Record again", action: onRecordAgain)
                    .buttonStyle(.bordered)
                Button("Done", action: onDone)
                    .buttonStyle(.borderedProminent)
            }
            .controlSize(.large)
        }
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
