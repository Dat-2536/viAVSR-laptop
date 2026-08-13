import 'package:flutter/material.dart';

class ControlAndStatus extends StatefulWidget {
  const ControlAndStatus({super.key});

  @override
  State<ControlAndStatus> createState() => _ControlAndStatusState();
}

class _ControlAndStatusState extends State<ControlAndStatus> {
  String selectedFormat = "";

  void exportResult(String selectedFormat) {}

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          flex: 1,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              ElevatedButton(
                onPressed: () {},
                style: ElevatedButton.styleFrom(
                  backgroundColor: Color(0xff0096FF),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadiusGeometry.all(Radius.circular(5)),
                  ),
                ),
                child: Row(
                  children: [
                    Icon(Icons.upload, color: Colors.white),
                    Text("Upload", style: TextStyle(color: Colors.white)),
                  ],
                ),
              ),

              RecordButton(),

              ElevatedButton(
                onPressed: () {},
                style: ElevatedButton.styleFrom(
                  backgroundColor: Color(0xff5CE65C),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadiusGeometry.all(Radius.circular(5)),
                  ),
                ),
                child: Row(
                  children: [
                    // Icon(Icons.upload, color: Colors.white),
                    Text("Process", style: TextStyle(color: Colors.white)),
                  ],
                ),
              ),

              ElevatedButton(
                onPressed: () {},
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.white,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadiusGeometry.all(Radius.circular(5)),
                  ),
                ),
                child: Row(
                  children: [
                    Icon(Icons.replay, color: Colors.black),
                    Text("Reset", style: TextStyle(color: Colors.black)),
                  ],
                ),
              ),
            ],
          ),
        ),
        VerticalDivider(color: Colors.grey, indent: 4, endIndent: 4),
        Expanded(
          flex: 2,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              Text("Status"),

              Container(
                decoration: BoxDecoration(
                  color: Colors.white,
                  border: Border.all(color: Colors.grey.shade300),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: IntrinsicHeight(
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      InkWell(
                        borderRadius: const BorderRadius.horizontal(
                          left: Radius.circular(10),
                        ),
                        onTap: () => exportResult(selectedFormat),
                        child: const Padding(
                          padding: EdgeInsets.symmetric(
                            horizontal: 16,
                            vertical: 12,
                          ),
                          child: Row(
                            children: [
                              Icon(Icons.download_outlined, size: 18),
                              SizedBox(width: 8),
                              Text(
                                "Export Result",
                                style: TextStyle(fontWeight: FontWeight.w600),
                              ),
                            ],
                          ),
                        ),
                      ),

                      VerticalDivider(
                        width: 1,
                        thickness: 1,
                        color: Colors.grey.shade300,
                      ),

                      PopupMenuButton<String>(
                        padding: EdgeInsets.zero,
                        icon: const Icon(Icons.keyboard_arrow_down),
                        onSelected: (value) {
                          setState(() {
                            selectedFormat = value;
                          });
                        },
                        itemBuilder: (_) => const [
                          PopupMenuItem(value: "csv", child: Text("CSV")),
                          PopupMenuItem(value: "xlsx", child: Text("Excel")),
                          PopupMenuItem(value: "json", child: Text("JSON")),
                          PopupMenuItem(value: "pdf", child: Text("PDF")),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class RecordButton extends StatefulWidget {
  const RecordButton({super.key});

  @override
  State<RecordButton> createState() => _RecordButtonState();
}

class _RecordButtonState extends State<RecordButton> {
  bool isRecording = false;

  @override
  Widget build(BuildContext context) {
    return FilledButton.icon(
      style: FilledButton.styleFrom(
        backgroundColor: isRecording
            ? Colors.red
            : Theme.of(context).colorScheme.primary,
        foregroundColor: Colors.white,
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(5)),
      ),
      onPressed: () {
        setState(() {
          isRecording = !isRecording;
        });

        // TODO:
        // if (isRecording) {
        //   startRecording();
        // } else {
        //   stopRecording();
        // }
      },
      icon: Icon(isRecording ? Icons.stop : Icons.fiber_manual_record),
      label: Text(isRecording ? "Stop Recording" : "Start Recording"),
    );
  }
}
