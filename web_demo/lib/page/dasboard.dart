import 'package:cv_web/widget/controlAndStatus.dart';
import 'package:flutter/material.dart';
import '../widget/header.dart';
import '../widget/border.dart';
import '../widget/controlAndStatus.dart';

class Dashboard extends StatefulWidget {
  const Dashboard({super.key});

  @override
  State<Dashboard> createState() => _DashboardState();
}

class _DashboardState extends State<Dashboard> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        padding: EdgeInsets.all(10),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Header(),

            Row(
              // mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                ItemBorder(child: Text("INput")),
                ItemBorder(child: Text("Processed")),
                ItemBorder(child: Text("Result")),
              ],
            ),

            ItemBorder(child: ControlAndStatus()),
          ],
        ),
      ),
    );
  }
}
