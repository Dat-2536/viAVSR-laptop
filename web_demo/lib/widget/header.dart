import 'package:flutter/material.dart';

const String name = "Vietnamese AVSR - Record & Infer Demo";

const String techUsed = "AV-HuBERT + CTC/Attention";

class Header extends StatelessWidget {
  const Header({super.key});

  final String name = "Vietnamese AVSR - Record & Infer Demo";

  final String techUsed = "AV-HuBERT + CTC/Attention";

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.all(10),
      child: Row(
        children: [
          Logo(),
          SizedBox(width: 10),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                name,
                style: TextStyle(fontSize: 25, fontWeight: FontWeight.bold),
              ),
              Text(techUsed, style: TextStyle(fontSize: 20)),
            ],
          ),
        ],
      ),
    );
  }
}

class Logo extends StatelessWidget {
  const Logo({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 70,
      height: 70,
      decoration: BoxDecoration(
        color: Colors.blue,
        borderRadius: BorderRadiusGeometry.all(Radius.circular(15)),
      ),
      child: Icon(Icons.graphic_eq, color: Colors.white, size: 50),
    );
  }
}
