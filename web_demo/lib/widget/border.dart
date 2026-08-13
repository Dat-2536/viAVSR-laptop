import 'package:flutter/material.dart';

class ItemBorder extends StatelessWidget {
  final Widget child;
  const ItemBorder({super.key, required this.child});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        margin: EdgeInsets.symmetric(horizontal: 5, vertical: 3),
        height: 800,
        decoration: BoxDecoration(
          border: Border.all(color: Colors.grey, width: 0.5),
          borderRadius: BorderRadius.all(Radius.circular(5)),
        ),
        child: child,
      ),
    );
  }
}
