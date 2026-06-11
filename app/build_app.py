#!/usr/bin/env python3
"""
build_app.py — one-shot LiDAR Viewer app installer.

Run this ONCE from the same folder as map_viewer.py:
    python3 build_app.py

macOS → builds LiDAR Viewer.app on ~/Desktop (drag to /Applications to keep)
Linux → installs .desktop entry + icon so it appears in your app launcher
"""

import sys, os

# ── Minimal Qt bootstrap just for the icon drawing ────────────────────────────
from PyQt6.QtWidgets import QApplication
_qapp = QApplication.instance() or QApplication(sys.argv)

# Pull everything from the viewer itself
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_viewer import create_app_launcher, _app_icon

_qapp.setWindowIcon(_app_icon())

def main():
    print("LiDAR Viewer — app installer")
    print(f"Platform : {sys.platform}")
    print(f"Python   : {sys.executable}")
    print()

    results = []

    def cb(msg, color="#00ff9f"):
        tag = "✓" if "00ff9f" in color else ("!" if "ffaa00" in color else "✗")
        print(f"  [{tag}] {msg}")
        results.append((msg, color))

    try:
        out = create_app_launcher(status_cb=cb)
        print()
        if sys.platform == "darwin":
            print(f"  App bundle → {out}")
            print()
            print("  Next steps:")
            print("   • Double-click the icon on your Desktop to launch")
            print("   • Drag it to /Applications to keep it permanently")
            print("   • Right-click → Options → Keep in Dock to pin it")
        elif sys.platform.startswith("linux"):
            print("  .desktop entry installed.")
            print()
            print("  Next steps:")
            print("   • Log out and back in (or run: gtk-update-icon-cache -f ~/.local/share/icons/hicolor)")
            print("   • The app appears in your Activities / app grid")
            print("   • Desktop shortcut also placed on ~/Desktop")
            print("   • GNOME: right-click the desktop icon → Allow Launching")
    except Exception as e:
        print(f"\n  [✗] Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
