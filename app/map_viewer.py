from vispy import app
app.use_app("pyqt6")
from vispy import scene
from vispy.visuals.transforms import STTransform
import numpy as np
import time, os, json, sys, base64, struct, zlib
from datetime import datetime
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QCheckBox, QFileDialog, QFrame,
    QSizePolicy, QMessageBox, QScrollArea, QApplication
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon, QPixmap, QPainter, QBrush, QPen, QRadialGradient
from PyQt6.QtCore import QRectF, QPointF, QIODeviceBase

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ─── Icon generation ─────────────────────────────────────────────────────────
def _make_icon_pixmap(size: int = 512) -> QPixmap:
    """
    Draw a LiDAR scanner icon programmatically:
    dark navy background, cyan radar rings, bright scan dot,
    X/Y/Z axis arms. Looks sharp at any size.
    """
    px = QPixmap(size, size)
    px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    s = size
    c = s / 2          # centre
    r = s * 0.46       # outer usable radius

    # ── Background circle ─────────────────────────────────────────────────────
    bg = QRadialGradient(QPointF(c, c * 0.85), s * 0.55)
    bg.setColorAt(0,   QColor("#1a1a2e"))
    bg.setColorAt(0.7, QColor("#0d0d18"))
    bg.setColorAt(1,   QColor("#07070f"))
    p.setBrush(QBrush(bg))
    p.setPen(QPen(QColor("#00d4ff"), s * 0.012))
    p.drawEllipse(QRectF(s * 0.02, s * 0.02, s * 0.96, s * 0.96))

    # ── Radar rings ───────────────────────────────────────────────────────────
    for frac in (0.25, 0.50, 0.75, 1.0):
        ring_r = r * frac
        alpha  = int(60 + 80 * (1 - frac))
        col    = QColor(0, 212, 255, alpha)
        p.setPen(QPen(col, s * 0.007))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QRectF(c - ring_r, c - ring_r, ring_r * 2, ring_r * 2))

    # ── Cross-hairs ───────────────────────────────────────────────────────────
    hair_col = QColor(0, 212, 255, 50)
    p.setPen(QPen(hair_col, s * 0.005))
    p.drawLine(QPointF(c, c - r), QPointF(c, c + r))
    p.drawLine(QPointF(c - r, c), QPointF(c + r, c))

    # ── Sweep wedge ───────────────────────────────────────────────────────────
    from PyQt6.QtGui import QConicalGradient
    sweep = QConicalGradient(QPointF(c, c), 60)
    sweep.setColorAt(0,    QColor(0, 212, 255, 0))
    sweep.setColorAt(0.12, QColor(0, 212, 255, 55))
    sweep.setColorAt(0.22, QColor(0, 212, 255, 0))
    sweep.setColorAt(1,    QColor(0, 212, 255, 0))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(sweep))
    p.drawEllipse(QRectF(c - r, c - r, r * 2, r * 2))

    # ── Scan dot (bright cyan circle with glow) ────────────────────────────────
    dot_r = s * 0.07
    dot_x = c + r * 0.55
    dot_y = c - r * 0.30
    glow = QRadialGradient(QPointF(dot_x, dot_y), dot_r * 2.5)
    glow.setColorAt(0,   QColor(0, 255, 255, 180))
    glow.setColorAt(0.4, QColor(0, 212, 255, 80))
    glow.setColorAt(1,   QColor(0, 212, 255, 0))
    p.setBrush(QBrush(glow))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(QRectF(dot_x - dot_r * 2.5, dot_y - dot_r * 2.5, dot_r * 5, dot_r * 5))
    p.setBrush(QBrush(QColor("#00ffff")))
    p.drawEllipse(QRectF(dot_x - dot_r, dot_y - dot_r, dot_r * 2, dot_r * 2))

    # ── XYZ axis arms (bottom-left corner indicator) ──────────────────────────
    import math
    ox, oy  = c - r * 0.58, c + r * 0.58
    arm_len = r * 0.28
    axes = [
        (0,         QColor("#ff4466"), "X"),  # right
        (-math.pi / 2, QColor("#00ff9f"), "Y"),  # up
        (math.pi * 0.75, QColor("#4488ff"), "Z"),  # diagonal
    ]
    font = p.font()
    font.setPixelSize(max(8, int(s * 0.055)))
    font.setBold(True)
    p.setFont(font)
    for angle, col, label in axes:
        ex = ox + arm_len * math.cos(angle)
        ey = oy - arm_len * math.sin(angle)
        p.setPen(QPen(col, s * 0.018, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(ox, oy), QPointF(ex, ey))
        lx = ox + (arm_len + s * 0.04) * math.cos(angle)
        ly = oy - (arm_len + s * 0.04) * math.sin(angle)
        p.setPen(col)
        p.drawText(QRectF(lx - s * 0.05, ly - s * 0.05, s * 0.10, s * 0.10),
                   Qt.AlignmentFlag.AlignCenter, label)

    p.end()
    return px


def _app_icon() -> QIcon:
    icon = QIcon()
    for sz in (16, 32, 48, 64, 128, 256, 512):
        icon.addPixmap(_make_icon_pixmap(sz))
    return icon


def _pixmap_to_png_bytes(px: QPixmap) -> bytes:
    """Convert QPixmap → raw PNG bytes (no temp file)."""
    from PyQt6.QtCore import QBuffer
    buf = QBuffer()
    buf.open(QIODeviceBase.OpenModeFlag.WriteOnly)
    px.save(buf, "PNG")
    buf.close()
    return bytes(buf.data())


# ─── macOS .app bundle builder ────────────────────────────────────────────────
def _build_macos_app(dest_dir: str, script_path: str, python_exe: str) -> str:
    """
    Build a proper macOS .app bundle at dest_dir/LiDAR Viewer.app
    Structure:
      LiDAR Viewer.app/
        Contents/
          Info.plist
          MacOS/
            LiDAR Viewer          ← executable shell script
          Resources/
            AppIcon.icns          ← multi-size icon
    Returns the path to the .app bundle.
    """
    import shutil

    app_path  = os.path.join(dest_dir, "LiDAR Viewer.app")
    macos_dir = os.path.join(app_path, "Contents", "MacOS")
    res_dir   = os.path.join(app_path, "Contents", "Resources")
    os.makedirs(macos_dir, exist_ok=True)
    os.makedirs(res_dir,   exist_ok=True)

    # ── Info.plist ────────────────────────────────────────────────────────────
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>             <string>LiDAR Viewer</string>
    <key>CFBundleDisplayName</key>      <string>LiDAR Viewer</string>
    <key>CFBundleIdentifier</key>       <string>com.lidarviewer.app</string>
    <key>CFBundleVersion</key>          <string>0.3</string>
    <key>CFBundleShortVersionString</key><string>0.3</string>
    <key>CFBundlePackageType</key>      <string>APPL</string>
    <key>CFBundleSignature</key>        <string>????</string>
    <key>CFBundleExecutable</key>       <string>LiDAR Viewer</string>
    <key>CFBundleIconFile</key>         <string>AppIcon</string>
    <key>NSHighResolutionCapable</key>  <true/>
    <key>LSUIElement</key>              <false/>
</dict>
</plist>
"""
    with open(os.path.join(app_path, "Contents", "Info.plist"), "w") as f:
        f.write(plist)

    # ── Executable shell script ───────────────────────────────────────────────
    exec_path = os.path.join(macos_dir, "LiDAR Viewer")
    exec_content = f"""#!/bin/bash
# Activate the same Python environment that built this app
cd "{os.path.dirname(script_path)}"
exec "{python_exe}" "{script_path}" "$@"
"""
    with open(exec_path, "w") as f:
        f.write(exec_content)
    os.chmod(exec_path, 0o755)

    # ── Icon: .icns ───────────────────────────────────────────────────────────
    icns_path = os.path.join(res_dir, "AppIcon.icns")
    _write_icns(icns_path)

    return app_path


def _write_icns(path: str):
    """
    Write a minimal valid .icns file with several icon sizes.
    Uses the iconset sizes macOS expects.
    Falls back gracefully if Qt isn't available yet (shouldn't happen here).
    """
    # icns type codes → pixel sizes
    sizes = [
        (b"ic07",  128),
        (b"ic08",  256),
        (b"ic09",  512),
        (b"ic10", 1024),
        (b"ic11",   32),
        (b"ic12",   64),
        (b"ic13",  256),
        (b"ic14",  512),
    ]

    chunks = []
    for type_code, sz in sizes:
        px   = _make_icon_pixmap(sz)
        data = _pixmap_to_png_bytes(px)
        # each chunk: 4-byte OSType + 4-byte length (incl. header) + data
        chunk_len = 8 + len(data)
        chunks.append(struct.pack(">4sI", type_code, chunk_len) + data)

    body       = b"".join(chunks)
    total_len  = 8 + len(body)
    with open(path, "wb") as f:
        f.write(struct.pack(">4sI", b"icns", total_len))
        f.write(body)


# ─── Linux .desktop + icon builder ───────────────────────────────────────────
def _build_linux_desktop(script_path: str, python_exe: str):
    """
    Install a .desktop entry so LiDAR Viewer appears in the app launcher
    with a proper icon, and also drop a shortcut on ~/Desktop.
    """
    desktop_dir = os.path.expanduser("~/Desktop")
    apps_dir    = os.path.expanduser("~/.local/share/applications")
    icons_dir   = os.path.expanduser("~/.local/share/icons/hicolor/256x256/apps")
    os.makedirs(desktop_dir, exist_ok=True)
    os.makedirs(apps_dir,    exist_ok=True)
    os.makedirs(icons_dir,   exist_ok=True)

    # Save 256×256 PNG icon
    icon_path = os.path.join(icons_dir, "lidar_viewer.png")
    px = _make_icon_pixmap(256)
    px.save(icon_path, "PNG")

    # Also save a copy next to the script for portability
    local_icon = os.path.join(os.path.dirname(script_path), "lidar_viewer.png")
    px.save(local_icon, "PNG")

    entry = (
        "[Desktop Entry]\n"
        "Version=1.0\n"
        "Type=Application\n"
        "Name=LiDAR Viewer\n"
        "Comment=LiDAR point-cloud visualiser\n"
        f"Exec={python_exe} {script_path}\n"
        "Icon=lidar_viewer\n"
        "Terminal=false\n"
        "StartupNotify=true\n"
        "Categories=Science;Engineering;\n"
    )

    # System-wide app menu entry
    app_entry = os.path.join(apps_dir, "lidar_viewer.desktop")
    with open(app_entry, "w") as f:
        f.write(entry)
    os.chmod(app_entry, 0o644)

    # Desktop shortcut (needs to be trusted on GNOME)
    desk_entry = os.path.join(desktop_dir, "LiDAR Viewer.desktop")
    with open(desk_entry, "w") as f:
        f.write(entry)
    os.chmod(desk_entry, 0o755)

    # Attempt to mark as trusted (GNOME 3.28+)
    try:
        import subprocess
        subprocess.run(
            ["gio", "set", desk_entry, "metadata::trusted", "true"],
            check=False, capture_output=True
        )
    except Exception:
        pass

    # Refresh icon cache
    try:
        import subprocess
        subprocess.run(
            ["gtk-update-icon-cache", "-f", os.path.expanduser("~/.local/share/icons/hicolor")],
            check=False, capture_output=True
        )
    except Exception:
        pass


# ─── Unified launcher dispatcher ─────────────────────────────────────────────
def create_app_launcher(status_cb=None):
    """
    Build a native app launcher for the current OS.
    status_cb(msg, color) is called with progress updates.
    Returns a human-readable result string.
    """
    def status(msg, color="#00ff9f"):
        if status_cb:
            status_cb(msg, color)

    script_path = os.path.abspath(__file__)
    python_exe  = sys.executable

    if sys.platform == "darwin":
        status("Building .app bundle…", "#ffaa00")
        dest = os.path.expanduser("~/Desktop")
        try:
            app_path = _build_macos_app(dest, script_path, python_exe)
            status(f"App created → {os.path.basename(app_path)}", "#00ff9f")
            # Bounce the Finder so Dock picks up the new icon
            try:
                import subprocess
                subprocess.run(["killall", "Finder"], check=False, capture_output=True)
            except Exception:
                pass
            return app_path
        except Exception as e:
            status(f"Build failed: {e}", "#ff4466")
            raise

    elif sys.platform.startswith("linux"):
        status("Installing desktop entry…", "#ffaa00")
        try:
            _build_linux_desktop(script_path, python_exe)
            status("App icon installed ✓", "#00ff9f")
            return "installed"
        except Exception as e:
            status(f"Install failed: {e}", "#ff4466")
            raise

    else:
        status("Unsupported OS", "#ffaa00")
        return "unsupported"


# ─── Scene setup ─────────────────────────────────────────────────────────────
canvas = scene.SceneCanvas(
    keys='interactive', show=True,
    bgcolor='#0a0a0f', size=(1280, 720), title='LiDAR Viewer'
)
view = canvas.central_widget.add_view()
view.camera = 'turntable'
view.camera.fov = 60
view.camera.distance = 8

scatter = scene.visuals.Markers()
view.add(scatter)
axis = scene.visuals.XYZAxis(parent=view.scene)
grid  = scene.visuals.GridLines(color=(0.2, 0.2, 0.3, 0.6), parent=view.scene)
grid.transform = STTransform(scale=(1, 1, 1))

# ─── State ───────────────────────────────────────────────────────────────────
all_points        = np.empty((0, 3))
point_size        = 3
color_mode        = "distance"   # "distance" | "white"
autosave_enabled  = False
last_autosave     = time.time()
AUTOSAVE_INTERVAL = 60
MAX_POINTS        = 50_000
SAVES_DIR         = "scans"
os.makedirs(SAVES_DIR, exist_ok=True)
_fps_times: list[float] = []

# ─── Helpers ─────────────────────────────────────────────────────────────────
def distance_colors(pts):
    """Red = close to origin, green = medium, blue = far."""
    if len(pts) == 0:
        return np.empty((0, 4))
    dist = np.sqrt((pts ** 2).sum(axis=1))
    t = (dist - dist.min()) / (dist.max() - dist.min() + 1e-9)
    # t=0 → red, t=0.5 → green, t=1 → blue
    r = np.clip(1 - 2 * t, 0, 1)
    g = np.clip(1 - np.abs(2 * t - 1) * 2, 0, 1)
    b = np.clip(2 * t - 1, 0, 1)
    return np.column_stack([r, g, b, np.ones_like(r)]).astype(np.float32)

def refresh_scatter():
    if color_mode == "distance":
        colors = distance_colors(all_points)
    else:
        colors = 'white'
    scatter.set_data(all_points, edge_color=None, face_color=colors, size=point_size)

def _scan_filename(prefix="scan", ext="ply"):
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return os.path.join(SAVES_DIR, f"{prefix}_{ts}.{ext}")

def save_ply(path=None):
    if len(all_points) == 0: return None
    path = path or _scan_filename("scan","ply")
    pts  = all_points.astype(np.float32)
    hdr  = f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    with open(path,"w") as f:
        f.write(hdr); np.savetxt(f, pts, fmt="%.6f")
    return path

def save_pcd(path=None):
    if len(all_points) == 0: return None
    path = path or _scan_filename("scan","pcd")
    pts  = all_points.astype(np.float32)
    hdr  = (f"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n"
            f"COUNT 1 1 1\nWIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {len(pts)}\nDATA ascii\n")
    with open(path,"w") as f:
        f.write(hdr); np.savetxt(f, pts, fmt="%.6f")
    return path

def save_xyz(path=None):
    if len(all_points) == 0: return None
    path = path or _scan_filename("scan","xyz")
    np.savetxt(path, all_points.astype(np.float32), fmt="%.6f")
    return path

def save_csv(path=None):
    if len(all_points) == 0: return None
    path = path or _scan_filename("scan","csv")
    np.savetxt(path, all_points.astype(np.float32), delimiter=",",
               header="x,y,z", comments="", fmt="%.6f")
    return path

def save_metadata(scan_path):
    meta = {
        "timestamp": datetime.now().isoformat(),
        "point_count": int(len(all_points)),
        "bounds": {ax: [float(all_points[:,i].min()), float(all_points[:,i].max())]
                   for i, ax in enumerate("xyz")} if len(all_points) else {},
        "source_file": os.path.basename(scan_path)
    }
    mp = scan_path.rsplit(".",1)[0] + "_meta.json"
    with open(mp,"w") as f: json.dump(meta, f, indent=2)
    return mp

def load_ply(path):
    global all_points
    pts, in_data = [], False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "end_header": in_data = True; continue
            if in_data:
                v = line.split()
                if len(v) >= 3: pts.append([float(x) for x in v[:3]])
    if pts:
        all_points = np.array(pts, dtype=np.float32)
        refresh_scatter()
    return len(pts)

def save_screenshot(path=None):
    path = path or _scan_filename("screenshot","png")
    img  = canvas.render()
    if HAS_PIL: Image.fromarray(img).save(path)
    else:
        from vispy.io import write_png; write_png(path, img)
    return path

# ─── Design tokens ───────────────────────────────────────────────────────────
DARK_BG   = "#0d0d14"
PANEL_BG  = "#13131e"
ACCENT    = "#00d4ff"
BTN_BG    = "#1e1e2e"
BTN_HOVER = "#2a2a3e"
TEXT      = "#e0e0f0"
MUTED     = "#5a5a7a"
SUCCESS   = "#00ff9f"
WARN      = "#ffaa00"
DANGER    = "#ff4466"

BUTTON_STYLE = f"""
QPushButton {{
    background: {BTN_BG}; color: {TEXT};
    border: 1px solid #2a2a3e; border-radius: 4px;
    padding: 5px 10px;
    font-family: 'JetBrains Mono','Consolas',monospace; font-size: 11px;
}}
QPushButton:hover  {{ background: {BTN_HOVER}; border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {ACCENT}; color: #000; }}
"""
LABEL_STYLE  = f"color:{TEXT}; font-family:'Consolas',monospace; font-size:11px;"
HEADER_STYLE = f"color:{ACCENT}; font-family:'Consolas',monospace; font-size:10px; font-weight:bold; letter-spacing:1px;"
MUTED_STYLE  = f"color:{MUTED}; font-family:'Consolas',monospace; font-size:10px;"

def _sep():
    l = QFrame(); l.setFrameShape(QFrame.Shape.HLine)
    l.setStyleSheet("border: 0; border-top: 1px solid #1e1e2e;"); return l

def _header(txt):
    l = QLabel(txt); l.setStyleSheet(HEADER_STYLE); return l

# ─── Sidebar ─────────────────────────────────────────────────────────────────
class Sidebar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(220)
        self.setStyleSheet(f"background:{PANEL_BG};")

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea{border:none;background:transparent;}"
            "QScrollBar:vertical{background:#1e1e2e;width:4px;}"
            "QScrollBar::handle:vertical{background:#2a2a3e;border-radius:2px;}")

        inner = QWidget(); inner.setStyleSheet(f"background:{PANEL_BG};")
        L = QVBoxLayout(inner)
        L.setContentsMargins(10,12,10,12); L.setSpacing(6)

        title = QLabel("LiDAR Viewer")
        title.setStyleSheet(f"color:{ACCENT};font-size:14px;font-weight:bold;font-family:'Consolas',monospace;")
        L.addWidget(title)
        sub = QLabel("v0.4  |  live scan"); sub.setStyleSheet(MUTED_STYLE); L.addWidget(sub)
        L.addWidget(_sep())

        L.addWidget(_header("STATS"))
        self.lbl_fps = QLabel("FPS: —"); self.lbl_fps.setStyleSheet(LABEL_STYLE); L.addWidget(self.lbl_fps)
        self.lbl_pts = QLabel("Points: 0"); self.lbl_pts.setStyleSheet(LABEL_STYLE); L.addWidget(self.lbl_pts)
        L.addWidget(_sep())

        L.addWidget(_header("SCAN CAPTURE"))
        self.btn_save_ply = QPushButton("💾  Save  .ply")
        self.btn_save_pcd = QPushButton("💾  Save  .pcd")
        self.btn_save_xyz = QPushButton("💾  Save  .xyz")
        self.btn_save_csv = QPushButton("💾  Save  .csv")
        self.btn_load     = QPushButton("📂  Load  .ply")
        self.btn_shot     = QPushButton("📷  Screenshot  .png")
        for b in (self.btn_save_ply,self.btn_save_pcd,self.btn_save_xyz,
                  self.btn_save_csv,self.btn_load,self.btn_shot):
            b.setStyleSheet(BUTTON_STYLE); L.addWidget(b)

        self.chk_autosave = QCheckBox("Autosave (60 s)")
        self.chk_autosave.setStyleSheet(f"color:{TEXT};font-size:11px;font-family:'Consolas',monospace;")
        L.addWidget(self.chk_autosave)
        self.lbl_last_save = QLabel("Last save: —"); self.lbl_last_save.setStyleSheet(MUTED_STYLE)
        self.lbl_last_save.setWordWrap(True); L.addWidget(self.lbl_last_save)
        L.addWidget(_sep())

        L.addWidget(_header("RENDERING"))
        lps = QLabel("Point size"); lps.setStyleSheet(LABEL_STYLE); L.addWidget(lps)
        self.slider_size = QSlider(Qt.Orientation.Horizontal)
        self.slider_size.setRange(1,10); self.slider_size.setValue(3)
        self.slider_size.setStyleSheet(f"accent-color:{ACCENT};"); L.addWidget(self.slider_size)
        self.lbl_pt_size_val = QLabel("Size: 3"); self.lbl_pt_size_val.setStyleSheet(MUTED_STYLE)
        L.addWidget(self.lbl_pt_size_val)

        self.chk_color_dist = QCheckBox("Distance coloring  🔴🟢🔵")
        self.chk_color_dist.setChecked(True)
        self.chk_color_dist.setStyleSheet(f"color:{TEXT};font-size:11px;font-family:'Consolas',monospace;")
        L.addWidget(self.chk_color_dist)
        self.chk_color_white = QCheckBox("White points")
        self.chk_color_white.setChecked(False)
        self.chk_color_white.setStyleSheet(f"color:{TEXT};font-size:11px;font-family:'Consolas',monospace;")
        L.addWidget(self.chk_color_white)
        self.chk_grid = QCheckBox("Ground grid"); self.chk_grid.setChecked(True)
        self.chk_grid.setStyleSheet(f"color:{TEXT};font-size:11px;font-family:'Consolas',monospace;")
        L.addWidget(self.chk_grid)
        self.chk_axis = QCheckBox("XYZ axis"); self.chk_axis.setChecked(True)
        self.chk_axis.setStyleSheet(f"color:{TEXT};font-size:11px;font-family:'Consolas',monospace;")
        L.addWidget(self.chk_axis)
        L.addWidget(_sep())

        L.addWidget(_header("CAMERA"))
        self.btn_reset_cam  = QPushButton("⟳  Reset Camera")
        self.btn_top_view   = QPushButton("↑  Top View")
        self.btn_front_view = QPushButton("→  Front View")
        self.btn_side_view  = QPushButton("◁  Side View")
        for b in (self.btn_reset_cam,self.btn_top_view,self.btn_front_view,self.btn_side_view):
            b.setStyleSheet(BUTTON_STYLE); L.addWidget(b)
        L.addWidget(_sep())

        L.addWidget(_header("MAP"))
        self.btn_clear = QPushButton("🗑  Clear Map")
        self.btn_clear.setStyleSheet(BUTTON_STYLE.replace(BTN_BG,"#2e1020").replace(BTN_HOVER,"#3e1530"))
        L.addWidget(self.btn_clear)
        L.addWidget(_sep())

        L.addWidget(_header("TOOLS"))
        self.btn_launcher = QPushButton("🖥  Install App Icon")
        self.btn_launcher.setStyleSheet(BUTTON_STYLE)
        L.addWidget(self.btn_launcher)
        self.btn_close = QPushButton("✕  Close App")
        self.btn_close.setStyleSheet(
            BUTTON_STYLE.replace(BTN_BG,"#2e1020").replace(BTN_HOVER,"#3e1530"))
        L.addWidget(self.btn_close)

        L.addStretch()
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet(f"color:{SUCCESS};font-size:10px;font-family:'Consolas',monospace;")
        self.lbl_status.setWordWrap(True); L.addWidget(self.lbl_status)

        scroll.setWidget(inner)
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0); outer.addWidget(scroll)

    def set_status(self, msg, color=SUCCESS):
        self.lbl_status.setText(msg)
        self.lbl_status.setStyleSheet(f"color:{color};font-size:10px;font-family:'Consolas',monospace;")


# ─── Main Window ─────────────────────────────────────────────────────────────
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiDAR Viewer")
        self.setWindowIcon(_app_icon())
        self.setStyleSheet(f"background:{DARK_BG};")
        self.resize(1500, 800)

        root = QHBoxLayout(self)
        root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        self.sidebar = Sidebar()
        root.addWidget(self.sidebar)

        native = canvas.native
        native.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(native)

        self._connect_signals()
        self._stats_timer = QTimer()
        self._stats_timer.timeout.connect(self._refresh_stats)
        self._stats_timer.start(200)

    def _connect_signals(self):
        sb = self.sidebar
        sb.btn_save_ply.clicked.connect(self._on_save_ply)
        sb.btn_save_pcd.clicked.connect(self._on_save_pcd)
        sb.btn_save_xyz.clicked.connect(self._on_save_xyz)
        sb.btn_save_csv.clicked.connect(self._on_save_csv)
        sb.btn_load.clicked.connect(self._on_load)
        sb.btn_shot.clicked.connect(self._on_screenshot)
        sb.chk_autosave.stateChanged.connect(self._on_autosave_toggle)
        sb.slider_size.valueChanged.connect(self._on_size_change)
        sb.chk_grid.stateChanged.connect(self._on_grid_toggle)
        sb.chk_axis.stateChanged.connect(self._on_axis_toggle)
        sb.btn_reset_cam.clicked.connect(self._on_reset_cam)
        sb.btn_top_view.clicked.connect(self._on_top_view)
        sb.btn_front_view.clicked.connect(self._on_front_view)
        sb.btn_side_view.clicked.connect(self._on_side_view)
        sb.btn_clear.clicked.connect(self._on_clear)
        sb.btn_launcher.clicked.connect(self._on_install_app)
        sb.btn_close.clicked.connect(self.close)
        sb.chk_color_dist.stateChanged.connect(self._on_color_dist)
        sb.chk_color_white.stateChanged.connect(self._on_color_white)

    def _notify_save(self, path):
        if path is None: self.sidebar.set_status("No points to save.", WARN); return
        name = os.path.basename(path)
        self.sidebar.lbl_last_save.setText(f"Last save: {name}")
        self.sidebar.set_status(f"Saved → {name}", SUCCESS)

    def _on_save_ply(self):
        path = save_ply()
        if path: save_metadata(path)
        self._notify_save(path)

    def _on_save_pcd(self):  self._notify_save(save_pcd())
    def _on_save_xyz(self):  self._notify_save(save_xyz())
    def _on_save_csv(self):  self._notify_save(save_csv())

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load PLY", SAVES_DIR, "PLY files (*.ply)")
        if path:
            n = load_ply(path)
            self.sidebar.set_status(f"Loaded {n} pts  ←  {os.path.basename(path)}", ACCENT)

    def _on_screenshot(self):
        path = save_screenshot()
        self.sidebar.set_status(f"Screenshot → {os.path.basename(path)}", SUCCESS)

    def _on_autosave_toggle(self, state):
        global autosave_enabled
        autosave_enabled = bool(state)
        self.sidebar.set_status("Autosave ON" if autosave_enabled else "Autosave OFF",
                                SUCCESS if autosave_enabled else MUTED)

    def _on_size_change(self, val):
        global point_size; point_size = val
        self.sidebar.lbl_pt_size_val.setText(f"Size: {val}")
        refresh_scatter()

    def _on_color_dist(self, state):
        global color_mode
        if state:
            color_mode = "distance"
            self.sidebar.chk_color_white.setChecked(False)
        refresh_scatter()

    def _on_color_white(self, state):
        global color_mode
        if state:
            color_mode = "white"
            self.sidebar.chk_color_dist.setChecked(False)
        refresh_scatter()

    def _on_grid_toggle(self, state):  grid.visible  = bool(state)
    def _on_axis_toggle(self, state):  axis.visible  = bool(state)

    def _on_reset_cam(self):
        view.camera.set_range(x=(-5,5),y=(-5,5),z=(-5,5))
        view.camera.distance=8; view.camera.elevation=30; view.camera.azimuth=45

    def _on_top_view(self):   view.camera.elevation=90; view.camera.azimuth=0
    def _on_front_view(self): view.camera.elevation=0;  view.camera.azimuth=0
    def _on_side_view(self):  view.camera.elevation=0;  view.camera.azimuth=90

    def _on_clear(self):
        global all_points; all_points = np.empty((0,3))
        scatter.set_data(np.zeros((1,3)), face_color=(0,0,0,0), size=1)
        self.sidebar.set_status("Map cleared.", WARN)

    def _on_install_app(self):
        try:
            create_app_launcher(status_cb=self.sidebar.set_status)
        except Exception as e:
            self.sidebar.set_status(f"Error: {e}", DANGER)

    def _refresh_stats(self):
        self.sidebar.lbl_pts.setText(f"Points: {len(all_points):,}")
        if _fps_times:
            self.sidebar.lbl_fps.setText(f"FPS: {sum(_fps_times)/len(_fps_times):.1f}")


# ─── Scan update loop ─────────────────────────────────────────────────────────
_last_frame_time = time.perf_counter()

def update(event):
    global all_points, last_autosave, _last_frame_time
    now = time.perf_counter(); dt = now - _last_frame_time; _last_frame_time = now
    if dt > 0:
        _fps_times.append(1.0/dt)
        if len(_fps_times) > 30: _fps_times.pop(0)
    new_pts = np.random.uniform(-2,2,(200,3)).astype(np.float32)
    all_points = np.vstack((all_points, new_pts))
    if len(all_points) > MAX_POINTS: all_points = all_points[-MAX_POINTS:]
    refresh_scatter()
    if autosave_enabled and (now - last_autosave) >= AUTOSAVE_INTERVAL:
        path = save_ply()
        if path: save_metadata(path)
        last_autosave = now

timer = app.Timer(interval=0.05, connect=update, start=True)

def try_load_from_args():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            timer.stop(); load_ply(path)
            print(f"Loaded {len(all_points)} points from {path}")

# ─── Entry ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    qt_app = QApplication.instance() or QApplication(sys.argv)
    qt_app.setWindowIcon(_app_icon())
    qt_app.setApplicationName("LiDAR Viewer")
    qt_app.setApplicationDisplayName("LiDAR Viewer")
    win = MainWindow()
    win.show()
    try_load_from_args()
    app.run()
