#!/usr/bin/env python3
"""
RPEA Scanner Project
====================
Stage 1: Live 2D LiDAR scan, save, load, viewer.
Stage 2: ESP32 stepper integration — true 3D sweep.

Run:
    cd ~/hailo-rpi5-examples && source setup_env.sh
    python3 ~/scanner_project/scanner.py

One command. No TCP. No ROS. No networking. No subprocesses.

──────────────────────────────────────────────────────────────
SECTIONS
  CONFIG
  GLOBALS
  DEBUG INSTRUMENTATION
  PORT DETECTION
  YDLIDAR
  RING BUFFER
  ESP32
  SCAN CONTROLLER
  PROJECTION
  SAVE / LOAD
  VISPY
  UI
  MAIN LOOP
──────────────────────────────────────────────────────────────

FIXES vs previous version
  • BUG FIX: accumulate mode no longer writes the same revolution
    multiple times. A monotonic _latest_rev_id counter is compared
    in _update(); a scan is only written to the ring buffer when the
    ID changes (i.e. once per real LiDAR revolution, not 4-5 times).
  • _ring_view() comment corrected — np.concatenate() does allocate.
  • DRAIN_PER_FRAME removed — it was defined but never used.
  • Level-of-detail (LOD) rendering: above LOD_THRESHOLD points the
    scatter uploads every Nth point while the user interacts, then
    restores full resolution when the camera is idle. Keeps the Pi
    smooth on large scans.
  • ESP32 serial port now closed in a finally block, even if
    readline() raises mid-loop on an abrupt disconnect.
  • ESP32 port is now re-detected on every reconnect attempt instead
    of once at thread start — fixes ESP32 never connecting when it
    raced the LiDAR thread for _lidar_status["port"] at startup.
  • Debug instrumentation (DEBUG flag) — throttled memory/thread/
    render heartbeat, gated so it costs nothing when DEBUG = False.
    Now runs on its own background thread instead of inside _update()
    on the GUI thread — gc.collect() + tracemalloc snapshot/diff are
    genuinely slow operations, and running them on the GUI thread
    caused a multi-second UI freeze every DEBUG_INTERVAL_SEC.
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import sys
import os
import time
import json
import math
import threading
import queue
import glob
import gc
import traceback
from datetime import datetime

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np

try:
    import serial
    _HAS_SERIAL = True
except ImportError:
    _HAS_SERIAL = False

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# ── YDLidar SDK ───────────────────────────────────────────────────────────────
try:
    sys.path.insert(0, os.path.expanduser("~/YDLidar-SDK/build/python"))
    import ydlidar
    _SDK_OK = True
except ImportError:
    _SDK_OK = False

# ── VisPy / PyQt6 ─────────────────────────────────────────────────────────────
from vispy import app as vapp
vapp.use_app("pyqt6")
from vispy import scene
from vispy.visuals.transforms import STTransform

from PyQt6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QSlider, QCheckBox,
    QFileDialog, QFrame, QSizePolicy, QScrollArea,
)
from PyQt6.QtCore import Qt, QTimer, QObject, pyqtSignal

# ── debug instrumentation (opt-in, throttled — see print_debug() below) ──────
DEBUG = True

if DEBUG:
    import tracemalloc
    tracemalloc.start(6)   # shallow depth — deep traces slow allocation itself
    try:
        import psutil
        _HAS_PSUTIL = True
    except ImportError:
        _HAS_PSUTIL = False
    try:
        import resource
        _HAS_RESOURCE = True
    except ImportError:
        _HAS_RESOURCE = False
else:
    _HAS_PSUTIL = False
    _HAS_RESOURCE = False


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

SDK_DIR   = os.path.expanduser("~/YDLidar-SDK")
SAVES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scans")

# ── Ring buffer ───────────────────────────────────────────────────────────────
# Pi 5 8 GB: 300_000 × 3 × 4 B = ~3.6 MB — very comfortable.
RING_CAPACITY = 300_000

# ── Performance ───────────────────────────────────────────────────────────────
AUTOSAVE_SEC     = 60
RENDER_HZ        = 15      # hard ceiling on GPU uploads/sec

# ── Level-of-detail ──────────────────────────────────────────────────────────
# Above LOD_THRESHOLD points, upload only every LOD_STRIDE-th point while
# the user is interacting (camera moving). Full detail restored after
# LOD_IDLE_SEC seconds of no camera movement.
LOD_THRESHOLD = 80_000
LOD_STRIDE    = 4
LOD_IDLE_SEC  = 1.0

# ── LiDAR — do not change ─────────────────────────────────────────────────────
LIDAR_BAUD       = 115200
LIDAR_TYPE       = 1
LIDAR_DEV_TYPE   = 0
LIDAR_SINGLE_CH  = True
LIDAR_FIXED_RES  = False   # False avoids "real points > fixed points" on X2
LIDAR_AUTO_RECON = True
LIDAR_MIN_RANGE  = 0.12    # metres
LIDAR_MAX_RANGE  = 12.0
LIDAR_MIN_ANGLE  = -180.0  # degrees
LIDAR_MAX_ANGLE  =  180.0
LIDAR_FREQ       = 7.0     # Hz

# ── ESP32 — do not change ─────────────────────────────────────────────────────
ESP32_BAUD     = 115200
ESP32_TIMEOUT  = 10.0
SCAN_INCREMENT = 0.6   # °/step; 400 half-steps/rev, 30:20 gear → 0.6°/platform step


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBALS
# ══════════════════════════════════════════════════════════════════════════════

os.makedirs(SAVES_DIR, exist_ok=True)

# ── LiDAR fan-out ─────────────────────────────────────────────────────────────
#
#   _latest_rev      — most recent complete revolution (Stage 1).
#                      Replaced atomically; never extended in-place.
#   _latest_rev_id   — monotonically increasing counter; incremented once per
#                      real revolution.  _update() only writes to the ring
#                      buffer when this ID changes — fixes the duplicate-scan
#                      accumulate bug.
#   _lidar_q_s2      — private queue for Stage 2 only; Stage 1 never reads it.
#
_latest_rev      = []          # list[tuple[float,float]]
_latest_rev_id   = 0           # int — revolution counter (monotonic)
_latest_rev_lock = threading.Lock()

_lidar_q_s2  = queue.Queue(maxsize=20_000)
_lidar_status = {"connected": False, "error": None, "port": "detecting…", "total": 0}
_stop_lidar   = threading.Event()

# ── ESP32 ─────────────────────────────────────────────────────────────────────
_esp32_status = {"connected": False, "error": None, "port": "detecting…",
                 "angle": 0.0, "state": "—"}
_esp32_ser    = None
_esp32_lock   = threading.Lock()
_stop_esp32   = threading.Event()

# ── Scan controller ───────────────────────────────────────────────────────────
_scan_active = False
_scan_lock   = threading.Lock()

# ── Ring buffer ───────────────────────────────────────────────────────────────
_ring_buf   = np.zeros((RING_CAPACITY, 3), dtype=np.float32)
_ring_head  = 0
_ring_count = 0
_pts_lock   = threading.Lock()

# Incremented on every real change to ring contents (write or clear). Lets
# _refresh_scatter() know whether it's safe to reuse cached height colors
# instead of recomputing them — see fix #2 in _refresh_scatter().
_data_version = 0

# ── Render control ────────────────────────────────────────────────────────────
_cloud_dirty   = threading.Event()
_RENDER_INTV   = 1.0 / RENDER_HZ
_last_render_t = 0.0

# ── LOD camera-idle tracking ──────────────────────────────────────────────────
_cam_last_move_t = 0.0   # wall-clock time of last camera-movement event

# ── VisPy scene (created after QApplication) ──────────────────────────────────
_canvas  = None
_view    = None
_scatter = None
_axis    = None
_grid    = None

# ── Viewer state ──────────────────────────────────────────────────────────────
_point_size    = 2
_height_col    = True
_autosave_on   = False
_last_autosave = time.time()
_fps_buf: list[float] = []
_s1_live_mode  = True   # True = live (latest rev); False = accumulate

# ── Debug thread control ──────────────────────────────────────────────────────
_stop_debug = threading.Event()


# ══════════════════════════════════════════════════════════════════════════════
#  DEBUG INSTRUMENTATION
#  Runs on its OWN background thread (_debug_thread), not inside _update() on
#  the GUI thread. gc.collect() and tracemalloc's snapshot/compare_to() are
#  genuinely slow (heap walks), and running them on the GUI thread caused a
#  multi-second freeze every DEBUG_INTERVAL_SEC — indistinguishable from the
#  app being "not responding." Moving this off the GUI thread fixes that
#  while keeping full diagnostics available whenever DEBUG = True.
# ══════════════════════════════════════════════════════════════════════════════

DEBUG_INTERVAL_SEC = 5.0

_process           = psutil.Process(os.getpid()) if _HAS_PSUTIL else None
_prev_snapshot     = None
_prev_obj_count    = 0
_peak_rss_mb       = 0.0
_last_thread_count = 0

# Counters accumulated between debug prints (cheap increments, not per-event prints)
_dbg_render_uploads   = 0     # GPU uploads since last report
_dbg_render_pts_total = 0     # total points uploaded since last report
_dbg_render_ms_total  = 0.0   # total render time since last report
_dbg_revs_seen        = 0     # LiDAR revolutions processed since last report (Stage 1)
_dbg_revs_seen_s2     = 0     # LiDAR revolutions processed since last report (Stage 2)
_dbg_update_ticks     = 0     # _update() calls since last report
_dbg_render_ticks     = 0     # _on_render_requested() dispatch calls since last report


def print_debug():
    """
    One full debug report. Called only from _debug_thread() on its own
    background thread — never from _update() / the GUI thread. The heavy
    parts here (gc.collect(), tracemalloc snapshot + diff) can take a
    meaningful fraction of a second on a Pi; that's fine on a background
    thread but was the direct cause of periodic GUI freezes when this used
    to run inline in _update().
    """
    global _prev_snapshot, _prev_obj_count, _peak_rss_mb
    global _last_thread_count
    global _dbg_render_uploads, _dbg_render_pts_total, _dbg_render_ms_total
    global _dbg_revs_seen, _dbg_revs_seen_s2, _dbg_update_ticks, _dbg_render_ticks

    rss_mb = _process.memory_info().rss / 1024 / 1024 if _process else float("nan")
    if rss_mb > _peak_rss_mb:
        _peak_rss_mb = rss_mb

    print("=" * 60)
    print("DEBUG REPORT")
    print(f"RSS            : {rss_mb:.1f} MB")
    print(f"Peak RSS       : {_peak_rss_mb:.1f} MB")

    if _HAS_RESOURCE:
        ru_max = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"OS max RSS     : {ru_max} KB")

    thread_count = threading.active_count()
    print(f"Threads        : {thread_count}"
          + ("  ← CHANGED" if thread_count != _last_thread_count else ""))
    _last_thread_count = thread_count

    objs = len(gc.get_objects())
    print(f"Python objs    : {objs}  (Δ {objs - _prev_obj_count:+d})")
    _prev_obj_count = objs

    with _pts_lock:
        n_pts = _ring_count
    print(f"Ring count     : {n_pts:,} / {RING_CAPACITY:,}")
    print(f"S2 queue depth : {_lidar_q_s2.qsize():,}")
    print(f"Latest rev len : {len(_latest_rev)}")

    # Rates over this interval — much more readable than per-event prints
    print(f"Update ticks   : {_dbg_update_ticks}  ({_dbg_update_ticks/DEBUG_INTERVAL_SEC:.1f}/s)")
    print(f"Render dispatch: {_dbg_render_ticks}  ({_dbg_render_ticks/DEBUG_INTERVAL_SEC:.1f}/s)")
    if _dbg_render_uploads:
        print(f"GPU uploads    : {_dbg_render_uploads}  "
              f"({_dbg_render_pts_total:,} pts total, "
              f"avg {(_dbg_render_ms_total/_dbg_render_uploads):.1f} ms/upload)")
    else:
        print("GPU uploads    : 0")
    print(f"LiDAR revs S1  : {_dbg_revs_seen}")
    print(f"LiDAR revs S2  : {_dbg_revs_seen_s2}")

    # reset interval counters
    _dbg_render_uploads = 0
    _dbg_render_pts_total = 0
    _dbg_render_ms_total = 0.0
    _dbg_revs_seen = 0
    _dbg_revs_seen_s2 = 0
    _dbg_update_ticks = 0
    _dbg_render_ticks = 0

    gc.collect()
    snapshot = tracemalloc.take_snapshot()
    if _prev_snapshot is not None:
        print("\nTop memory growth (by line):")
        for stat in snapshot.compare_to(_prev_snapshot, "lineno")[:10]:
            print(f"  {stat}")
    _prev_snapshot = snapshot

    # Divergence check: if RSS keeps climbing over many reports while
    # Python objs / tracemalloc stay flat, the growth is native-side
    # (YDLidar SDK, Qt, or the OpenGL driver), not in Python.
    print("=" * 60)


def _debug_thread():
    """
    Background thread that paces print_debug() calls every DEBUG_INTERVAL_SEC.
    Runs entirely off the GUI thread — see module docstring / DEBUG
    INSTRUMENTATION header for why this moved off _update().
    """
    while not _stop_debug.is_set():
        if _stop_debug.wait(timeout=DEBUG_INTERVAL_SEC):
            break
        print_debug()


# ══════════════════════════════════════════════════════════════════════════════
#  PORT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _detect_port(prefer: list[str]) -> str:
    for p in prefer:
        if os.path.exists(p):
            return p
    candidates = sorted(
        glob.glob("/dev/ydlidar*") +
        glob.glob("/dev/ttyUSB*") +
        glob.glob("/dev/ttyACM*")
    )
    return candidates[0] if candidates else "/dev/ttyUSB0"

def detect_lidar_port() -> str:
    return _detect_port(["/dev/ydlidar", "/dev/ttyUSB0", "/dev/ttyUSB1"])

def detect_esp32_port(lidar_port: str) -> str:
    candidates = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    for c in candidates:
        if c != lidar_port:
            return c
    return "/dev/ttyUSB1"


# ══════════════════════════════════════════════════════════════════════════════
#  YDLIDAR
#  Fan-out to _latest_rev (Stage 1) and _lidar_q_s2 (Stage 2).
#  _latest_rev_id incremented once per revolution so _update() can
#  detect new data without duplicating scans.
# ══════════════════════════════════════════════════════════════════════════════

def _lidar_thread():
    global _latest_rev, _latest_rev_id, _dbg_revs_seen

    if not _SDK_OK:
        _lidar_status["error"] = (
            "YDLidar SDK not found.\n"
            "  cd ~/YDLidar-SDK && mkdir -p build && cd build\n"
            "  cmake .. && make -j4 && sudo make install"
        )
        return

    port = detect_lidar_port()
    _lidar_status["port"] = port

    laser = ydlidar.CYdLidar()
    laser.setlidaropt(ydlidar.LidarPropSerialPort,      port)
    laser.setlidaropt(ydlidar.LidarPropSerialBaudrate,  LIDAR_BAUD)
    laser.setlidaropt(ydlidar.LidarPropLidarType,       LIDAR_TYPE)
    laser.setlidaropt(ydlidar.LidarPropDeviceType,      LIDAR_DEV_TYPE)
    laser.setlidaropt(ydlidar.LidarPropSingleChannel,   LIDAR_SINGLE_CH)
    laser.setlidaropt(ydlidar.LidarPropFixedResolution, LIDAR_FIXED_RES)
    laser.setlidaropt(ydlidar.LidarPropAutoReconnect,   LIDAR_AUTO_RECON)
    laser.setlidaropt(ydlidar.LidarPropMinRange,        LIDAR_MIN_RANGE)
    laser.setlidaropt(ydlidar.LidarPropMaxRange,        LIDAR_MAX_RANGE)
    laser.setlidaropt(ydlidar.LidarPropMinAngle,        LIDAR_MIN_ANGLE)
    laser.setlidaropt(ydlidar.LidarPropMaxAngle,        LIDAR_MAX_ANGLE)
    laser.setlidaropt(ydlidar.LidarPropScanFrequency,   LIDAR_FREQ)

    if not laser.initialize():
        _lidar_status["error"] = f"Init failed: {laser.DescribeError()}"
        return
    if not laser.turnOn():
        _lidar_status["error"] = f"Turn-on failed: {laser.DescribeError()}"
        laser.disconnecting(); return

    _lidar_status["connected"] = True
    _lidar_status["error"]     = None

    scan = ydlidar.LaserScan()
    while not _stop_lidar.is_set():
        if not laser.doProcessSimple(scan):
            # Small sleep prevents this from becoming a tight busy-loop (and
            # starving the GIL / other threads) on a noisy serial link where
            # this branch is hit continuously (e.g. repeated checksum errors).
            time.sleep(0.005)
            continue

        pts: list[tuple[float, float]] = []
        for pt in scan.points:
            if LIDAR_MIN_RANGE <= pt.range <= LIDAR_MAX_RANGE:
                a = math.degrees(pt.angle) % 360.0
                pts.append((a, pt.range))
                _lidar_status["total"] += 1
                try:
                    _lidar_q_s2.put_nowait((a, pt.range))
                except queue.Full:
                    pass

        if pts:
            with _latest_rev_lock:
                _latest_rev   = pts       # atomic replace
                _latest_rev_id += 1       # signal that a new revolution arrived
            if DEBUG:
                _dbg_revs_seen += 1

    laser.turnOff()
    laser.disconnecting()
    _lidar_status["connected"] = False


# ══════════════════════════════════════════════════════════════════════════════
#  RING BUFFER
#  Pre-allocated; _ring_write() never calls any numpy allocator in the
#  steady-state (split-write path uses pre-existing buffer slices).
#
#  NOTE: _ring_view() calls np.concatenate() only when the buffer has
#  wrapped, which allocates a temporary array for the GPU upload.
#  This is unavoidable without a double-buffer scheme, and happens at
#  most RENDER_HZ times/sec, so it is not a hot-path concern.
# ══════════════════════════════════════════════════════════════════════════════

def _ring_write(pts: np.ndarray):
    """Write (N,3) float32 into the ring buffer. Caller must hold _pts_lock."""
    global _ring_head, _ring_count, _data_version
    n = len(pts)
    if n == 0: return
    if n >= RING_CAPACITY:
        pts = pts[-RING_CAPACITY:]; n = RING_CAPACITY

    space = RING_CAPACITY - _ring_head
    if n <= space:
        _ring_buf[_ring_head : _ring_head + n] = pts
    else:
        _ring_buf[_ring_head:] = pts[:space]
        _ring_buf[:n - space]  = pts[space:]

    _ring_head  = (_ring_head + n) % RING_CAPACITY
    _ring_count = min(_ring_count + n, RING_CAPACITY)
    _data_version += 1


def _ring_view() -> np.ndarray:
    """
    Return ordered (oldest→newest) view of valid ring contents.
    Caller must hold _pts_lock.
    When the buffer has not yet wrapped: returns a no-copy slice.
    When wrapped: returns np.concatenate result (one allocation).
    """
    if _ring_count < RING_CAPACITY:
        return _ring_buf[:_ring_count]
    return np.concatenate([_ring_buf[_ring_head:], _ring_buf[:_ring_head]], axis=0)


def _ring_clear():
    """Reset the ring buffer. Caller must hold _pts_lock."""
    global _ring_head, _ring_count, _data_version
    _ring_head = _ring_count = 0
    _data_version += 1


# ══════════════════════════════════════════════════════════════════════════════
#  ESP32
# ══════════════════════════════════════════════════════════════════════════════

def _esp32_send(cmd: str):
    global _esp32_ser
    with _esp32_lock:
        if _esp32_ser and _esp32_ser.is_open:
            _esp32_ser.write((cmd.strip() + "\n").encode())


def _esp32_thread():
    """
    FIX: port is now re-detected on every reconnect attempt (inside the
    while loop), not once before it. Previously, the port was computed a
    single time from _lidar_status.get("port", "") at thread start — if
    this thread ran before the LiDAR thread had set its port, that read
    an empty string, and detect_esp32_port() could return the LiDAR's own
    port by mistake. Every retry after that kept reusing the same wrong
    port forever, so the ESP32 never connected.
    """
    global _esp32_ser

    while not _stop_esp32.is_set():
        # Re-detect every attempt — cheap, and the only way to recover if
        # the first detection raced the LiDAR thread or the ESP32 was
        # unplugged/replugged into a different /dev/ttyUSBx.
        port = detect_esp32_port(_lidar_status.get("port", ""))
        _esp32_status["port"] = port

        ser = None
        try:
            ser = serial.Serial(port, ESP32_BAUD, timeout=1.0)
            with _esp32_lock: _esp32_ser = ser
            _esp32_status.update(connected=True, error=None)

            while not _stop_esp32.is_set():
                raw = ser.readline()
                if not raw: continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line: continue

                if DEBUG and not line.startswith("ANGLE:"):
                    # ANGLE: streams continuously while moving — everything
                    # else (READY/MOVING/DONE/BUSY/ERROR/STATE) is rare
                    # enough to print without flooding the terminal, and is
                    # exactly what tells you whether HOME/NEXT are actually
                    # being acknowledged by the firmware.
                    print(f"[ESP32] << {line}")

                if   line == "READY":           _esp32_status["state"] = "IDLE"; _esp32_send(f"SETSTEP:{SCAN_INCREMENT:.3f}")
                elif line == "MOVING":          _esp32_status["state"] = "MOVING"
                elif line.startswith("ANGLE:"):
                    try: _esp32_status["angle"] = float(line[6:])
                    except ValueError: pass
                elif line == "DONE":            _esp32_status["state"] = "IDLE"
                elif line == "BUSY":            _esp32_status["state"] = "BUSY"
                elif line.startswith("ERROR:"): _esp32_status["error"] = line
                elif line.startswith("STATE:"): _esp32_status["state"] = line

        except Exception as e:
            _esp32_status.update(connected=False, error=str(e))
            with _esp32_lock: _esp32_ser = None
            time.sleep(2.0)
        finally:
            # Explicit close even on abrupt disconnect (e.g. readline() raising
            # mid-loop) — don't rely on refcounting/__del__ to release the fd.
            if ser is not None:
                try: ser.close()
                except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
#  SCAN CONTROLLER  (Stage 2)
# ══════════════════════════════════════════════════════════════════════════════

def _wait_for_done(timeout: float = ESP32_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _esp32_status.get("state") == "IDLE":
            return True
        time.sleep(0.01)
    return False


def _capture_one_revolution() -> list[tuple[float, float]]:
    """
    Phase A — sync to revolution start (wait for angle wrap > 300° → < 60°).
    Phase B — capture exactly one complete revolution.
    Uses _lidar_q_s2 only; Stage 1 is completely unaffected.
    """
    deadline = time.time() + 5.0

    prev = None
    while time.time() < deadline:   # Phase A: sync
        try: ang, rng = _lidar_q_s2.get(timeout=0.05)
        except queue.Empty: continue
        if prev is not None and prev > 300.0 and ang < 60.0: break
        prev = ang
    if time.time() >= deadline: return []

    pts: list[tuple[float, float]] = []
    prev = None
    while time.time() < deadline:   # Phase B: capture
        try: ang, rng = _lidar_q_s2.get(timeout=0.05)
        except queue.Empty: continue
        if prev is not None and prev > 300.0 and ang < 60.0: break
        pts.append((ang, rng)); prev = ang
    return pts


def _flush_s2_queue():
    while not _lidar_q_s2.empty():
        try: _lidar_q_s2.get_nowait()
        except queue.Empty: break


def _scan_loop():
    global _scan_active, _dbg_revs_seen_s2
    for _ in range(50):
        if _lidar_status["connected"] and _esp32_status["connected"]: break
        time.sleep(0.1)
    if not (_lidar_status["connected"] and _esp32_status["connected"]):
        _esp32_status["error"] = "Scan aborted: LiDAR/ESP32 not both connected."
        with _scan_lock: _scan_active = False
        return

    _esp32_status["state"] = "MOVING"
    if DEBUG: print("[scan] >> HOME")
    _esp32_send("HOME")
    if not _wait_for_done(timeout=5.0):
        # BUG FIX: previously this result was discarded and the loop below
        # ran anyway, immediately sending NEXT while the firmware may still
        # have been mid-HOME. A simple embedded serial state machine often
        # just ignores commands received while busy — so every NEXT after
        # an unconfirmed HOME could get silently dropped, meaning the motor
        # never moves again even though the rest of the app (Stage-1 LiDAR
        # feed, render loop, FPS counter) keeps running normally, since
        # those are all independent of Stage 2 / the stepper.
        _esp32_status["error"] = "HOME did not complete — aborting scan. Check ESP32 firmware/wiring."
        if DEBUG: print("[scan] HOME timed out — aborting scan instead of sending NEXT anyway")
        with _scan_lock: _scan_active = False
        return
    if DEBUG: print("[scan] HOME confirmed (state=IDLE)")

    consecutive_timeouts = 0
    while True:
        with _scan_lock:
            if not _scan_active: break
        if not _esp32_status["connected"]:
            time.sleep(0.5); continue

        _esp32_status["state"] = "MOVING"
        if DEBUG: print("[scan] >> NEXT")
        _esp32_send("NEXT")
        if not _wait_for_done():
            consecutive_timeouts += 1
            _esp32_status["error"] = f"Timeout waiting for DONE ({consecutive_timeouts} in a row)"
            if DEBUG: print(f"[scan] NEXT timed out ({consecutive_timeouts} in a row)")
            # BUG FIX: previously this just slept 0.5s and looped straight
            # back to sending another NEXT — stacking a second unconfirmed
            # command on top of the first. If the firmware can't handle
            # overlapping commands, that permanently wedges it (every
            # subsequent NEXT keeps getting ignored). After repeated
            # timeouts, re-HOME instead of continuing to hammer NEXT — this
            # gives the firmware a clean command to resync on, rather than
            # assuming it's still safe to send NEXT into an unknown state.
            if consecutive_timeouts >= 3:
                _esp32_status["error"] = "Repeated NEXT timeouts — re-homing."
                if DEBUG: print("[scan] 3+ consecutive NEXT timeouts — re-homing")
                _esp32_status["state"] = "MOVING"
                _esp32_send("HOME")
                _wait_for_done(timeout=5.0)
                consecutive_timeouts = 0
            time.sleep(0.5); continue
        consecutive_timeouts = 0

        rotation_deg = _esp32_status["angle"]
        _flush_s2_queue()
        rev_pts = _capture_one_revolution()
        if not rev_pts: continue

        if DEBUG:
            _dbg_revs_seen_s2 += 1

        arr = _project_array(rev_pts, rotation_deg)
        with _pts_lock: _ring_write(arr)
        _cloud_dirty.set()
        _request_render()

    with _scan_lock: _scan_active = False


def start_scan():
    global _scan_active
    with _scan_lock:
        if _scan_active: return
        _scan_active = True
    threading.Thread(target=_scan_loop, daemon=True).start()


def stop_scan():
    global _scan_active
    with _scan_lock: _scan_active = False


# ══════════════════════════════════════════════════════════════════════════════
#  PROJECTION  (vectorised — no Python loop over points)
# ══════════════════════════════════════════════════════════════════════════════

def _project_array(pts: list[tuple[float, float]],
                   rotation_deg: float = 0.0) -> np.ndarray:
    if not pts:
        return np.empty((0, 3), dtype=np.float32)
    arr   = np.asarray(pts, dtype=np.float32)      # (N,2): angle, range
    a     = np.radians(arr[:, 0])
    d     = arr[:, 1]
    cos_a = np.cos(a)
    r     = math.radians(rotation_deg)
    x     = d * cos_a * math.cos(r)
    y     = d * np.sin(a)
    z     = d * cos_a * math.sin(r)
    return np.column_stack([x, y, z]).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE / LOAD
# ══════════════════════════════════════════════════════════════════════════════

def _ts()         -> str: return datetime.now().strftime("%Y%m%d_%H%M%S")
def _fp(ext: str) -> str: return os.path.join(SAVES_DIR, f"scan_{_ts()}.{ext}")

def _snapshot() -> np.ndarray:
    with _pts_lock: return _ring_view().copy()

def save_ply(path: str | None = None) -> str | None:
    pts = _snapshot()
    if not len(pts): return None
    path = path or _fp("ply")
    h = (f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n"
         "property float x\nproperty float y\nproperty float z\nend_header\n")
    with open(path, "w") as f: f.write(h); np.savetxt(f, pts, fmt="%.6f")
    return path

def save_pcd(path: str | None = None) -> str | None:
    pts = _snapshot()
    if not len(pts): return None
    path = path or _fp("pcd"); n = len(pts)
    h = (f"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n"
         f"COUNT 1 1 1\nWIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
         f"POINTS {n}\nDATA ascii\n")
    with open(path, "w") as f: f.write(h); np.savetxt(f, pts, fmt="%.6f")
    return path

def save_xyz(path: str | None = None) -> str | None:
    pts = _snapshot()
    if not len(pts): return None
    path = path or _fp("xyz")
    np.savetxt(path, pts, fmt="%.6f"); return path

def save_csv(path: str | None = None) -> str | None:
    pts = _snapshot()
    if not len(pts): return None
    path = path or _fp("csv")
    np.savetxt(path, pts, delimiter=",", header="x,y,z", comments="", fmt="%.6f")
    return path

def save_meta(scan_path: str):
    pts = _snapshot()
    m = {"timestamp": datetime.now().isoformat(), "point_count": int(len(pts)),
         "bounds": {ax: [float(pts[:,i].min()), float(pts[:,i].max())]
                    for i,ax in enumerate("xyz")} if len(pts) else {},
         "source": os.path.basename(scan_path)}
    with open(scan_path.rsplit(".",1)[0]+"_meta.json","w") as f: json.dump(m,f,indent=2)

def load_ply(path: str) -> int:
    raw, body = [], False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "end_header": body = True; continue
            if body:
                v = line.split()
                if len(v) >= 3: raw.append([float(x) for x in v[:3]])
    if not raw: return 0
    arr = np.array(raw, dtype=np.float32)
    with _pts_lock: _ring_clear(); _ring_write(arr)
    _cloud_dirty.set()
    _request_render()
    return len(arr)

def save_screenshot(path: str | None = None) -> str:
    path = path or _fp("png")
    img  = _canvas.render()
    if _HAS_PIL: Image.fromarray(img).save(path)
    else:
        from vispy.io import write_png; write_png(path, img)
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  VISPY
# ══════════════════════════════════════════════════════════════════════════════

def create_viewer():
    global _canvas, _view, _scatter, _axis, _grid
    _canvas = scene.SceneCanvas(
        keys="interactive", show=False,
        bgcolor="#08080f", size=(1280, 720), title="RPEA Scanner"
    )
    _view = _canvas.central_widget.add_view()
    _view.camera = "turntable"
    _view.camera.fov = 60; _view.camera.distance = 8
    _scatter = scene.visuals.Markers(); _view.add(_scatter)
    _axis = scene.visuals.XYZAxis(parent=_view.scene)
    _grid = scene.visuals.GridLines(color=(0.14,0.14,0.24,0.45), parent=_view.scene)
    _grid.transform = STTransform(scale=(1,1,1))

    # Hook into camera transform changes to track user interaction for LOD
    @_view.camera.transform.changed.connect
    def _on_cam_change(_event):
        global _cam_last_move_t
        _cam_last_move_t = time.perf_counter()


# ══════════════════════════════════════════════════════════════════════════════
#  FIX #3: event-driven rendering instead of a perpetual polling QTimer.
#
#  Previously a QTimer fired _render_tick() every 1000/RENDER_HZ ms forever,
#  checking _cloud_dirty even when nothing had changed. Replaced with:
#  new data / render-option change -> _request_render() -> (thread-safe Qt
#  signal) -> _on_render_requested() [runs on the Qt/main thread] ->
#  _refresh_scatter(), rate-limited to RENDER_HZ via a single-shot reschedule
#  rather than a repeating timer. When nothing is dirty, no timer runs at all.
#
#  _request_render() is safe to call from ANY thread (including the
#  background Stage-2 scan thread and the LiDAR thread) because it only
#  emits a Qt signal; Qt automatically delivers it via a queued connection
#  onto the thread that owns the receiving QObject (the main/GUI thread).
# ══════════════════════════════════════════════════════════════════════════════

class _RenderSignal(QObject):
    requested = pyqtSignal()

_render_signal: "_RenderSignal | None" = None   # created after QApplication exists

# Collapses redundant _request_render() calls into a single outstanding
# request, so a burst of calls (e.g. several LiDAR revolutions landing
# before the GUI thread catches up) doesn't queue up multiple renders.
_render_pending      = False
_render_pending_lock = threading.Lock()


def _request_render():
    """Thread-safe entry point. Call this any time _cloud_dirty is set."""
    global _render_pending
    if _render_signal is None:
        return
    with _render_pending_lock:
        if _render_pending:
            return
        _render_pending = True
    _render_signal.requested.emit()


def _on_render_requested():
    """Runs on the Qt/main thread. Self-throttles to RENDER_HZ via a
    single-shot reschedule instead of a perpetual repeating timer."""
    global _render_pending, _dbg_render_ticks
    if DEBUG:
        _dbg_render_ticks += 1   # now counts dispatch calls, not timer polls
    with _render_pending_lock:
        _render_pending = False
    if not _cloud_dirty.is_set():
        return
    now = time.perf_counter()
    elapsed = now - _last_render_t
    if elapsed >= _RENDER_INTV:
        _refresh_scatter()
    else:
        remaining_ms = max(0, int((_RENDER_INTV - elapsed) * 1000))
        QTimer.singleShot(remaining_ms, _on_render_requested)


def _height_colors(pts: np.ndarray) -> np.ndarray:
    if not len(pts): return np.empty((0,4), dtype=np.float32)
    z = pts[:,2]; t = (z - z.min()) / (z.max() - z.min() + 1e-9)
    r = np.clip(2.0*t - 1.0,               0.0, 1.0)
    g = np.clip(1.0 - np.abs(2.0*t-1.0)*2, 0.0, 1.0)
    b = np.clip(1.0 - 2.0*t,               0.0, 1.0)
    return np.column_stack([r, g, b, np.ones_like(r)]).astype(np.float32)


# Color cache for fix #2 below — see _refresh_scatter().
_color_cache_arr:     np.ndarray | None = None
_color_cache_version: int = -1
_color_cache_n:       int = -1


def _refresh_scatter():
    """
    Upload cloud to GPU. Called only from _render_tick() (main thread).
    Applies LOD: above LOD_THRESHOLD points, uses stride subsampling while
    the camera is moving; restores full detail after LOD_IDLE_SEC of stillness.

    Fix #1: _pts_lock is now held ONLY long enough to copy the current ring
    contents out. Everything after that — LOD striding, color computation,
    and the _scatter.set_data() GPU upload itself — runs unlocked. Before
    this fix, the lock was held across set_data(), which can take real time
    on hundreds of thousands of points; that blocked _ring_write() (called
    from both the LiDAR-driven _update() and the Stage 2 scan thread) for
    the full upload duration, stalling acquisition every single render.

    Fix #2: height colors are cached and only recomputed when the
    underlying ring data has actually changed (_data_version) or height
    coloring was toggled. In Live Stage-1 mode the ring is cleared and
    rewritten every revolution, so _data_version changes almost every
    render and this cache mostly won't hit — that's expected, the data
    really is different every frame there. Where it does help: dragging
    the point-size slider, toggling other rendering options, or any render
    triggered by something other than new points arriving (accumulate mode
    between revolutions, idle/LOD-only re-renders) now skips recomputing
    colors for a cloud that hasn't changed.
    """
    global _last_render_t, _dbg_render_uploads, _dbg_render_pts_total, _dbg_render_ms_total
    global _color_cache_arr, _color_cache_version, _color_cache_n
    t0 = time.perf_counter() if DEBUG else None
    _cloud_dirty.clear()
    _last_render_t = time.perf_counter()

    with _pts_lock:
        pts = _ring_view()      # may allocate when ring has wrapped
        n   = len(pts)
        if n == 0:
            return
        pts = pts.copy()        # copy while locked so the lock can be released
        version = _data_version
    # ── _pts_lock released here — LOD, color computation, and the GPU
    # upload below all run without holding it. ────────────────────────────

    if _height_col:
        if (_color_cache_arr is not None
                and _color_cache_version == version
                and _color_cache_n == n):
            colors_full = _color_cache_arr   # cache hit — data hasn't changed
        else:
            colors_full = _height_colors(pts)
            _color_cache_arr     = colors_full
            _color_cache_version = version
            _color_cache_n       = n
    else:
        colors_full = None

    # LOD: subsample while user is interacting. Colors are subsampled with
    # the identical stride so they stay index-aligned with the points.
    idle = (_last_render_t - _cam_last_move_t) > LOD_IDLE_SEC
    if n > LOD_THRESHOLD and not idle:
        display_pts = pts[::LOD_STRIDE]
        c = colors_full[::LOD_STRIDE] if colors_full is not None else "white"
    else:
        display_pts = pts
        c = colors_full if colors_full is not None else "white"

    _scatter.set_data(display_pts, edge_color=None, face_color=c, size=_point_size)

    if DEBUG:
        _dbg_render_uploads   += 1
        _dbg_render_pts_total += len(display_pts)
        _dbg_render_ms_total  += (time.perf_counter() - t0) * 1000.0


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

_C = dict(
    bg="#08080f", panel="#0e0e1a", border="#1a1a2e",
    acc="#6c63ff", acc2="#00d4ff", tx="#d4d4e8",
    muted="#44445a", ok="#00e676", warn="#ffab40", err="#ff5252",
)
_BTN = f"""
QPushButton {{
    background:#14142a; color:{_C['tx']}; border:1px solid {_C['border']};
    border-radius:3px; padding:5px 10px;
    font-family:'JetBrains Mono','Consolas',monospace; font-size:11px;
}}
QPushButton:hover    {{ background:#1e1e38; border-color:{_C['acc']}; color:#fff; }}
QPushButton:pressed  {{ background:{_C['acc']}; color:#fff; }}
QPushButton:disabled {{ background:#0e0e1a; color:{_C['muted']}; border-color:{_C['border']}; }}
"""
_BTN_GREEN = _BTN.replace("#14142a","#0a2a14").replace("#1e1e38","#0f3d1f")
_BTN_RED   = _BTN.replace("#14142a","#2a0a0a").replace("#1e1e38","#3d0f0f")
_LBL = f"color:{_C['tx']};  font-family:'Consolas',monospace; font-size:11px;"
_HDR = f"color:{_C['acc2']}; font-family:'Consolas',monospace; font-size:9px; font-weight:bold; letter-spacing:2px;"
_MUT = f"color:{_C['muted']}; font-family:'Consolas',monospace; font-size:10px;"
_CHK = f"color:{_C['tx']};  font-family:'Consolas',monospace; font-size:11px;"

def _sep() -> QFrame:
    w = QFrame(); w.setFrameShape(QFrame.Shape.HLine)
    w.setStyleSheet(f"border:0; border-top:1px solid {_C['border']};"); return w

def _hdr(t: str) -> QLabel:
    w = QLabel(t); w.setStyleSheet(_HDR); return w


class Sidebar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(245)
        self.setStyleSheet(f"background:{_C['panel']};")
        scroll = QScrollArea(self); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea{{border:none;background:{_C['panel']};}}"
            f"QScrollBar:vertical{{background:{_C['border']};width:4px;border-radius:2px;}}"
            f"QScrollBar::handle:vertical{{background:{_C['acc']};border-radius:2px;}}")
        inner = QWidget(); inner.setStyleSheet(f"background:{_C['panel']};")
        L = QVBoxLayout(inner); L.setContentsMargins(12,14,12,14); L.setSpacing(6)

        t1 = QLabel("RPEA SCANNER")
        t1.setStyleSheet(f"color:{_C['acc']};font-size:15px;font-weight:bold;"
                         "font-family:'Consolas',monospace;letter-spacing:1px;")
        L.addWidget(t1)
        self.lbl_stage = QLabel("Stage 1  ·  live mode")
        self.lbl_stage.setStyleSheet(_MUT); L.addWidget(self.lbl_stage)
        L.addWidget(_sep())

        # LiDAR
        L.addWidget(_hdr("LIDAR"))
        self.lbl_lidar = QLabel("connecting…")
        self.lbl_lidar.setStyleSheet(f"color:{_C['warn']};font-size:11px;font-family:'Consolas',monospace;")
        self.lbl_lidar.setWordWrap(True); L.addWidget(self.lbl_lidar)
        self.lbl_lport = QLabel("Port: —"); self.lbl_lport.setStyleSheet(_MUT); L.addWidget(self.lbl_lport)
        L.addWidget(_sep())

        # ESP32
        L.addWidget(_hdr("ESP32  /  PLATFORM"))
        self.lbl_esp32 = QLabel("not connected")
        self.lbl_esp32.setStyleSheet(f"color:{_C['muted']};font-size:11px;font-family:'Consolas',monospace;")
        self.lbl_esp32.setWordWrap(True); L.addWidget(self.lbl_esp32)
        self.lbl_eport  = QLabel("Port: —");  self.lbl_eport.setStyleSheet(_MUT);  L.addWidget(self.lbl_eport)
        self.lbl_angle  = QLabel("Angle: —"); self.lbl_angle.setStyleSheet(_LBL);  L.addWidget(self.lbl_angle)
        self.lbl_estate = QLabel("State: —"); self.lbl_estate.setStyleSheet(_MUT); L.addWidget(self.lbl_estate)
        L.addWidget(_sep())

        # Scan control
        L.addWidget(_hdr("SCAN CONTROL"))
        self.btn_start = QPushButton("▶️  Start 3-D scan"); self.btn_start.setStyleSheet(_BTN_GREEN)
        self.btn_stop  = QPushButton("■  Stop scan");      self.btn_stop.setStyleSheet(_BTN_RED)
        self.btn_home  = QPushButton("⌂  Home platform");  self.btn_home.setStyleSheet(_BTN)
        self.btn_stop.setEnabled(False)
        for b in (self.btn_start, self.btn_stop, self.btn_home): L.addWidget(b)
        L.addWidget(_sep())

        # Stage 1 mode
        L.addWidget(_hdr("STAGE 1 MODE"))
        self.btn_s1_live = QPushButton("🔴  Live  (latest sweep)"); self.btn_s1_live.setStyleSheet(_BTN_GREEN)
        self.btn_s1_acc  = QPushButton("📦  Accumulate");           self.btn_s1_acc.setStyleSheet(_BTN)
        for b in (self.btn_s1_live, self.btn_s1_acc): L.addWidget(b)
        L.addWidget(_sep())

        # Live stats
        L.addWidget(_hdr("LIVE"))
        self.lbl_fps   = QLabel("FPS: —");      self.lbl_fps.setStyleSheet(_LBL);   L.addWidget(self.lbl_fps)
        self.lbl_pts   = QLabel("Points: 0");    self.lbl_pts.setStyleSheet(_LBL);   L.addWidget(self.lbl_pts)
        self.lbl_rate  = QLabel("pts/s: 0");     self.lbl_rate.setStyleSheet(_LBL);  L.addWidget(self.lbl_rate)
        self.lbl_total = QLabel("Total raw: 0"); self.lbl_total.setStyleSheet(_MUT); L.addWidget(self.lbl_total)
        self.lbl_lod   = QLabel("");             self.lbl_lod.setStyleSheet(_MUT);   L.addWidget(self.lbl_lod)
        L.addWidget(_sep())

        # Scan capture
        L.addWidget(_hdr("SCAN CAPTURE"))
        self.btn_ply  = QPushButton("💾  Save .ply")
        self.btn_pcd  = QPushButton("💾  Save .pcd")
        self.btn_xyz  = QPushButton("💾  Save .xyz")
        self.btn_csv  = QPushButton("💾  Save .csv")
        self.btn_load = QPushButton("📂  Load .ply")
        self.btn_shot = QPushButton("📷  Screenshot")
        for b in (self.btn_ply,self.btn_pcd,self.btn_xyz,
                  self.btn_csv,self.btn_load,self.btn_shot):
            b.setStyleSheet(_BTN); L.addWidget(b)
        self.chk_auto = QCheckBox("Autosave every 60 s")
        self.chk_auto.setStyleSheet(_CHK); L.addWidget(self.chk_auto)
        self.lbl_save = QLabel("Last save: —")
        self.lbl_save.setStyleSheet(_MUT); self.lbl_save.setWordWrap(True); L.addWidget(self.lbl_save)
        L.addWidget(_sep())

        # Rendering
        L.addWidget(_hdr("RENDERING"))
        lps = QLabel("Point size"); lps.setStyleSheet(_LBL); L.addWidget(lps)
        self.sld = QSlider(Qt.Orientation.Horizontal)
        self.sld.setRange(1,8); self.sld.setValue(2)
        self.sld.setStyleSheet(f"accent-color:{_C['acc']};"); L.addWidget(self.sld)
        self.chk_ht = QCheckBox("Height coloring"); self.chk_ht.setChecked(True); self.chk_ht.setStyleSheet(_CHK); L.addWidget(self.chk_ht)
        self.chk_gr = QCheckBox("Ground grid");     self.chk_gr.setChecked(True); self.chk_gr.setStyleSheet(_CHK); L.addWidget(self.chk_gr)
        self.chk_ax = QCheckBox("XYZ axis");        self.chk_ax.setChecked(True); self.chk_ax.setStyleSheet(_CHK); L.addWidget(self.chk_ax)
        L.addWidget(_sep())

        # Camera
        L.addWidget(_hdr("CAMERA"))
        self.btn_rst = QPushButton("⟳  Reset"); self.btn_top = QPushButton("↑  Top")
        self.btn_frt = QPushButton("→  Front"); self.btn_sid = QPushButton("◁  Side")
        for b in (self.btn_rst,self.btn_top,self.btn_frt,self.btn_sid):
            b.setStyleSheet(_BTN); L.addWidget(b)
        L.addWidget(_sep())

        # Map
        L.addWidget(_hdr("MAP"))
        self.btn_clr = QPushButton("🗑  Clear map")
        self.btn_clr.setStyleSheet(_BTN.replace("#14142a","#200a0a").replace("#1e1e38","#300a0a"))
        L.addWidget(self.btn_clr)
        L.addStretch()
        self.lbl_st = QLabel("Initialising…")
        self.lbl_st.setStyleSheet(f"color:{_C['warn']};font-size:10px;font-family:'Consolas',monospace;")
        self.lbl_st.setWordWrap(True); L.addWidget(self.lbl_st)

        scroll.setWidget(inner)
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0); outer.addWidget(scroll)

    @staticmethod
    def _set_text(lbl: QLabel, text: str):
        """Skip setText() when the value hasn't changed — Qt still does
        layout/repaint bookkeeping on setText() even for an identical
        string, and _tick() calls this ~10 times every 150ms."""
        if lbl.text() != text:
            lbl.setText(text)

    def set_status(self, msg: str, color: str | None = None):
        color = color or _C["ok"]
        self._set_text(self.lbl_st, msg)
        style = f"color:{color};font-size:10px;font-family:'Consolas',monospace;"
        if self.lbl_st.styleSheet() != style:
            self.lbl_st.setStyleSheet(style)

    def _cl(self, lbl: QLabel, ok: bool, ok_t: str, fail_t: str):
        self._set_text(lbl, ok_t if ok else fail_t)
        style = f"color:{_C['ok'] if ok else _C['err']};font-size:11px;font-family:'Consolas',monospace;"
        if lbl.styleSheet() != style:
            lbl.setStyleSheet(style)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RPEA Scanner")
        self.setStyleSheet(f"background:{_C['bg']};")
        self.resize(1500, 820)
        root = QHBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        self.sb = Sidebar(); root.addWidget(self.sb)
        native = _canvas.native
        native.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(native)
        self._wire()
        t = QTimer(self); t.timeout.connect(self._tick); t.start(150)

    def _wire(self):
        sb = self.sb
        sb.btn_start.clicked.connect(self._on_start)
        sb.btn_stop.clicked.connect(self._on_stop)
        sb.btn_home.clicked.connect(lambda: _esp32_send("HOME"))
        sb.btn_s1_live.clicked.connect(lambda: self._set_s1(True))
        sb.btn_s1_acc.clicked.connect(lambda:  self._set_s1(False))
        sb.btn_ply.clicked.connect(self._save_ply)
        sb.btn_pcd.clicked.connect(lambda: self._notify(save_pcd()))
        sb.btn_xyz.clicked.connect(lambda: self._notify(save_xyz()))
        sb.btn_csv.clicked.connect(lambda: self._notify(save_csv()))
        sb.btn_load.clicked.connect(self._load)
        sb.btn_shot.clicked.connect(self._shot)
        sb.chk_auto.stateChanged.connect(self._toggle_auto)
        sb.sld.valueChanged.connect(self._pt_size)
        sb.chk_ht.stateChanged.connect(self._height)
        sb.chk_gr.stateChanged.connect(lambda s: setattr(_grid,"visible",bool(s)))
        sb.chk_ax.stateChanged.connect(lambda s: setattr(_axis,"visible",bool(s)))
        sb.btn_rst.clicked.connect(self._cam_reset)
        sb.btn_top.clicked.connect(lambda:[setattr(_view.camera,"elevation",90),setattr(_view.camera,"azimuth",0)])
        sb.btn_frt.clicked.connect(lambda:[setattr(_view.camera,"elevation",0), setattr(_view.camera,"azimuth",0)])
        sb.btn_sid.clicked.connect(lambda:[setattr(_view.camera,"elevation",0), setattr(_view.camera,"azimuth",90)])
        sb.btn_clr.clicked.connect(self._clear)

    def _set_s1(self, live: bool):
        global _s1_live_mode
        _s1_live_mode = live
        self.sb.btn_s1_live.setStyleSheet(_BTN_GREEN if live else _BTN)
        self.sb.btn_s1_acc.setStyleSheet(_BTN if live else _BTN_GREEN)
        self.sb.lbl_stage.setText("Stage 1  ·  live mode" if live else "Stage 1  ·  accumulate mode")
        self.sb.set_status("Live mode" if live else "Accumulate mode")

    def _on_start(self):
        if not _esp32_status["connected"]:
            self.sb.set_status("ESP32 not connected.", _C["warn"]); return
        start_scan()
        self.sb.btn_start.setEnabled(False); self.sb.btn_stop.setEnabled(True)
        self.sb.lbl_stage.setText("Stage 2  ·  3-D sweep")
        self.sb.set_status("3-D scan started")

    def _on_stop(self):
        stop_scan()
        self.sb.btn_start.setEnabled(True); self.sb.btn_stop.setEnabled(False)
        self.sb.lbl_stage.setText("Stage 1  ·  live mode" if _s1_live_mode else "Stage 1  ·  accumulate mode")
        self.sb.set_status("Scan stopped.", _C["warn"])

    def _notify(self, path: str | None):
        if not path: self.sb.set_status("No points to save.", _C["warn"]); return
        self.sb.lbl_save.setText(f"Last save: {os.path.basename(path)}")
        self.sb.set_status(f"Saved → {os.path.basename(path)}")

    def _save_ply(self):
        p = save_ply()
        if p: save_meta(p)
        self._notify(p)

    def _load(self):
        p,_ = QFileDialog.getOpenFileName(self,"Load PLY",SAVES_DIR,"PLY files (*.ply)")
        if p: self.sb.set_status(f"Loaded {load_ply(p):,} pts ← {os.path.basename(p)}", _C["acc2"])

    def _shot(self):
        self.sb.set_status(f"Screenshot → {os.path.basename(save_screenshot())}")

    def _toggle_auto(self, s):
        global _autosave_on
        _autosave_on = bool(s)
        self.sb.set_status("Autosave ON" if _autosave_on else "Autosave OFF",
                           _C["ok"] if _autosave_on else _C["muted"])

    def _pt_size(self, v):
        global _point_size; _point_size = v; _cloud_dirty.set(); _request_render()

    def _height(self, s):
        global _height_col; _height_col = bool(s); _cloud_dirty.set(); _request_render()

    def _cam_reset(self):
        _view.camera.set_range(x=(-5,5),y=(-5,5),z=(-5,5))
        _view.camera.distance=8; _view.camera.elevation=30; _view.camera.azimuth=45

    def _clear(self):
        with _pts_lock: _ring_clear()
        _scatter.set_data(np.zeros((1,3)),face_color=(0,0,0,0),size=1)
        _cloud_dirty.clear()
        self.sb.set_status("Map cleared.", _C["warn"])

    def _tick(self):
        sb = self.sb
        sb._cl(sb.lbl_lidar, _lidar_status["connected"], "connected ✓",
               _lidar_status.get("error") or "waiting…")
        sb._set_text(sb.lbl_lport, f"Port: {_lidar_status['port']}")
        sb._cl(sb.lbl_esp32, _esp32_status["connected"], "connected ✓",
               _esp32_status.get("error") or "not connected")
        sb._set_text(sb.lbl_eport, f"Port: {_esp32_status['port']}")
        sb._set_text(sb.lbl_angle, f"Angle: {_esp32_status['angle']:.2f}°")
        sb._set_text(sb.lbl_estate, f"State: {_esp32_status['state']}")

        with _pts_lock: n_pts = _ring_count
        with _scan_lock: scanning = _scan_active

        sb.btn_start.setEnabled(not scanning and _esp32_status["connected"])
        sb.btn_stop.setEnabled(scanning)
        sb.btn_home.setEnabled(_esp32_status["connected"])
        sb._set_text(sb.lbl_pts, f"Points: {n_pts:,}")
        sb._set_text(sb.lbl_total, f"Total raw: {_lidar_status['total']:,}")
        if _fps_buf: sb._set_text(sb.lbl_fps, f"FPS: {sum(_fps_buf)/len(_fps_buf):.1f}")

        # LOD status indicator
        now = time.perf_counter()
        idle = (now - _cam_last_move_t) > LOD_IDLE_SEC
        if n_pts > LOD_THRESHOLD and not idle:
            sb._set_text(sb.lbl_lod, f"LOD: 1/{LOD_STRIDE}  (moving)")
        elif n_pts > LOD_THRESHOLD:
            sb._set_text(sb.lbl_lod, "LOD: full  (idle)")
        else:
            sb._set_text(sb.lbl_lod, "")

        if scanning:                    sb.set_status(f"3-D scanning  ·  {_esp32_status['angle']:.1f}°")
        elif n_pts > 100:               sb.set_status("Live — rotate view with mouse")
        elif _lidar_status["connected"]:sb.set_status("LiDAR connected — waiting for points…", _C["warn"])
        elif _lidar_status["error"]:    sb.set_status(_lidar_status["error"][:70], _C["err"])


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
#
#  _update()      VisPy timer ~30 Hz — acquisition only, zero GPU work, and
#                 (as of this fix) zero debug-report work. It only increments
#                 a cheap counter now; print_debug() runs on _debug_thread().
#                 BUG FIX: compares _latest_rev_id to last_processed_id
#                 so each real revolution is written to the ring buffer
#                 exactly once, regardless of how fast _update() fires.
#
#  _render_tick() QTimer ~15 Hz — ONLY place that calls _refresh_scatter().
#                 Hard FPS ceiling via _RENDER_INTV check.
# ══════════════════════════════════════════════════════════════════════════════

_lt  = time.perf_counter()
_lrc = time.perf_counter()
_prc = 0
_win_ref      = None
_vispy_timer  = None


def _update(event):
    """Acquisition tick — no GPU work. Each real LiDAR revolution processed once."""
    global _last_autosave, _lt, _lrc, _prc, _dbg_update_ticks

    if DEBUG:
        _dbg_update_ticks += 1   # cheap increment only — the actual report
                                 # runs on _debug_thread(), never here

    now = time.perf_counter(); dt = now - _lt; _lt = now
    if dt > 0:
        _fps_buf.append(1.0/dt)
        if len(_fps_buf) > 60: _fps_buf.pop(0)

    with _scan_lock: s2 = _scan_active

    if not s2:
        # Grab latest revolution data and its ID atomically
        with _latest_rev_lock:
            rev    = _latest_rev
            rev_id = _latest_rev_id

        # KEY FIX: only process if this is a revolution we haven't seen yet.
        # _update.last_id is a function attribute initialised to -1 below.
        if rev and rev_id != _update.last_id:
            _update.last_id = rev_id
            arr = _project_array(rev, 0.0)   # vectorised, one allocation per rev

            with _pts_lock:
                if _s1_live_mode:
                    _ring_clear()        # live: replace
                _ring_write(arr)         # accumulate: append

            _prc += len(arr)
            _cloud_dirty.set()
            _request_render()

    if now - _lrc >= 1.0:
        if _win_ref: _win_ref.sb.lbl_rate.setText(f"pts/s: {int(_prc/(now-_lrc)):,}")
        _prc = 0; _lrc = now

    if _autosave_on and now - _last_autosave >= AUTOSAVE_SEC:
        p = save_ply()
        if p: save_meta(p)
        _last_autosave = now

# Initialise the function attribute used as persistent local state
_update.last_id = -1


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    if not _SDK_OK:
        print("=" * 60)
        print("ERROR: YDLidar Python SDK not found.")
        print("  cd ~/YDLidar-SDK && mkdir -p build && cd build")
        print("  cmake .. && make -j4 && sudo make install && sudo ldconfig")
        print("=" * 60)
        sys.exit(1)

    if not _HAS_SERIAL:
        print("WARNING: pyserial not installed — ESP32/Stage 2 unavailable.")

    print("=" * 60)
    print("  RPEA Scanner  —  Stage 1 + Stage 2")
    print(f"  Ring buffer : {RING_CAPACITY:,} pts  ({RING_CAPACITY*12//1024} KB)")
    print(f"  LOD kicks in: > {LOD_THRESHOLD:,} pts  (stride {LOD_STRIDE}x while moving)")
    print(f"  Render cap  : {RENDER_HZ} Hz")
    print(f"  Saves       : {SAVES_DIR}")
    print(f"  DEBUG mode  : {DEBUG}")
    print("=" * 60)

    threading.Thread(target=_lidar_thread, daemon=True).start()
    if _HAS_SERIAL:
        threading.Thread(target=_esp32_thread, daemon=True).start()
    if DEBUG:
        threading.Thread(target=_debug_thread, daemon=True).start()

    qt_app = QApplication.instance() or QApplication(sys.argv)
    create_viewer()   # must be after QApplication

    # Fix #3: create the render-request signal AFTER QApplication exists, so
    # its thread affinity is the GUI thread — connect BEFORE any other
    # thread can call _request_render().
    _render_signal = _RenderSignal()
    _render_signal.requested.connect(_on_render_requested)

    win = MainWindow(); _win_ref = win; win.show()

    _vispy_timer  = vapp.Timer(interval=0.033, connect=_update, start=True)

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        _vispy_timer.stop()
        print(f"Loaded {load_ply(sys.argv[1]):,} points from {sys.argv[1]}")

    def _shutdown():
        _stop_lidar.set(); _stop_esp32.set(); stop_scan(); _stop_debug.set()
    qt_app.aboutToQuit.connect(_shutdown)

    try:
        vapp.run()
    except Exception:
        traceback.print_exc()
