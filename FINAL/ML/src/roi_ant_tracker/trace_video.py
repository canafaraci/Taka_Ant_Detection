"""Draw a tracking run onto the original video: boxes, IDs and the full path of every ant.

Usage (from the repository root):
  python FINAL/pipeline.py trace-video -- --run FINAL/runs/track_rules_test5 [--scale 1.0] [--output trace.mp4]

Reads tracks.csv, feeder_events.csv and track_manifest.json in the run folder; re-reads the video frames
the run covered. --scale 1.0 keeps full resolution (4K), 0.5 gives 1080p (4x smaller file).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def colour(i: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(int(i) * 7919)
    c = rng.integers(60, 255, 3)
    return int(c[0]), int(c[1]), int(c[2])


def passthrough_flag(ffmpeg: str) -> list[str]:
    """Keep only the selected frames: '-fps_mode passthrough' on newer ffmpeg, '-vsync 0' on older."""
    help_text = subprocess.run([ffmpeg, "-hide_banner", "-h", "full"], capture_output=True, text=True).stdout
    return ["-fps_mode", "passthrough"] if "-fps_mode" in help_text else ["-vsync", "0"]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="Tracking run folder (has tracks.csv and track_manifest.json).")
    p.add_argument("--output", help="Default: <run>/trace.mp4 (or trace_4k.mp4 at full scale).")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--video", help="Override the video path stored in the manifest.")
    p.add_argument("--progress-every", type=int, default=500)
    return p


def main() -> None:
    a = build_parser().parse_args()
    run = Path(a.run)
    man = json.loads((run / "track_manifest.json").read_text())
    video = a.video or man["video"]
    if not Path(video).exists():                      # manifest paths are relative to the repository root
        alt = Path(man.get("repo_root", ".")) / video
        video = str(alt) if alt.exists() else video
    fps = float(man["fps"])
    step = int(man.get("frame_step", 1))           # tracking runs may use every Nth frame
    W, H = man["size"]
    w, h = int(round(W * a.scale)) // 2 * 2, int(round(H * a.scale)) // 2 * 2
    zcfg = man.get("zones_config") or {}
    out = Path(a.output) if a.output else run / ("trace_4k.mp4" if a.scale >= 0.99 else "trace.mp4")

    by_frame: dict[int, list[dict]] = defaultdict(list)
    for r in read_csv(run / "tracks.csv"):
        by_frame[int(r["frame"])].append(r)
    if not by_frame:
        sys.exit(f"No tracks in {run}")
    fe = defaultdict(list)
    for r in read_csv(run / "feeder_events.csv"):
        fe[int(r["frame"])].append(r)
    f0, f1 = min(by_frame), max(by_frame)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("ffmpeg is needed (Mac: brew install ffmpeg).")
    vf = (f"select=not(mod(n\\,{step}))," if step > 1 else "") + f"scale={w}:{h}"
    dec = subprocess.Popen([ffmpeg, "-v", "error", "-ss", f"{f0 / fps:.3f}", "-t", f"{(f1 - f0 + 1) / fps:.3f}", "-i", video,
                            "-vf", vf, *(passthrough_flag(ffmpeg) if step > 1 else []), "-pix_fmt", "bgr24", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen([ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps / step),
                            "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    s = a.scale
    lw = max(1, int(round(3 * s)))
    fs = 1.2 * s / 0.5 * 0.5 if s < 0.99 else 1.2
    trails: dict[int, list] = defaultdict(list)
    tcache: dict[int, np.ndarray] = {}
    feeders = {f["id"]: dict(f, n_in=0, n_out=0) for f in zcfg.get("feeders", [])}
    n = 0
    n_total = (f1 - f0) // step + 1
    print(f"[trace] {n_total} frames -> {out} ({w}x{h})", flush=True)
    while True:
        buf = dec.stdout.read(w * h * 3)
        if len(buf) < w * h * 3:
            break
        im = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
        f = f0 + n * step
        n += 1
        for e in fe.get(f, []):
            if e["feeder"] in feeders:
                feeders[e["feeder"]]["n_in" if e["event"] == "in" else "n_out"] += 1
        rows = by_frame.get(f, [])
        present = {}
        for r in rows:
            tid = int(r["track_id"])
            present[tid] = r
            if r["observed"] == "1" and r["state"] == "active":
                trails[tid].append((float(r["cx"]) * s, float(r["cy"]) * s))
        for tid, pts in trails.items():
            if len(pts) > 1:
                arr = tcache.get(tid)
                if arr is None or len(arr) != len(pts):
                    arr = np.array(pts, np.int32)
                    tcache[tid] = arr
                cv2.polylines(im, [arr], False, colour(tid), lw if tid in present else max(1, lw - 1), cv2.LINE_AA)
        st = defaultdict(int)
        for tid, r in present.items():
            st[r["state"]] += 1
            col = colour(tid)
            cx, cy = int(float(r["cx"]) * s), int(float(r["cy"]) * s)
            x1, y1, x2, y2 = (int(float(r[k]) * s) for k in ("x1", "y1", "x2", "y2"))
            pad = int(8 * s)
            if r["state"] == "active":
                cv2.rectangle(im, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), col, lw, cv2.LINE_AA)
                cv2.putText(im, str(tid), (x2 + pad, y1), cv2.FONT_HERSHEY_SIMPLEX, fs, col, lw, cv2.LINE_AA)
            elif r["state"] == "merged":
                cv2.putText(im, f"+{tid}", (x2 + pad, y1 + int(40 * s)), cv2.FONT_HERSHEY_SIMPLEX, fs, col, lw, cv2.LINE_AA)
            elif r["state"] == "lost":
                cv2.circle(im, (cx, cy), int(22 * s), col, max(1, lw - 1), cv2.LINE_AA)
                cv2.putText(im, f"{tid}?", (cx + int(26 * s), cy), cv2.FONT_HERSHEY_SIMPLEX, fs, col, lw, cv2.LINE_AA)
            elif r["state"] == "dead":
                cv2.drawMarker(im, (cx, cy), (0, 0, 0), cv2.MARKER_TILTED_CROSS, int(36 * s), lw + 1)
                cv2.putText(im, f"{tid} dead", (cx + int(22 * s), cy), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.85, (0, 0, 0), lw, cv2.LINE_AA)
        for fz in feeders.values():
            c = (int(fz["cx"] * s), int(fz["cy"] * s))
            cv2.circle(im, c, int(fz["cup_radius"] * s), (0, 140, 255), lw, cv2.LINE_AA)
            inside = max(fz["n_in"] - fz["n_out"], 0)
            cv2.putText(im, f"inside {inside}  in {fz['n_in']}  out {fz['n_out']}",
                        (c[0] - int(230 * s), c[1] + int((fz["cup_radius"] + 60) * s)), cv2.FONT_HERSHEY_SIMPLEX,
                        fs * 1.15, (0, 110, 255), lw + 1, cv2.LINE_AA)
        if "nest" in zcfg:
            nz = zcfg["nest"]
            cv2.circle(im, (int(nz["cx"] * s), int(nz["cy"] * s)), int(nz["enter_radius"] * s), (200, 0, 200), lw, cv2.LINE_AA)
        cv2.putText(im, f"t = {f / fps:.1f} s   ants on tray {st['active'] + st['merged'] + st['lost']} (lost {st['lost']})"
                        f"   dead {st['dead']}", (int(60 * s), int(90 * s)), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.8, (0, 0, 255),
                    lw + 2, cv2.LINE_AA)
        enc.stdin.write(im.tobytes())
        if n % a.progress_every == 0:
            print(f"[trace] {n}/{n_total} frames", flush=True)
    enc.stdin.close()
    enc.wait()
    print(f"[trace] done: {n} frames -> {out}", flush=True)


if __name__ == "__main__":
    main()
