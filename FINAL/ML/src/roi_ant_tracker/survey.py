"""Whole-tray survey of a long video.

Samples one frame every --step-seconds, cuts the tray into overlapping tiles (the size the
detector was trained on), runs the detector on the tiles that could contain an ant, merges
detections across tile overlaps, labels each detection with an area (feeder_left, feeder_right,
nest_entrance, tray) and drops detections inside ignore zones (the nest-opening clearance).
Output: one CSV per hour of video (resumable), plus survey_manifest.json.

A tile is skipped when it contains nothing darker than its own median by --dark-gate grey
levels (ants are dark on the white tray). Tiles overlapping a labelled area are always run.
Use --dark-gate 0 to run every tile.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms

from .model import choose_device, load_detector

FIELDS = ["frame", "t_sec", "x1", "y1", "x2", "y2", "cx", "cy", "score", "area"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Detect ants across the whole tray every N seconds of a long video.")
    p.add_argument("--video", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True, help="Output folder (one CSV per hour; existing complete hours are skipped).")
    p.add_argument("--mask", required=True, help="Tray mask PNG (white = survey area), same size as the video frames.")
    p.add_argument("--zones", help="Ignore-zone JSON (circles); detections centred inside are dropped.")
    p.add_argument("--areas", help="Area JSON (rectangles) used to label detections; everything else is 'tray'.")
    p.add_argument("--step-seconds", type=float, default=25.0)
    p.add_argument("--start-hour", type=int, default=0)
    p.add_argument("--end-hour", type=int, help="Exclusive. Default: end of video.")
    p.add_argument("--start-seconds", type=float, help="Survey only from this time (overrides hour blocks; for pilots).")
    p.add_argument("--duration-seconds", type=float, help="With --start-seconds: how long to survey.")
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--tile", type=int, default=320)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--dark-gate", type=float, default=30.0)
    p.add_argument("--nms-iou", type=float, default=0.3)
    p.add_argument("--device", default="auto")
    p.add_argument("--decoder", choices=["auto", "ffmpeg", "opencv"], default="auto")
    p.add_argument("--keyframe-interval", type=int, default=0, help="Frames between keyframes (0 = detect with ffprobe).")
    p.add_argument("--progress-every", type=int, default=20)
    return p


# ---------------------------------------------------------------- geometry

def tile_origins(mask: np.ndarray, tile: int, overlap: int, min_cover: float = 0.02) -> list[tuple[int, int]]:
    h, w = mask.shape
    stride = tile - overlap
    xs = list(range(0, w - tile + 1, stride))
    ys = list(range(0, h - tile + 1, stride))
    if xs[-1] != w - tile:
        xs.append(w - tile)
    if ys[-1] != h - tile:
        ys.append(h - tile)
    return [(x, y) for y in ys for x in xs if mask[y:y + tile, x:x + tile].mean() >= min_cover]


def load_areas(path: str | None) -> list[dict]:
    if not path:
        return []
    return [a for a in json.loads(Path(path).read_text())["areas"] if a["type"] == "rect"]


def load_zones(path: str | None) -> list[dict]:
    if not path:
        return []
    return [z for z in json.loads(Path(path).read_text())["zones"] if z["type"] == "circle"]


def area_of(cx: float, cy: float, areas: list[dict]) -> str:
    for a in areas:
        if a["x"] <= cx < a["x"] + a["w"] and a["y"] <= cy < a["y"] + a["h"]:
            return a["id"]
    return "tray"


def in_zone(cx: float, cy: float, zones: list[dict]) -> bool:
    return any((cx - z["cx"]) ** 2 + (cy - z["cy"]) ** 2 <= z["r"] ** 2 for z in zones)


def tile_touches_area(x: int, y: int, tile: int, areas: list[dict]) -> bool:
    return any(not (x + tile <= a["x"] or a["x"] + a["w"] <= x or y + tile <= a["y"] or a["y"] + a["h"] <= y) for a in areas)


# ---------------------------------------------------------------- frames

def probe(video: str) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, n, w, h


def detect_keyframe_interval(video: str, fps: float) -> int:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time,flags",
         "-of", "csv=p=0", "-read_intervals", "%+120", video],
        capture_output=True, text=True,
    ).stdout.splitlines()
    times = []
    for line in out:
        parts = line.split(",")
        if len(parts) >= 2 and "K" in parts[1] and parts[0] not in ("", "N/A"):
            times.append(float(parts[0]))
    times = sorted(times)[:6]
    if len(times) < 3:
        return 0
    gaps = {round((b - a) * fps) for a, b in zip(times, times[1:])}
    return gaps.pop() if len(gaps) == 1 else 0


_PASSTHROUGH: list[str] | None = None


def _passthrough_flag(ffmpeg: str) -> list[str]:
    """'-fps_mode passthrough' on ffmpeg >= 5.1; '-vsync 0' on older builds (removed in newer ones)."""
    global _PASSTHROUGH
    if _PASSTHROUGH is None:
        help_text = subprocess.run([ffmpeg, "-hide_banner", "-h", "full"], capture_output=True, text=True).stdout
        _PASSTHROUGH = ["-fps_mode", "passthrough"] if "-fps_mode" in help_text else ["-vsync", "0"]
    return _PASSTHROUGH


def frames_ffmpeg(video, start_frame, end_frame, step, fps, w, h, keyframes_only):
    ffmpeg = shutil.which("ffmpeg")
    cmd = [ffmpeg, "-v", "error", "-nostdin"]
    if sys.platform == "darwin":
        cmd += ["-hwaccel", "videotoolbox"]
    elif sys.platform == "win32":
        cmd += ["-hwaccel", "auto"]          # NVIDIA / Intel / AMD video decoding when available, else CPU
    if keyframes_only:
        cmd += ["-skip_frame", "nokey"]
    cmd += ["-ss", f"{start_frame / fps:.3f}", "-t", f"{(end_frame - start_frame) / fps:.3f}", "-i", video]
    if not keyframes_only and step > 1:
        cmd += ["-vf", f"select=not(mod(n\\,{step}))"]
    cmd += [*_passthrough_flag(ffmpeg), "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3)
    size, i = w * h * 3, 0
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield start_frame + i * step, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            i += 1
    finally:
        proc.stdout.close()
        proc.wait()


def frames_opencv(video, start_frame, end_frame, step):
    cap = cv2.VideoCapture(video)
    try:
        if step >= 100:
            for f in range(start_frame, end_frame, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, frame = cap.read()
                if ok:
                    yield f, frame
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for f in range(start_frame, end_frame):
                if (f - start_frame) % step == 0:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    yield f, frame
                elif not cap.grab():
                    break
    finally:
        cap.release()


# ---------------------------------------------------------------- detection

def detect_frame(model, device, frame, origins, always, tile, gate, batch, thr, nms_iou):
    run = []
    for (x, y) in origins:
        crop = frame[y:y + tile, x:x + tile]
        if gate > 0 and not always[(x, y)]:
            g = cv2.blur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (3, 3))
            if float(np.median(g)) - float(g.min()) < gate:
                continue
        run.append((x, y, crop))
    boxes, scores = [], []
    with torch.inference_mode():
        for i in range(0, len(run), batch):
            chunk = run[i:i + batch]
            tensors = [torch.from_numpy(np.ascontiguousarray(c[2][:, :, ::-1])).permute(2, 0, 1).float().div(255.0).to(device)
                       for c in chunk]
            tensors += [torch.zeros_like(tensors[0])] * (batch - len(tensors))   # same batch shape every call
            for (x, y, _), out in zip(chunk, model(tensors)):
                keep = (out["labels"] == 1) & (out["scores"] >= thr)
                b = out["boxes"][keep].detach().cpu()
                if len(b):
                    b[:, [0, 2]] += x
                    b[:, [1, 3]] += y
                    boxes.append(b)
                    scores.append(out["scores"][keep].detach().cpu())
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), len(run)
    b, s = torch.cat(boxes), torch.cat(scores)
    k = nms(b, s, nms_iou)
    return b[k].numpy(), s[k].numpy(), len(run)


def main() -> None:
    a = build_parser().parse_args()
    fps, n_frames, w, h = probe(a.video)
    mask = (cv2.imread(a.mask, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    if mask.shape != (h, w):
        raise ValueError(f"Mask is {mask.shape[::-1]}, video is {(w, h)}")
    areas, zones = load_areas(a.areas), load_zones(a.zones)
    origins = tile_origins(mask, a.tile, a.overlap)
    always = {o: tile_touches_area(o[0], o[1], a.tile, areas) for o in origins}
    step = max(1, int(round(a.step_seconds * fps)))
    kf = a.keyframe_interval or detect_keyframe_interval(a.video, fps)
    use_ffmpeg = a.decoder == "ffmpeg" or (a.decoder == "auto" and shutil.which("ffmpeg"))
    keyframes_only = bool(use_ffmpeg and kf and step == kf)  # decode keyframes only (fast path)
    device = choose_device(a.device)
    model, _ = load_detector(a.checkpoint, device)
    model.roi_heads.score_thresh = a.score_threshold
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if a.start_seconds is not None:
        s0 = int(round(a.start_seconds * fps))
        s1 = min(n_frames, s0 + int(round((a.duration_seconds or 3600) * fps)))
        blocks = [(f"from_{int(a.start_seconds):06d}s", s0, s1)]
    else:
        last = a.end_hour if a.end_hour is not None else int(np.ceil(n_frames / fps / 3600))
        blocks = [(f"hour_{hh:03d}", int(hh * 3600 * fps), min(n_frames, int((hh + 1) * 3600 * fps))) for hh in range(a.start_hour, last)]
    manifest = dict(video=str(Path(a.video).resolve()), checkpoint=str(Path(a.checkpoint).resolve()), fps=fps, frames=n_frames,
                    size=[w, h], step_seconds=a.step_seconds, step_frames=step, keyframe_interval=kf,
                    decoder="ffmpeg" if use_ffmpeg else "opencv", keyframes_only=keyframes_only, device=str(device),
                    score_threshold=a.score_threshold, tile=a.tile, overlap=a.overlap, tiles_in_mask=len(origins),
                    tiles_always_run=int(sum(always.values())), dark_gate=a.dark_gate, nms_iou=a.nms_iou,
                    mask=str(Path(a.mask).resolve()), zones=a.zones, areas=a.areas, fields=FIELDS)
    (out / "survey_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[survey] {manifest['decoder']} keyframes_only={keyframes_only} step={step} frames ({a.step_seconds}s) "
          f"tiles={len(origins)} (always {manifest['tiles_always_run']}) device={device} blocks={len(blocks)}", flush=True)
    t_all = time.time()
    for name, f0, f1 in blocks:
        final = out / f"{name}.csv"
        if final.exists():
            print(f"[survey] {name}: done already, skipping", flush=True)
            continue
        if f1 <= f0:
            continue
        first = ((f0 + step - 1) // step) * step   # samples on a fixed global grid
        src = (frames_ffmpeg(a.video, first, f1, step, fps, w, h, keyframes_only) if use_ffmpeg
               else frames_opencv(a.video, first, f1, step))
        part = out / f"{name}.csv.part"
        t0, nf, ntiles, ndet, ndrop = time.time(), 0, 0, 0, 0
        expected = max(1, (f1 - first + step - 1) // step)
        with part.open("w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(FIELDS)
            for fidx, frame in src:
                b, s, nt = detect_frame(model, device, frame, origins, always, a.tile, a.dark_gate, a.batch, a.score_threshold, a.nms_iou)
                nf += 1
                ntiles += nt
                for (x1, y1, x2, y2), sc in zip(b, s):
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    if not mask[min(h - 1, int(cy)), min(w - 1, int(cx))]:
                        continue
                    if in_zone(cx, cy, zones):
                        ndrop += 1
                        continue
                    wr.writerow([fidx, round(fidx / fps, 2), round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
                                 round(cx, 1), round(cy, 1), round(float(sc), 3), area_of(cx, cy, areas)])
                    ndet += 1
                if a.progress_every and nf % a.progress_every == 0:
                    el = time.time() - t0
                    print(f"[survey] {name}: {nf}/{expected} frames, {el / nf:.2f} s/frame, {ntiles / nf:.1f} tiles/frame, "
                          f"{ndet} ants, eta {el / nf * (expected - nf) / 60:.1f} min", flush=True)
        if nf == 0:
            part.unlink()
            raise RuntimeError(f"No frames decoded for {name}; try --decoder opencv")
        part.rename(final)
        el = time.time() - t0
        print(f"[survey] {name}: {nf} frames in {el / 60:.1f} min ({el / nf:.2f} s/frame, {ntiles / nf:.1f} tiles/frame), "
              f"{ndet} detections, {ndrop} dropped in zones", flush=True)
    print(f"[survey] finished in {(time.time() - t_all) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
