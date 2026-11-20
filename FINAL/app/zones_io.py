"""Read and write the zone files used by survey-video and track-ants (no GUI code here).

A zones folder holds:
  tray_mask.png          white = tray floor where ants are detected (nest slide cut out)
  nest_clearance.json    circle around the nest opening; detections inside are ignored
  tracking_zones.json    nest opening and feeder cups for the tracking rules
  areas.json             rectangles that label detections (feeder_left, feeder_right, nest_entrance)
  app_zones.json         what was clicked in the app (to edit later)
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

DEFAULTS = dict(nest_clearance=50, nest_enter=70, nest_birth=90, cup_radius=100, hide_margin=25, exit_margin=60)


def save_zones(folder: str | Path, z: dict, video_name: str = "") -> list[str]:
    """z: frame_w, frame_h, tray (list of [x,y]) or None, nest_slide [x,y,w,h] or None,
    nest {cx,cy}, feeders {feeder_left: {cx,cy,r}, feeder_right: {...}}, plus optional radii (see DEFAULTS)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    o = dict(DEFAULTS, **{k: v for k, v in z.items() if k in DEFAULTS})
    W, H = int(z["frame_w"]), int(z["frame_h"])
    written = []
    if z.get("tray"):
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [np.array(z["tray"], np.int32)], 255)
        if z.get("nest_slide"):
            x, y, w, h = map(int, z["nest_slide"])
            mask[y:y + h, x:x + w] = 0
        cv2.imwrite(str(folder / "tray_mask.png"), mask)
        written.append("tray_mask.png")
    nest = z.get("nest")
    if nest:
        (folder / "nest_clearance.json").write_text(json.dumps(dict(
            zone_version=1, video=video_name, frame_width=W, frame_height=H,
            zones=[dict(id="nest_opening_clearance", type="circle", cx=int(nest["cx"]), cy=int(nest["cy"]),
                        r=int(o["nest_clearance"]), rule="Detections whose centre is inside are ignored (ants still in the clump at the opening).")]),
            indent=2))
        written.append("nest_clearance.json")
    feeders = []
    for fid in ("feeder_left", "feeder_right"):
        fz = (z.get("feeders") or {}).get(fid)
        if fz:
            r = int(fz.get("r", o["cup_radius"]))
            feeders.append(dict(id=fid, cx=int(fz["cx"]), cy=int(fz["cy"]), cup_radius=r,
                                hide_radius=r + int(o["hide_margin"]), exit_radius=r + int(o["exit_margin"])))
    if nest or feeders:
        tz = dict(note=f"Zones for track-ants, saved by the ant app for {video_name}.", feeders=feeders)
        if nest:
            tz["nest"] = dict(cx=int(nest["cx"]), cy=int(nest["cy"]), enter_radius=int(o["nest_enter"]),
                              birth_radius=int(o["nest_birth"]),
                              note="Nest = the opening. Ants lost within enter_radius go in; new ants appear only within birth_radius.")
        (folder / "tracking_zones.json").write_text(json.dumps(tz, indent=2))
        written.append("tracking_zones.json")
        areas = [dict(id=f["id"], type="rect", x=f["cx"] - 190, y=f["cy"] - 175, w=380, h=350) for f in feeders]
        if nest:
            areas.append(dict(id="nest_entrance", type="rect", x=int(nest["cx"]) - 195, y=int(nest["cy"]) - 40, w=390, h=130))
        (folder / "areas.json").write_text(json.dumps(dict(areas_version=1, note="Area labels (full-frame px); saved by the ant app.",
                                                            areas=areas), indent=2))
        written.append("areas.json")
    (folder / "app_zones.json").write_text(json.dumps(dict(z, video=video_name), indent=2))
    written.append("app_zones.json")
    return written


def load_zones(folder: str | Path) -> dict:
    """Best effort: app_zones.json if present, else rebuilt from the separate zone files."""
    folder = Path(folder)
    if (folder / "app_zones.json").exists():
        return json.loads((folder / "app_zones.json").read_text())
    z: dict = {"feeders": {}}
    if (folder / "tracking_zones.json").exists():
        t = json.loads((folder / "tracking_zones.json").read_text())
        if "nest" in t:
            z["nest"] = dict(cx=t["nest"]["cx"], cy=t["nest"]["cy"])
            z["nest_enter"], z["nest_birth"] = t["nest"].get("enter_radius", 70), t["nest"].get("birth_radius", 90)
        for f in t.get("feeders", []):
            z["feeders"][f["id"]] = dict(cx=f["cx"], cy=f["cy"], r=f["cup_radius"])
    elif (folder / "nest_clearance.json").exists():
        c = json.loads((folder / "nest_clearance.json").read_text())["zones"][0]
        z["nest"] = dict(cx=c["cx"], cy=c["cy"])
    if (folder / "tray_mask.png").exists():
        m = cv2.imread(str(folder / "tray_mask.png"), cv2.IMREAD_GRAYSCALE)
        z["frame_h"], z["frame_w"] = m.shape
        cs, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            outer = [i for i in range(len(cs)) if hier[0][i][3] == -1]
            k = max(outer, key=lambda i: cv2.contourArea(cs[i]))
            c = cs[k]
            poly = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True)[:, 0, :]
            z["tray"] = poly.tolist()
            holes = [cs[i] for i in range(len(cs)) if hier[0][i][3] == k]
            if holes:
                x, y, w, h = cv2.boundingRect(max(holes, key=cv2.contourArea))
                if w * h > 5000:
                    z["nest_slide"] = [x, y, w, h]
    return z


def zones_complete(folder: str | Path) -> list[str]:
    """Names of zone files missing for a tracking run."""
    folder = Path(folder)
    return [n for n in ("tray_mask.png", "nest_clearance.json", "tracking_zones.json", "areas.json") if not (folder / n).exists()]
