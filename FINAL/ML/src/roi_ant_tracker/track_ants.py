"""Track ants frame by frame (every frame of the video, any frame rate) under "conservation" rules (Can, 2026-10-30).

All timings are in seconds and distances in px per second, so 10 fps and 15 fps footage behave the same.

Rules
  1. Ants are conserved. An ID never disappears in the open field. It ends only by
     - going into the nest (lost within the nest enter radius), or
     - being marked dead/debris (no movement for --dead-minutes, default 5).
     An ant hidden inside a feeder cup keeps its ID in that feeder's queue.
  2. New IDs start only (a) for ants already on the tray in the first frame, or (b) at the nest opening.
     A detection anywhere else that cannot be explained by a known ID is ignored (logged as a false detection).
  3. Feeders: nothing is detected inside the cups. An ID lost at a cup has gone in and joins that feeder's
     queue; the next ant to come out gets the ID at the front (first in, first out). An ant coming out of an
     empty cup was inside before tracking started and gets a new ID. Individual identity is not the aim:
     the counts (on the tray, inside each feeder, in/out events) are.
  4. Merging / touching: when two ants become one detection, both IDs stay on it ("merged"). When the
     detection splits again, the IDs are handed back by distance. Identity through a merge is not guaranteed,
     but no ID is lost or created.
  5. Lost ants (detector missed them in the open field) keep their ID at the last position and are re-found
     by the nearest unexplained detection, with a search radius that grows with the time lost.
  6. Dead/debris: an ID that has not moved more than --still-radius px for --dead-minutes is marked dead.
     Dead is final: the ID stays on its spot for good, never moves, never changes ID and is never counted
     as an ant again. A live ant walking over it keeps its own ID.

Detection: full-tray sweep every --sweep-seconds; in between, small crops around every visible or lost
ID plus fixed crops at the nest opening.

Outputs (in --output)
  tracks.csv            frame, t_sec, track_id, state, cx, cy, x1..y2, score, observed, area, partner
  tracks_summary.csv    one row per ID
  counts.csv            once per second: ants on the tray, inside each feeder, gone into the nest, dead
  feeder_events.csv     every ant going into / coming out of a cup (hidden >= --feeder-min-seconds), with the number inside
  feeder_visits.csv     in -> out pairs (first in, first out), with duration
  events.csv            merges, splits, losses, re-finds, deaths, revivals
  ignored_detections.csv detections that rule 2 refused to turn into new IDs
  overlay.mp4           (with --overlay) 1920x1080 video, paths kept for the whole clip
  track_manifest.json
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import nms

from .model import choose_device, load_detector
from .survey import (area_of, detect_frame, frames_ffmpeg, frames_opencv, in_zone, load_areas, load_zones, probe,
                     tile_origins, tile_touches_area)

FIELDS = ["frame", "t_sec", "track_id", "state", "cx", "cy", "x1", "y1", "x2", "y2", "det_score", "p_ant",
          "observed", "area", "partner"]
VISIBLE = ("active", "merged", "dead")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--zones", help="nest_clearance.json (detections inside are dropped).")
    p.add_argument("--areas", help="areas.json (labels in the output).")
    p.add_argument("--tracking-zones", default="FINAL/runs/zones/tracking_zones.json")
    p.add_argument("--start-seconds", type=float, default=0.0)
    p.add_argument("--duration-seconds", type=float, default=60.0)
    p.add_argument("--sweep-seconds", type=float, default=2.0, help="Time between full-tray detections.")
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--scale", type=float, default=2.4)
    p.add_argument("--score-threshold", type=float, default=0.3)
    p.add_argument("--birth-threshold", type=float, default=0.5)
    p.add_argument("--tile", type=int, default=320)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--dark-gate", type=float, default=30.0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--nms-iou", type=float, default=0.3)
    p.add_argument("--max-speed", type=float, default=250.0, help="px/s a visible ant may move (search radius per frame = this / fps).")
    p.add_argument("--merge-radius", type=float, default=25.0, help="An unmatched ID this close to another ID's detection is merged with it.")
    p.add_argument("--lost-after-seconds", type=float, default=0.3, help="Unseen this long -> the ID counts as hidden/lost.")
    p.add_argument("--lost-speed", type=float, default=40.0, help="px/s the search radius grows while an ID is lost.")
    p.add_argument("--lost-max-radius", type=float, default=600.0)
    p.add_argument("--still-radius", type=float, default=8.0)
    p.add_argument("--pin-seconds", type=float, default=3.0, help="Still this long -> pinned (15 px search).")
    p.add_argument("--dead-minutes", type=float, default=5.0)
    p.add_argument("--static-minutes", type=float, default=5.0,
                   help="A spot detected in nearly every full search for this long is a fixed mark (debris, dead ant): ignored from then on.")
    p.add_argument("--static-fraction", type=float, default=0.8, help="...'nearly every' = this fraction of full searches.")
    p.add_argument("--feeder-min-seconds", type=float, default=3.0,
                   help="An ant must stay out of sight in a cup this long to count as going in (shorter = missed at the rim).")
    p.add_argument("--nest-grace", type=float, default=10.0, help="s: an ID that just entered the nest and pops out again keeps its ID.")
    p.add_argument("--orphan-seconds", type=float, default=1.5, help="An unexplained detection must be followed this long before it becomes an ID.")
    p.add_argument("--orphan-move", type=float, default=20.0, help="px it must move (still specks never qualify).")
    p.add_argument("--calibration", default="FINAL/runs/zones/score_calibration.json",
                   help="Maps detector score to P(real ant) measured on the test set.")
    p.add_argument("--overlay", action="store_true")
    p.add_argument("--trail", type=int, default=0, help="Overlay trail length in frames (0 = whole path).")
    p.add_argument("--device", default="auto")
    p.add_argument("--decoder", choices=["auto", "ffmpeg", "opencv"], default="auto")
    p.add_argument("--progress-every", type=int, default=100)
    return p


# ---------------------------------------------------------------- detection helpers

def set_scale(model, size: int) -> None:
    model.transform.min_size = (int(size),)
    model.transform.max_size = max(int(size), 1333)


def detect_crops(model, device, frame, centers, crop, scale, batch, thr):
    h, w = frame.shape[:2]
    origins = set()
    q = max(crop // 4, 1)
    for cx, cy in centers:
        x = int(min(max(cx - crop / 2, 0), w - crop))
        y = int(min(max(cy - crop / 2, 0), h - crop))
        origins.add((x // q * q, y // q * q))
    origins = sorted(origins)
    if not origins:
        return np.zeros((0, 4)), np.zeros(0), 0
    set_scale(model, round(crop * scale))
    boxes, scores = [], []
    with torch.inference_mode():
        for i in range(0, len(origins), batch):
            chunk = origins[i:i + batch]
            tensors = [torch.from_numpy(np.ascontiguousarray(frame[y:y + crop, x:x + crop, ::-1])).permute(2, 0, 1)
                       .float().div(255.0).to(device) for x, y in chunk]
            tensors += [torch.zeros_like(tensors[0])] * (batch - len(tensors))   # same shape every call
            for (x, y), out in zip(chunk, model(tensors)):
                keep = (out["labels"] == 1) & (out["scores"] >= thr)
                b = out["boxes"][keep].detach().cpu()
                if len(b):
                    b[:, [0, 2]] += x
                    b[:, [1, 3]] += y
                    boxes.append(b)
                    scores.append(out["scores"][keep].detach().cpu())
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), len(origins)
    b, s = torch.cat(boxes), torch.cat(scores)
    k = nms(b, s, 0.3)
    return b[k].numpy(), s[k].numpy(), len(origins)


# ---------------------------------------------------------------- tracker

@dataclass
class Ant:
    tid: int
    x: float
    y: float
    box: tuple
    score: float
    state: str = "active"          # active | merged | lost | dead | feeder | nest
    born: str = ""
    born_frame: int = 0
    last_seen: int = 0
    missed: int = 0
    vx: float = 0.0
    vy: float = 0.0
    partner: int = 0               # merged: ID whose detection this one shares
    feeder: str = ""
    anchor: tuple = (0.0, 0.0)
    still_since: int = 0
    lost_since: int = 0
    nest_frame: int = 0
    merge_pos: tuple = (0.0, 0.0)  # position / velocity / stillness when it joined a merge group
    merge_v: tuple = (0.0, 0.0)
    merge_frame: int = 0
    merge_pinned: bool = False


def dist(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class ConservedTracker:
    def __init__(self, a, fps, zones_cfg):
        self.a, self.fps = a, fps
        self.ants: dict[int, Ant] = {}
        self.next_id = 1
        self.nest = zones_cfg["nest"]
        self.feeders = {f["id"]: f for f in zones_cfg["feeders"]}
        self.queue: dict[str, deque] = {f: deque() for f in self.feeders}
        self.visits: list[dict] = []
        self.open_visit: dict[int, dict] = {}
        self.feeder_events: list[dict] = []
        self.nest_events: list[dict] = []
        self.events: list[dict] = []
        self.ignored: list[dict] = []
        self.cands: list[dict] = []
        self._frame = 0    # unexplained detections followed until they prove to be ants
        self.pin_frames = int(a.pin_seconds * fps)
        self.step = 1.0 / fps                               # seconds per frame
        self.max_dist = a.max_speed / fps                   # px per frame
        self.lost_after = max(1, int(round(a.lost_after_seconds * fps)))
        self.orphan_hits = max(3, int(round(a.orphan_seconds * fps)))
        self.feeder_min_frames = max(1, int(round(a.feeder_min_seconds * fps)))
        self.dead_frames = int(a.dead_minutes * 60 * fps)

    # -- helpers
    def ev(self, f, kind, tid, x, y, note=""):
        self.events.append(dict(frame=f, t_sec=round(f / self.fps, 2), event=kind, track_id=tid, x=round(x), y=round(y), note=note))

    def near_nest(self, x, y, r):
        """Within r px of the nest: of the nest slide rectangle if given, else of the opening centre."""
        if "rect" in self.nest:
            rx, ry, rw, rh = self.nest["rect"]
            dx, dy = max(rx - x, 0, x - (rx + rw)), max(ry - y, 0, y - (ry + rh))
            return float(np.hypot(dx, dy)) <= r
        return dist((x, y), (self.nest["cx"], self.nest["cy"])) <= r

    def feeder_at(self, x, y, key):
        for fid, f in self.feeders.items():
            if dist((x, y), (f["cx"], f["cy"])) <= f[key]:
                return fid
        return ""

    def new_ant(self, f, d, why):
        t = Ant(self.next_id, d["x"], d["y"], d["box"], d["s"], born=why, born_frame=f, last_seen=f,
                anchor=(d["x"], d["y"]), still_since=f)
        self.ants[t.tid] = t
        self.next_id += 1
        if why == "nest":
            self.nest_events.append(dict(frame=f, t_sec=round(f / self.fps, 2), event="out_of_nest", track_id=t.tid))
        return t

    def place(self, t: Ant, f, d):
        gap = max(f - t.last_seen, 1)
        if t.state == "active" and t.missed == 0:
            t.vx, t.vy = (d["x"] - t.x) / gap, (d["y"] - t.y) / gap
        else:
            t.vx = t.vy = 0.0
        t.x, t.y, t.box, t.score = d["x"], d["y"], d["box"], d["s"]
        t.last_seen, t.missed = f, 0
        if dist((t.x, t.y), t.anchor) > self.a.still_radius:
            t.anchor, t.still_since = (t.x, t.y), f

    def close_visit(self, v):
        v["enter_sec"] = round(v["enter_frame"] / self.fps, 2)
        done = v.get("exit_frame", "") != ""
        v["exit_sec"] = round(v["exit_frame"] / self.fps, 2) if done else ""
        v["duration_sec"] = round((v["exit_frame"] - v["enter_frame"]) / self.fps, 2) if done else ""
        self.visits.append(v)

    def remember_merge(self, t: Ant, f):
        t.merge_pos, t.merge_v, t.merge_frame = (t.x, t.y), (t.vx, t.vy), f
        t.merge_pinned = t.state == "dead" or f - t.still_since >= self.pin_frames

    def merge_cost(self, t: Ant, f, d) -> float:
        """Where should this member of a merge group be now? A still ant stays on its spot; a moving ant
        continues on its pre-merge heading (up to 3 s)."""
        # squared distance in units of the expected error: a still ant should be within ~3 px of its spot,
        # a moving ant's predicted position gets less certain the longer the merge lasts
        if t.merge_pinned:
            return (dist(t.anchor, (d["x"], d["y"])) / 3.0) ** 2
        k = min(f - t.merge_frame, int(3 * self.fps))
        pred = (t.merge_pos[0] + t.merge_v[0] * k, t.merge_pos[1] + t.merge_v[1] * k)
        return (dist(pred, (d["x"], d["y"])) / (10.0 + 20.0 * (f - t.merge_frame) * self.step)) ** 2

    def pinned(self, t: Ant, f) -> bool:
        return t.state == "dead" or (f - t.still_since >= self.pin_frames and t.missed <= 2)

    # -- one frame
    def update(self, f, dets, first_frame):
        a = self.a
        self._frame = f
        D = [dict(x=(b[0] + b[2]) / 2, y=(b[1] + b[3]) / 2, box=tuple(map(float, b)), s=float(s)) for b, s in dets]
        used = set()
        took_det: dict[int, int] = {}

        # 1a. live IDs <-> detections, one to one
        vis = [t for t in self.ants.values() if t.state == "active"]
        corpses = [t.anchor for t in self.ants.values() if t.state == "dead"]
        on_corpse = [any(dist(c, (d["x"], d["y"])) <= 6 for c in corpses) for d in D]
        if vis and D:
            C = np.full((len(vis), len(D)), 1e6)
            for i, t in enumerate(vis):
                # an ant resting on a dead one is never pinned to that spot: it must be free to walk off
                if self.pinned(t, f) and not any(dist(c, t.anchor) <= 10 for c in corpses):
                    c, r = t.anchor, 15.0
                else:
                    c, r = (t.x + t.vx, t.y + t.vy), self.max_dist + 80.0 * self.step * t.missed
                for j, d in enumerate(D):
                    dd = dist(c, (d["x"], d["y"]))
                    if dd <= r:
                        C[i, j] = dd / r + (0.6 if on_corpse[j] else 0.0)   # prefer anything but the corpse itself
            for i, j in zip(*linear_sum_assignment(C)):
                if C[i, j] < 1e5:
                    self.place(vis[i], f, D[j])
                    used.add(j)
                    took_det[vis[i].tid] = j
        # 1b. dead objects: a leftover detection on their exact spot is the dead object itself. It never moves,
        #     never changes ID, and never takes a detection a live ant could use.
        for t in self.ants.values():
            if t.state != "dead":
                continue
            near = [j for j in range(len(D)) if j not in used and dist(t.anchor, (D[j]["x"], D[j]["y"])) <= 10]
            if near:
                j = min(near, key=lambda j: dist(t.anchor, (D[j]["x"], D[j]["y"])))
                used.add(j)
                t.last_seen, t.missed = f, 0
        vis = [t for t in self.ants.values() if t.state in ("active", "dead")]
        matched_ids = {t.tid for t in vis if t.last_seen == f}

        # 2. merged IDs follow their partner; unmatched visible IDs merge, hide or get lost
        for t in list(self.ants.values()):
            if t.state == "merged":
                p = self.ants.get(t.partner)
                if p is not None and p.last_seen == f and p.state in ("active", "dead"):
                    t.x, t.y, t.box, t.last_seen = p.x, p.y, p.box, f
                else:                                  # partner vanished too: treat like a missed ant
                    t.state, t.missed = "active", self.lost_after
        for t in vis:
            if t.tid in matched_ids:
                continue
            was_pinned = self.pinned(t, f)
            t.missed += 1
            if t.state == "dead":
                continue                               # dead objects stay where they are
            partner = None
            for p in vis:
                if p.tid in matched_ids and p.state == "active" and dist((t.x + t.vx, t.y + t.vy), (p.x, p.y)) <= a.merge_radius:
                    if partner is None or dist((t.x, t.y), (p.x, p.y)) < dist((t.x, t.y), (partner.x, partner.y)):
                        partner = p
            if partner is not None:
                if not any(o.state == "merged" and o.partner == partner.tid for o in self.ants.values()):
                    self.remember_merge(partner, f)    # partner's own pre-merge state
                self.remember_merge(t, f)
                t.state, t.partner = "merged", partner.tid
                t.x, t.y, t.box, t.last_seen = partner.x, partner.y, partner.box, f
                self.ev(f, "merged", t.tid, t.x, t.y, f"with {partner.tid}")
                continue
            if t.missed >= self.lost_after:
                fid = self.feeder_at(t.x, t.y, "hide_radius")
                if fid:
                    t.state, t.feeder = "feeder", fid
                    self.queue[fid].append(t.tid)
                    # provisional until it has stayed out of sight for --feeder-min-seconds
                    self.open_visit[t.tid] = dict(feeder=fid, track_id_in=t.tid, enter_frame=f, pending=True)
                    self.ev(f, "into_feeder", t.tid, t.x, t.y, fid)
                elif self.near_nest(t.x, t.y, self.nest["enter_radius"]):
                    t.state, t.nest_frame = "nest", f
                    self.nest_events.append(dict(frame=f, t_sec=round(f / self.fps, 2), event="into_nest", track_id=t.tid))
                    self.ev(f, "into_nest", t.tid, t.x, t.y)
                else:
                    t.state, t.lost_since = "lost", f
                    self.ev(f, "lost", t.tid, t.x, t.y)

        # 3. explain the remaining detections
        free = [j for j in range(len(D)) if j not in used]
        # 3a. duplicates of an ID that was just placed
        free = [j for j in free if not any(t.last_seen == f and t.state in ("active", "dead") and
                                           dist((t.x, t.y), (D[j]["x"], D[j]["y"])) < 10 for t in self.ants.values())]
        # 3b. split: a merge group (partner + merged IDs) meets 2+ detections -> hand IDs back jointly,
        #     still ants to their spot, moving ants along their pre-merge heading
        for pid in sorted({t.partner for t in self.ants.values() if t.state == "merged"}):
            p = self.ants.get(pid)
            if p is None or pid not in took_det:
                continue
            members = [p] + [t for t in self.ants.values() if t.state == "merged" and t.partner == pid]
            near = [j for j in free if dist((p.x, p.y), (D[j]["x"], D[j]["y"])) <= 40]
            if not near:
                continue
            cands = [took_det[pid]] + near
            C = np.array([[self.merge_cost(m, f, D[j]) for j in cands] for m in members])
            rows, cols = linear_sum_assignment(C)
            for i, k in zip(rows, cols):
                m, j = members[i], cands[k]
                was_merged = m.state == "merged"
                m.state, m.partner = "active", 0
                m.last_seen = f - 1 if was_merged else m.last_seen
                m.missed = 0
                self.place(m, f, D[j])
                took_det[m.tid] = j
                if was_merged:
                    self.ev(f, "split", m.tid, m.x, m.y)
                if j in free:
                    free.remove(j)
            # members left without a detection stay merged with whoever took the partner's detection
            holder = next((m for m in members if took_det.get(m.tid) == cands[0] and m.state == "active"), None)
            for i, m in enumerate(members):
                if i not in rows and holder is not None and m is not holder:
                    m.state, m.partner = "merged", holder.tid
        # 3c. out of a feeder: first in, first out. A new ant at a cup with nobody queued was already inside
        #     when tracking started: it gets an ID straight away so the count stays right.
        rest = []
        for j in free:
            d = D[j]
            fid = self.feeder_at(d["x"], d["y"], "exit_radius")
            if not fid or first_frame:                 # first frame: ants already near a cup are just ants on the tray
                rest.append(j)
                continue
            if self.queue[fid]:
                tid = self.queue[fid].popleft()
                t = self.ants[tid]
                t.state, t.feeder = "active", ""
                t.anchor, t.still_since = (d["x"], d["y"]), f
                self.place(t, f, d)
                v = self.open_visit.pop(tid, None)
                if v and v.get("pending"):
                    # back within --feeder-min-seconds: the ant was only missed by the detector at the rim
                    self.ev(f, "brief_hide_ignored", tid, d["x"], d["y"], f"{fid} {(f - v['enter_frame']) / self.fps:.1f}s")
                    continue
                if v:
                    v.update(track_id_out=tid, exit_frame=f)
                    self.close_visit(v)
                self.ev(f, "out_of_feeder", tid, d["x"], d["y"], fid)
            elif d["s"] >= a.birth_threshold and not any(t.state == "lost" and dist((t.x, t.y), (d["x"], d["y"])) < 200
                                                         for t in self.ants.values()):
                t = self.new_ant(f, d, f"from_{fid}")
                self.ev(f, "born", t.tid, d["x"], d["y"], t.born + " (inside before tracking started)")
            else:
                rest.append(j)
                continue
            self.feeder_events.append(dict(frame=f, t_sec=round(f / self.fps, 2), feeder=fid, event="out",
                                           track_id=t.tid, ants_inside=len(self.queue[fid])))
        free = rest
        # 3d. re-find lost IDs (radius grows with time lost)
        lost = [t for t in self.ants.values() if t.state == "lost"]
        if lost and free:
            C = np.full((len(lost), len(free)), 1e6)
            for i, t in enumerate(lost):
                r = min(self.max_dist + a.lost_speed * (f - t.lost_since) / self.fps, a.lost_max_radius)
                t_near_nest = self.near_nest(t.x, t.y, self.nest["birth_radius"] + 50)
                for k, j in enumerate(free):
                    if not t_near_nest and self.near_nest(D[j]["x"], D[j]["y"], self.nest["birth_radius"]):
                        continue                       # a detection at the nest opening is a new ant, not a far lost one
                    dd = dist((t.x, t.y), (D[j]["x"], D[j]["y"]))
                    if dd <= r:
                        C[i, k] = dd
            took = set()
            for i, k in zip(*linear_sum_assignment(C)):
                if C[i, k] < 1e5:
                    t = lost[i]
                    t.state = "active"
                    self.place(t, f, D[free[k]])
                    self.ev(f, "refound", t.tid, t.x, t.y, f"lost {(f - t.lost_since) / self.fps:.1f}s")
                    took.add(free[k])
            free = [j for j in free if j not in took]
        # 3e. new IDs: first frame, or at the nest opening. An ID that went into the nest in the last
        #     --nest-grace seconds and pops out again is the same ant flickering at the entrance.
        for j in list(free):
            d = D[j]
            if first_frame and d["s"] >= a.birth_threshold:
                t = self.new_ant(f, d, "start")
                self.ev(f, "born", t.tid, d["x"], d["y"], "start")
                free.remove(j)
            elif self.near_nest(d["x"], d["y"], self.nest["birth_radius"]) and d["s"] >= a.birth_threshold:
                back = [t for t in self.ants.values() if t.state == "nest" and f - t.nest_frame <= a.nest_grace * self.fps]
                if back:
                    t = min(back, key=lambda t: dist((t.x, t.y), (d["x"], d["y"])))
                    t.state = "active"
                    t.anchor, t.still_since = (d["x"], d["y"]), f
                    self.place(t, f, d)
                    self.nest_events = [e for e in self.nest_events if not (e["event"] == "into_nest" and e["track_id"] == t.tid
                                                                         and e["frame"] == t.nest_frame)]
                    self.ev(f, "back_from_nest_entrance", t.tid, d["x"], d["y"])
                else:
                    t = self.new_ant(f, d, "nest")
                    self.ev(f, "born", t.tid, d["x"], d["y"], "nest")
                free.remove(j)
        # 3f. anything else is not trusted yet: follow it as a candidate; it becomes an ID only if it behaves
        #     like an ant (followed for --orphan-seconds, moved >= --orphan-move px, mean score >= 0.7).
        #     Typical case: an ant that was already hidden in a cup when tracking started.
        for c in self.cands:
            c["seen"] = False
        for j in free:
            d = D[j]
            c = min((c for c in self.cands if not c["seen"] and dist((c["x"], c["y"]), (d["x"], d["y"])) <= 30 + 30 * (f - c["last"]) * self.step),
                    key=lambda c: dist((c["x"], c["y"]), (d["x"], d["y"])), default=None)
            if c is None:
                c = dict(x=d["x"], y=d["y"], first=f, last=f, hits=0, ssum=0.0, start=(d["x"], d["y"]), moved=0.0, seen=False)
                self.cands.append(c)
            c.update(x=d["x"], y=d["y"], last=f, seen=True)
            c["hits"] += 1
            c["ssum"] += d["s"]
            c["moved"] = max(c["moved"], dist(c["start"], (d["x"], d["y"])))
            if c["hits"] >= self.orphan_hits and c["moved"] >= a.orphan_move and c["ssum"] / c["hits"] >= 0.7:
                lost_near = [t for t in self.ants.values() if t.state == "lost" and dist((t.x, t.y), (d["x"], d["y"])) <= a.lost_max_radius]
                if lost_near:                          # it is most likely an ant we lost, not a new one
                    t = min(lost_near, key=lambda t: dist((t.x, t.y), (d["x"], d["y"])))
                    t.state = "active"
                    self.place(t, f, d)
                    self.ev(f, "refound", t.tid, d["x"], d["y"], "via candidate")
                    c["hits"] = -10 ** 9
                    continue
                fid = self.feeder_at(c["start"][0], c["start"][1], "exit_radius")
                t = self.new_ant(f, d, f"from_{fid}" if fid else "unexplained")
                self.ev(f, "born", t.tid, d["x"], d["y"], t.born + f" (followed {(f - c['first']) / self.fps:.1f}s)")
                c["hits"] = -10 ** 9                   # retire candidate
            else:
                self.ignored.append(dict(frame=f, t_sec=round(f / self.fps, 2), cx=round(d["x"], 1), cy=round(d["y"], 1),
                                         score=round(d["s"], 3), area=self.feeder_at(d["x"], d["y"], "exit_radius") or "open"))
        self.cands = [c for c in self.cands if c["hits"] > 0 and f - c["last"] <= 3 * self.fps]

        # 4. confirm feeder entries that have lasted long enough
        for tid, v in self.open_visit.items():
            if v.get("pending") and f - v["enter_frame"] >= self.feeder_min_frames:
                v["pending"] = False
                self.feeder_events.append(dict(frame=v["enter_frame"], t_sec=round(v["enter_frame"] / self.fps, 2),
                                               feeder=v["feeder"], event="in", track_id=tid,
                                               ants_inside=sum(not w.get("pending") for w in self.open_visit.values()
                                                               if w["feeder"] == v["feeder"])))

        # 5. dead / debris after a long stillness
        for t in self.ants.values():
            if t.state == "active" and f - t.still_since >= self.dead_frames:
                t.state = "dead"
                self.ev(f, "marked_dead", t.tid, t.x, t.y, f"still {(f - t.still_since) / self.fps / 60:.1f} min")

    def mark_static(self, f, x, y):
        """A fixed mark was found at (x, y): the ID sitting on it (if any) is dead; a pending feeder entry by it is dropped."""
        best = None
        for t in self.ants.values():
            if t.state in ("active", "lost", "feeder", "merged") and dist((t.x, t.y), (x, y)) <= 12:
                if best is None or dist((t.x, t.y), (x, y)) < dist((best.x, best.y), (x, y)):
                    best = t
        if best is None:
            return
        if best.state == "feeder":
            q = self.queue.get(best.feeder)
            if q and best.tid in q:
                q.remove(best.tid)
            v = self.open_visit.pop(best.tid, None)
            if v and not v.get("pending"):
                # it had been counted as going in: take that back
                self.feeder_events = [e for e in self.feeder_events if not (e["event"] == "in" and e["track_id"] == best.tid
                                                                           and e["frame"] == v["enter_frame"])]
        best.state, best.anchor, best.x, best.y = "dead", (x, y), x, y
        self.ev(f, "marked_dead", best.tid, x, y, "fixed mark (seen in most full searches)")

    def lookahead_centres(self):
        """Where to look between sweeps: every visible, merged or lost ID."""
        f = self._frame
        return ([(t.x + t.vx, t.y + t.vy) for t in self.ants.values()
                 if t.state in ("active", "merged") or (t.state == "lost" and f - t.lost_since <= 5 * self.fps)]
                + [(c["x"], c["y"]) for c in self.cands])


class StaticSpots:
    """Fixed marks that look like ants (debris on the floor or on a cup rim, dead ants).

    Counted on full searches only: a spot (8 px cell, with its neighbours) found in at least `fraction` of the
    full searches of the last `minutes` becomes static. Static spots are ignored for the rest of the run, and an
    ID sitting on one is marked dead. This catches marks the 5-minute dead rule misses because the detector
    keeps losing them (e.g. a mark on a cup rim that would otherwise 'go into the feeder' again and again)."""

    def __init__(self, sweeps: int, fraction: float, cell: int = 8):
        self.sweeps, self.fraction, self.cell = max(5, sweeps), fraction, cell
        self.hits: dict[tuple, deque] = defaultdict(deque)
        self.n_sweeps = 0
        self.static: set[tuple] = set()

    def key(self, x, y):
        return int(x // self.cell), int(y // self.cell)

    def add_sweep(self, points) -> list[tuple]:
        self.n_sweeps += 1
        seen = set()
        for x, y in points:
            kx, ky = self.key(x, y)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    seen.add((kx + dx, ky + dy))
        for k in seen:
            self.hits[k].append(self.n_sweeps)
        new = []
        for k in list(self.hits):
            q = self.hits[k]
            while q and q[0] <= self.n_sweeps - self.sweeps:
                q.popleft()
            if not q:
                del self.hits[k]
                continue
            if k not in self.static and self.n_sweeps >= self.sweeps and len(q) >= self.fraction * self.sweeps:
                self.static.add(k)
                new.append(((k[0] + 0.5) * self.cell, (k[1] + 0.5) * self.cell))
        return new

    def is_static(self, x, y) -> bool:
        return self.key(x, y) in self.static


# ---------------------------------------------------------------- output

def id_color(i: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(i * 7919)
    c = rng.integers(60, 255, 3)
    return int(c[0]), int(c[1]), int(c[2])


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


def summarise(tracks_csv: Path, tracker: ConservedTracker, out: Path, fps: float) -> None:
    rows = defaultdict(list)
    with tracks_csv.open() as fh:
        for r in csv.DictReader(fh):
            rows[int(r["track_id"])].append(r)
    visits = defaultdict(int)
    for v in tracker.visits:
        visits[v["track_id_in"]] += 1
    out_rows = []
    for tid, t in sorted(tracker.ants.items()):
        rs = [r for r in rows.get(tid, []) if r["observed"] == "1" and r["state"] == "active"]
        xy = np.array([[float(r["cx"]), float(r["cy"])] for r in rs]) if rs else np.zeros((0, 2))
        path = float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) > 1 else 0.0
        allr = rows.get(tid, [])
        out_rows.append(dict(track_id=tid, born=t.born, born_sec=round(t.born_frame / fps, 1),
                             final_state=t.state, frames_listed=len(allr), frames_seen=len(rs),
                             path_px=round(path, 1),
                             mean_speed_px_s=round(path / max(len(rs) / fps, 1e-6), 2) if rs else 0.0,
                             feeder_visits=visits[tid],
                             mean_p_ant=round(float(np.mean([float(r["p_ant"]) for r in rs])), 3) if rs else ""))
    write_csv(out, out_rows, list(out_rows[0].keys()) if out_rows else ["track_id"])


def main() -> None:
    a = build_parser().parse_args()
    fps, n_frames, w, h = probe(a.video)
    mask = cv2.imread(a.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        sys.exit(f"Could not read mask {a.mask}")
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    zones, areas = load_zones(a.zones), load_areas(a.areas)
    zcfg = json.loads(Path(a.tracking_zones).read_text())
    origins = tile_origins(mask.astype(np.uint8) * 255, a.tile, a.overlap)
    always = {o: tile_touches_area(o[0], o[1], a.tile, areas) for o in origins}
    device = choose_device(a.device)
    model, _ = load_detector(a.checkpoint, device)
    model.roi_heads.score_thresh = a.score_threshold
    tile_size = round(a.tile * a.scale)
    nest_crops = [(zcfg["nest"]["cx"] + dx, zcfg["nest"]["cy"] + 60) for dx in (-110, 0, 110)]

    f0 = int(round(a.start_seconds * fps))
    f1 = min(n_frames, f0 + int(round(a.duration_seconds * fps)))
    use_ffmpeg = a.decoder == "ffmpeg" or (a.decoder == "auto" and shutil.which("ffmpeg"))
    src = frames_ffmpeg(a.video, f0, f1, 1, fps, w, h, False) if use_ffmpeg else frames_opencv(a.video, f0, f1, 1)

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    tr = ConservedTracker(a, fps, zcfg)
    sweep_every = max(1, int(round(a.sweep_seconds * fps)))
    static = StaticSpots(int(round(a.static_minutes * 60 * fps / sweep_every)), a.static_fraction)
    cal = json.loads(Path(a.calibration).read_text())["bins"] if Path(a.calibration).exists() else []

    def p_ant(score: float) -> float:
        for b_ in cal:
            if b_["lo"] <= score < b_["hi"]:
                return b_["p_ant"]
        return score
    (out / "track_manifest.json").write_text(json.dumps(dict(vars(a), fps=fps, size=[w, h], device=str(device),
                                                             zones_config=zcfg), indent=2))
    writer = cv2.VideoWriter(str(out / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080)) if a.overlay else None
    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=a.trail or None))
    _tcache: dict[int, tuple[int, np.ndarray]] = {}

    def trail_arr(tid, pts_full):
        n_have, arr = _tcache.get(tid, (0, np.zeros((0, 2), np.int32)))
        if a.trail or len(pts_full) < n_have:
            arr = (np.array(pts_full, np.float32) / 2).astype(np.int32)
        elif len(pts_full) > n_have:
            new = list(pts_full)[n_have:]
            arr = np.vstack([arr, (np.array(new, np.float32) / 2).astype(np.int32)])
        _tcache[tid] = (len(pts_full), arr)
        return arr
    t_start, n, t_sweep, t_local, n_sweep = time.time(), 0, 0.0, 0.0, 0
    count_rows: list[dict] = []
    print(f"[track] frames {f0}-{f1} ({(f1 - f0) / fps:.0f} s at {fps:g} fps, {w}x{h}), full sweep every {sweep_every} frames, "
          f"device={device}", flush=True)
    with (out / "tracks.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(FIELDS)
        for fidx, frame in src:
            tt = time.time()
            sweep = (n % sweep_every == 0) or not tr.ants
            if sweep:
                set_scale(model, tile_size)
                b, s, _ = detect_frame(model, device, frame, origins, always, a.tile, a.dark_gate, a.batch,
                                       a.score_threshold, a.nms_iou)
                t_sweep += time.time() - tt
                n_sweep += 1
            else:
                b, s, _ = detect_crops(model, device, frame, tr.lookahead_centres() + nest_crops, a.crop, a.scale,
                                       a.batch, a.score_threshold)
                t_local += time.time() - tt
            dets = []
            for box, sc in zip(b, s):
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                if (mask[min(int(cy), h - 1), min(int(cx), w - 1)] and not in_zone(cx, cy, zones)
                        and not tr.feeder_at(cx, cy, "cup_radius")):     # nothing is detected inside the cups
                    dets.append((box, sc))
            if sweep:
                for sx, sy in static.add_sweep([((b_[0] + b_[2]) / 2, (b_[1] + b_[3]) / 2) for b_, _ in dets]):
                    tr.mark_static(fidx, sx, sy)
            dets = [(b_, s_) for b_, s_ in dets if not static.is_static((b_[0] + b_[2]) / 2, (b_[1] + b_[3]) / 2)]
            tr.update(fidx, dets, first_frame=(n == 0))
            for t in tr.ants.values():
                if t.state in ("nest", "feeder"):
                    continue
                obs = int(t.last_seen == fidx and t.state in ("active", "dead"))
                wr.writerow([fidx, round(fidx / fps, 2), t.tid, t.state, round(t.x, 1), round(t.y, 1),
                             *[round(v, 1) for v in t.box], round(t.score, 3) if obs else "",
                             round(p_ant(t.score), 3) if obs else "", obs,
                             area_of(t.x, t.y, areas), t.partner or ""])
                if obs and t.state == "active":
                    trails[t.tid].append((t.x, t.y))
            if writer is not None:
                vis = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
                for tid, pts_full in trails.items():
                    pts = trail_arr(tid, pts_full)
                    if len(pts) > 1:
                        cv2.polylines(vis, [pts], False, id_color(tid), 2 if tr.ants[tid].state in ("active", "merged") else 1)
                for t in tr.ants.values():
                    col = id_color(t.tid)
                    x1, y1, x2, y2 = (v / 2 for v in t.box)
                    if t.state == "active":
                        cv2.rectangle(vis, (int(x1) - 4, int(y1) - 4), (int(x2) + 4, int(y2) + 4), col, 2)
                        cv2.putText(vis, str(t.tid), (int(x2) + 5, int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                    elif t.state == "merged":
                        cv2.putText(vis, f"+{t.tid}", (int(x2) + 5, int(y1) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                    elif t.state == "lost":
                        c = (int(t.x / 2), int(t.y / 2))
                        cv2.circle(vis, c, 10, col, 1)
                        cv2.putText(vis, f"{t.tid}?", (c[0] + 12, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                    elif t.state == "dead":
                        c = (int(t.x / 2), int(t.y / 2))
                        cv2.drawMarker(vis, c, (0, 0, 0), cv2.MARKER_TILTED_CROSS, 16, 2)
                        cv2.putText(vis, f"{t.tid} dead", (c[0] + 10, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                for fid, fz in tr.feeders.items():
                    q = list(tr.queue[fid])
                    cv2.circle(vis, (fz["cx"] // 2, fz["cy"] // 2), fz["hide_radius"] // 2, (0, 160, 255), 1)
                    cv2.putText(vis, f"inside: {len(q)}  in {sum(e['feeder'] == fid and e['event'] == 'in' for e in tr.feeder_events)}"
                                     f"  out {sum(e['feeder'] == fid and e['event'] == 'out' for e in tr.feeder_events)}",
                                (fz["cx"] // 2 - 90, fz["cy"] // 2 + fz["hide_radius"] // 2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 255), 2)
                counts = defaultdict(int)
                for t in tr.ants.values():
                    counts[t.state] += 1
                cv2.putText(vis, f"t={fidx / fps:.1f}s  ants on tray {counts['active'] + counts['merged'] + counts['lost']}"
                                 f" (lost {counts['lost']})  in feeders {counts['feeder']}  dead {counts['dead']}"
                                 f"{'  [sweep]' if sweep else ''}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                writer.write(vis)
            if n % int(round(fps)) == 0:
                st_ = defaultdict(int)
                for t in tr.ants.values():
                    st_[t.state] += 1
                row = dict(frame=fidx, t_sec=round(fidx / fps, 1),
                           ants_on_tray=st_["active"] + st_["merged"] + st_["lost"], of_which_lost=st_["lost"],
                           **{f"in_{fid}": len(q) for fid, q in tr.queue.items()},
                           dead=st_["dead"])
                row["total_ants_outside_nest"] = row["ants_on_tray"] + sum(len(q) for q in tr.queue.values())
                count_rows.append(row)
            n += 1
            if device.type == "mps" and n % 500 == 0:
                torch.mps.empty_cache()
            if n % a.progress_every == 0:
                el = time.time() - t_start
                st = defaultdict(int)
                for t in tr.ants.values():
                    st[t.state] += 1
                print(f"[track] {n}/{f1 - f0} frames  {n / el:.1f} fps ({n / el / fps:.2f}x real time)  IDs {len(tr.ants)} "
                      f"{ {k: v for k, v in st.items() if k != 'nest'} }  feeder in/out {sum(e['event'] == 'in' for e in tr.feeder_events)}/"
                      f"{sum(e['event'] == 'out' for e in tr.feeder_events)}  ignored dets {len(tr.ignored)}  "
                      f"sweep {1000 * t_sweep / max(n_sweep, 1):.0f} ms  local {1000 * t_local / max(n - n_sweep, 1):.0f} ms", flush=True)
    if writer is not None:
        writer.release()
    for v in list(tr.open_visit.values()):                  # still inside when the clip ends
        if not v.get("pending"):
            tr.close_visit(dict(v, track_id_out="", exit_frame=""))
    tr.feeder_events.sort(key=lambda e: e["frame"])
    write_csv(out / "feeder_visits.csv", tr.visits, ["feeder", "track_id_in", "track_id_out", "enter_frame", "enter_sec",
                                                      "exit_frame", "exit_sec", "duration_sec"])
    write_csv(out / "feeder_events.csv", tr.feeder_events, ["frame", "t_sec", "feeder", "event", "track_id", "ants_inside"])
    # objects marked dead later (debris, dead ants) were counted as ants before their 5 minutes were up:
    # add a corrected count that leaves them out from the start
    dead_ids = {str(t.tid) for t in tr.ants.values() if t.state == "dead"}
    want = {r["frame"] for r in count_rows}
    per_frame: dict[int, int] = defaultdict(int)
    with (out / "tracks.csv").open() as fh:
        for r in csv.DictReader(fh):
            fr_ = int(r["frame"])
            if fr_ in want and r["state"] in ("active", "merged", "lost") and r["track_id"] not in dead_ids:
                per_frame[fr_] += 1
    for r in count_rows:
        r["ants_on_tray_excl_dead_objects"] = per_frame.get(r["frame"], 0)
    write_csv(out / "counts.csv", count_rows, list(count_rows[0].keys()) if count_rows else ["frame"])
    write_csv(out / "events.csv", tr.events, ["frame", "t_sec", "event", "track_id", "x", "y", "note"])
    write_csv(out / "ignored_detections.csv", tr.ignored, ["frame", "t_sec", "cx", "cy", "score", "area"])
    summarise(out / "tracks.csv", tr, out / "tracks_summary.csv", fps)
    el = time.time() - t_start
    st = defaultdict(int)
    for t in tr.ants.values():
        st[t.state] += 1
    print(f"[track] done: {n} frames in {el / 60:.1f} min ({n / max(el, 1e-6) / fps:.2f}x real time); {len(tr.ants)} IDs "
          f"{ {k: v for k, v in st.items() if k != 'nest'} }; feeder in {sum(e['event'] == 'in' for e in tr.feeder_events)}, out "
          f"{sum(e['event'] == 'out' for e in tr.feeder_events)}, completed visits {sum(v['exit_sec'] != '' for v in tr.visits)}; "
          f"ignored detections {len(tr.ignored)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
