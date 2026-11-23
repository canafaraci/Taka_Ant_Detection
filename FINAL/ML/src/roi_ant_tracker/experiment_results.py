"""Feeder-choice experiment: match the Raspberry Pi feeder log to the video and summarise each trial.

Design (Taka lab): repeating trials. A tap cue, about a minute later ONE feeder opens (which one is drawn
with a set chance, e.g. left 80 %), it stays open ~10 min, then closes. Example: tap 29 min, open 30, close 40,
tap 59, open 60, close 70, ...

Usage (from the repository root):
  python FINAL/pipeline.py experiment-results -- --log pi_log.txt --video M:\\2026-11-07_02-20-04.mp4 \\
      --runs FINAL/runs/exp_trial01 FINAL/runs/exp_trial02 [--survey FINAL/runs/survey_x] \\
      [--chance-left "80" | "1-10:80, 11-20:20"] [--feeder-map "1=left,2=right"] [--clock-offset 0]
  python FINAL/pipeline.py experiment-results -- --log pi_log.txt --video ... --check     (only read the log)

The Pi log can be any text or CSV file with one event per line. A line needs a date and time
(2026-11-07 02:49:04, 2026-11-07T02:49:04.5, 2026/11/07 02:49:04, 2026-11-07_02-49-04, or a Unix timestamp)
and a word for the event (tap / open / close, also opened, closing, shut). Open and close lines name the
feeder: left/right, "feeder 1", "servo 2", A/B, L/R, ... --feeder-map says which name is which cup.
Check the parsed result in log_events.csv and trials.csv.

Clock: video time = Pi time - clock offset - video start. The video start comes from the file name
(YYYY-MM-DD_HH-MM-SS) unless --video-start is given. --clock-offset = how many seconds the Pi clock is ahead
of the camera clock.

Data: tracking runs give ants inside each cup, ants at each cup (feeder area, outside the cup), entries and
exits; a survey run gives ants at each cup every few seconds (no inside counts, no entries). Where both
cover a time, tracking is used.

Windows per trial (relative to the actual log times):
  before = tap - --before-min .. tap      cue = tap .. open      open = open .. close      after = close .. close + --after-min

Outputs (in --output, default FINAL/runs/experiment_<video>):
  log_events.csv         every event read from the log, with video time
  trials.csv             one row per trial: times, opened feeder, chance, coverage, key results
  trial_windows.csv      trial x window x feeder: ants at / inside the cup, entries, exits
  conditions.csv         per chance setting: trials, how often each feeder opened, mean shares, latency
  peri_opening.csv/.png  ants at the opened vs the closed feeder around the opening (mean over trials)
  trial_shares.png       per trial: share of ants at the left feeder before the opening, opened feeder, chance
  matching.png           share at left vs chance left (when there are at least 2 chance settings)
  experiment_summary.json, report.txt
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ Pi log

DT_RE = re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})[ T_,;]+(\d{1,2})[:\-.](\d{2})[:\-.](\d{2})(?:[.,](\d+))?")
EPOCH_RE = re.compile(r"(?<![\d.])(1[5-9]\d{8}(?:\.\d+)?)(?![\d])")
KIND_RES = [("close", re.compile(r"\bclos|\bshut", re.I)), ("open", re.compile(r"\bopen", re.I)),
            ("tap", re.compile(r"\btap", re.I))]
FEEDER_RES = [re.compile(r"\b(left|right)\b", re.I),
              re.compile(r"feeder[\s_:=#-]*([A-Za-z0-9]+)", re.I),
              re.compile(r"\b(?:servo|motor|pin|gpio|door|cup|valve)[\s_:=#-]*([A-Za-z0-9]+)", re.I),
              re.compile(r"\b([ABLR12])\b")]
P_RES = [re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%"), re.compile(r"\bp(?:rob(?:ability)?)?\s*[=:]\s*(0?\.\d+|1(?:\.0+)?)\b", re.I)]
DEFAULT_MAP = {"left": "feeder_left", "l": "feeder_left", "a": "feeder_left", "1": "feeder_left",
               "right": "feeder_right", "r": "feeder_right", "b": "feeder_right", "2": "feeder_right"}


def parse_time(line: str) -> tuple[datetime | None, str]:
    m = DT_RE.search(line)
    if m:
        y, mo, d, h, mi, s, frac = m.groups()
        t = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s))
        if frac:
            t += timedelta(seconds=float("0." + frac))
        return t, line[:m.start()] + " " + line[m.end():]
    m = EPOCH_RE.search(line)
    if m:
        return datetime.fromtimestamp(float(m.group(1))), line[:m.start()] + " " + line[m.end():]
    return None, line


def parse_feeder_map(text: str) -> dict[str, str]:
    out = dict(DEFAULT_MAP)
    for part in re.split(r"[,;\n]+", text or ""):
        if "=" in part:
            k, v = (x.strip() for x in part.split("=", 1))
            v = v.lower()
            out[k.lower()] = v if v.startswith("feeder_") else f"feeder_{v}"
    return out


def read_log(path: Path, fmap: dict[str, str]) -> list[dict]:
    events = []
    for no, raw in enumerate(path.read_text(errors="replace").splitlines(), 1):
        t, rest = parse_time(raw)
        if t is None:
            continue
        kind = next((k for k, rx in KIND_RES if rx.search(rest)), None)
        if kind is None:
            continue
        feeder, token = "", ""
        if kind != "tap":
            for rx in FEEDER_RES:
                m = rx.search(rest)
                if m:
                    token = m.group(1)
                    feeder = fmap.get(token.lower(), "")
                    break
        p = None
        for rx in P_RES:
            m = rx.search(rest)
            if m:
                v = float(m.group(1))
                p = v / 100 if v > 1 else v
                break
        events.append(dict(line=no, time=t, event=kind, feeder=feeder, feeder_token=token, chance_logged=p, text=raw.strip()))
    events.sort(key=lambda e: (e["time"], e["line"]))
    return events


@dataclass
class Trial:
    n: int
    tap: datetime | None = None
    open: datetime | None = None
    close: datetime | None = None
    feeder: str = ""
    chance_logged: float | None = None
    notes: list = field(default_factory=list)


def build_trials(events: list[dict]) -> list[Trial]:
    trials: list[Trial] = []
    cur: Trial | None = None
    for e in events:
        if e["event"] == "tap":
            cur = Trial(len(trials) + 1, tap=e["time"])
            trials.append(cur)
        elif e["event"] == "open":
            if cur is None or cur.open is not None or (cur.tap and (e["time"] - cur.tap).total_seconds() > 600):
                cur = Trial(len(trials) + 1)
                cur.notes.append("no tap before this opening")
                trials.append(cur)
            cur.open, cur.feeder, cur.chance_logged = e["time"], e["feeder"], e["chance_logged"]
            if not e["feeder"]:
                cur.notes.append(f"feeder not recognised in '{e['text']}'")
        elif e["event"] == "close":
            if cur is not None and cur.open is not None and cur.close is None:
                cur.close = e["time"]
                if e["feeder"] and cur.feeder and e["feeder"] != cur.feeder:
                    cur.notes.append(f"closed {e['feeder']} but {cur.feeder} was open")
    for t in trials:
        if t.open is None:
            t.notes.append("tap without an opening")
        elif t.close is None:
            t.notes.append("no close found")
    return trials


def parse_conditions(text: str, n_trials: int) -> dict[int, float]:
    """'80' -> every trial 80 % left; '1-10:80, 11-20:20' -> per trial range. Returns trial -> chance left (0-1)."""
    text = (text or "").strip()
    out: dict[int, float] = {}
    if not text:
        return out
    if re.fullmatch(r"\d+(\.\d+)?", text):
        v = float(text)
        return {i: v / 100 if v > 1 else v for i in range(1, n_trials + 1)}
    for part in re.split(r"[,;\n]+", text):
        m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+|end))?\s*:\s*(\d+(?:\.\d+)?)\s*%?\s*", part)
        if not m:
            raise SystemExit(f"Could not read the chance setting '{part}'. Use e.g. 80  or  1-10:80, 11-20:20")
        a = int(m.group(1))
        b = n_trials if m.group(2) == "end" else int(m.group(2) or a)
        v = float(m.group(3))
        for i in range(a, b + 1):
            out[i] = v / 100 if v > 1 else v
    return out


# ------------------------------------------------------------------ video data

def video_start_from_name(name: str) -> datetime | None:
    m = DT_RE.search(Path(name).stem)
    if not m:
        return None
    y, mo, d, h, mi, s, _ = m.groups()
    return datetime(int(y), int(mo), int(d), int(h), int(mi), int(s))


def read_csv(path: Path) -> list[dict]:
    import csv
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


class Series:
    """Per-second values (ants at / inside each feeder, on tray) from tracking and survey runs."""

    def __init__(self, feeders: list[str]):
        self.feeders = feeders
        self.at: dict[str, dict[int, float]] = {f: {} for f in feeders}      # ants in the feeder area, outside the cup
        self.inside: dict[str, dict[int, float]] = {f: {} for f in feeders}
        self.tray: dict[int, float] = {}
        self.src: dict[int, str] = {}
        self.entries: dict[str, list[float]] = {f: [] for f in feeders}
        self.exits: dict[str, list[float]] = {f: [] for f in feeders}
        self.track_spans: list[tuple[float, float, str]] = []

    def add_tracking(self, run: Path) -> None:
        man = json.loads((run / "track_manifest.json").read_text())
        s0, dur = float(man["start_seconds"]), float(man["duration_seconds"])
        # where runs overlap, the first run added wins (so entries are not counted twice)
        taken = lambda t: any(a <= t < b for a, b, _ in self.track_spans)  # noqa: E731
        dead = {r["track_id"] for r in read_csv(run / "tracks_summary.csv") if r.get("final_state") == "dead"}
        per_frame: dict[float, dict[str, int]] = {}
        for r in read_csv(run / "tracks.csv"):
            t = float(r["t_sec"])
            d = per_frame.setdefault(t, {})
            if r["state"] in ("active", "merged", "lost") and r["track_id"] not in dead and r["area"] in self.at:
                d[r["area"]] = d.get(r["area"], 0) + 1
        acc: dict[int, list] = {}
        for t, d in per_frame.items():
            acc.setdefault(int(t), []).append(d)
        for sec, ds in acc.items():
            if self.src.get(sec) == "tracking":
                continue
            for f in self.feeders:
                self.at[f][sec] = float(np.mean([d.get(f, 0) for d in ds]))
            self.src[sec] = "tracking"
        for r in read_csv(run / "counts.csv"):
            sec = int(float(r["t_sec"]))
            if taken(sec):
                continue
            for f in self.feeders:
                if f"in_{f}" in r:
                    self.inside[f][sec] = float(r[f"in_{f}"])
            self.tray[sec] = float(r.get("ants_on_tray_excl_dead_objects") or r["ants_on_tray"])
        for e in read_csv(run / "feeder_events.csv"):
            if e["feeder"] in self.entries and not taken(float(e["t_sec"])):
                (self.entries if e["event"] == "in" else self.exits)[e["feeder"]].append(float(e["t_sec"]))
        self.track_spans.append((s0, s0 + dur, run.name))

    def add_survey(self, run: Path) -> None:
        pf = run / "analysis" / "per_frame.csv"
        if not pf.exists():
            raise SystemExit(f"{run}: no analysis/per_frame.csv - run the survey analysis first.")
        for r in read_csv(pf):
            sec = int(float(r["t_sec"]))
            if self.src.get(sec) == "tracking":
                continue
            for f in self.feeders:
                self.at[f][sec] = float(r.get(f"{f}__moving", 0) or 0) + float(r.get(f"{f}__stationary", 0) or 0)
            self.tray[sec] = float(r.get("tray__moving", 0) or 0) + float(r.get("tray__stationary", 0) or 0)
            self.src[sec] = "survey"

    def window(self, f: str, a: float, b: float) -> dict:
        secs = [s for s in range(int(math.floor(a)), int(math.ceil(b))) if s in self.src]
        tracked = [s for s in secs if self.src[s] == "tracking"]
        at = [self.at[f][s] for s in secs if s in self.at[f]]
        ins = [self.inside[f][s] for s in tracked if s in self.inside[f]]
        full = max(b - a, 1e-9)
        # survey samples every few seconds: coverage = sampled span, judged by gaps <= 30 s
        cov = 0.0
        if secs:
            cov = min(1.0, (len(tracked) + sum(1 for s in secs if self.src[s] == "survey") *
                            self._survey_step()) / full)
        return dict(at_mean=np.mean(at) if at else np.nan, inside_mean=np.mean(ins) if ins else np.nan,
                    entries=sum(a <= t < b for t in self.entries[f]) if tracked else np.nan,
                    exits=sum(a <= t < b for t in self.exits[f]) if tracked else np.nan,
                    coverage=round(cov, 2), source="tracking" if tracked else ("survey" if secs else ""))

    def _survey_step(self) -> float:
        if not hasattr(self, "_step"):
            ss = sorted(s for s, v in self.src.items() if v == "survey")
            self._step = float(np.median(np.diff(ss))) if len(ss) > 2 else 5.0
        return self._step


# ------------------------------------------------------------------ main

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", required=True, help="Raspberry Pi feeder log (text or CSV).")
    p.add_argument("--video", help="The recording (used for its start time and to check runs belong to it).")
    p.add_argument("--video-start", help="Camera clock at the first frame, e.g. '2026-11-07 02:20:04' (default: from the file name).")
    p.add_argument("--clock-offset", type=float, default=0.0, help="Seconds the Pi clock is ahead of the camera clock.")
    p.add_argument("--feeder-map", default="", help="Pi feeder name = cup, e.g. '1=left,2=right' (left/right/A/B/L/R/1/2 work by default).")
    p.add_argument("--chance-left", default="", help="Chance (%%) that the LEFT feeder opens: '80', or per trial range '1-10:80, 11-20:20'.")
    p.add_argument("--runs", nargs="*", default=[], help="Tracking run folders.")
    p.add_argument("--survey", nargs="*", default=[], help="Survey run folders (with analysis/).")
    p.add_argument("--feeders", default="feeder_left,feeder_right")
    p.add_argument("--before-min", type=float, default=5.0)
    p.add_argument("--after-min", type=float, default=5.0)
    p.add_argument("--peri-before-min", type=float, default=5.0, help="Time course from this long before the opening ...")
    p.add_argument("--peri-after-min", type=float, default=15.0, help="... to this long after it.")
    p.add_argument("--bin-seconds", type=float, default=30.0, help="Time-course bins.")
    p.add_argument("--output")
    p.add_argument("--check", action="store_true", help="Only read the log and list the trials (no results).")
    p.add_argument("--json", help="With --check: also write the trial list to this JSON file (used by the app).")
    return p


def share(a: float, b: float) -> float:
    return a / (a + b) if (a + b) > 0 and not (np.isnan(a) or np.isnan(b)) else np.nan


def fmt(v, nd=2):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else (round(v, nd) if isinstance(v, float) else v)


def main() -> None:
    a = build_parser().parse_args()
    import csv as _csv
    fmap = parse_feeder_map(a.feeder_map)
    feeders = [f.strip() for f in a.feeders.split(",") if f.strip()]
    events = read_log(Path(a.log), fmap)
    if not events:
        raise SystemExit("No events found in the log. Each line needs a date+time and the word tap, open or close.")
    trials = build_trials(events)
    start = (datetime.fromisoformat(a.video_start.strip().replace("_", " ")) if a.video_start
             else video_start_from_name(a.video or ""))
    if start is None:
        raise SystemExit("Video start time unknown: the file name has no date/time - give --video-start 'YYYY-MM-DD HH:MM:SS'.")
    vt = lambda t: None if t is None else (t - start).total_seconds() - a.clock_offset  # noqa: E731
    chance = parse_conditions(a.chance_left, len(trials))
    if not chance and any(t.chance_logged is not None for t in trials):
        print("[experiment] chances found in the log are listed as chance_logged; give --chance-left to use them.", flush=True)

    print(f"[experiment] log: {len(events)} events -> {len(trials)} trials; video start {start}; Pi clock offset {a.clock_offset:g} s",
          flush=True)
    rows = []
    for t in trials:
        rows.append(dict(trial=t.n, tap=t.tap, open=t.open, close=t.close, feeder=t.feeder,
                         tap_s=vt(t.tap), open_s=vt(t.open), close_s=vt(t.close),
                         chance_left=chance.get(t.n), chance_logged=t.chance_logged, notes="; ".join(t.notes)))
    for r in rows:
        print(f"  trial {r['trial']:>3}: tap {r['tap'] or '-'}  open {r['open'] or '-'}  {r['feeder'] or '?':<13} close {r['close'] or '-'}"
              f"   video {fmt_t(r['open_s'])}{'   ' + r['notes'] if r['notes'] else ''}", flush=True)
    if a.check:
        if a.json:
            Path(a.json).write_text(json.dumps([{k: (v.isoformat(sep=' ') if isinstance(v, datetime) else v) for k, v in r.items()}
                                                for r in rows], indent=1))
        return

    out = Path(a.output) if a.output else Path("FINAL/runs") / f"experiment_{Path(a.video or a.log).stem[:30].replace(' ', '_')}"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "log_events.csv").open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["line", "time", "video_sec", "event", "feeder", "feeder_token", "chance_logged", "text"])
        for e in events:
            w.writerow([e["line"], e["time"], fmt(vt(e["time"]), 1), e["event"], e["feeder"], e["feeder_token"], e["chance_logged"] or "", e["text"]])

    ser = Series(feeders)
    video_name = Path(a.video).name if a.video else None
    for r in a.runs:
        run = Path(r)
        if not (run / "track_manifest.json").exists():
            print(f"[experiment] skip {run}: not a tracking run", flush=True)
            continue
        man = json.loads((run / "track_manifest.json").read_text())
        if video_name and Path(man.get("video", "")).name != video_name:
            print(f"[experiment] skip {run.name}: made from {Path(man.get('video', '')).name}", flush=True)
            continue
        if not (run / "counts.csv").exists():
            print(f"[experiment] skip {run.name}: unfinished (no counts.csv)", flush=True)
            continue
        ser.add_tracking(run)
        print(f"[experiment] tracking {run.name}: {fmt_t(float(man['start_seconds']))} + {float(man['duration_seconds']) / 60:g} min", flush=True)
    for r in a.survey:
        ser.add_survey(Path(r))
        print(f"[experiment] survey {Path(r).name}", flush=True)
    if not ser.src:
        raise SystemExit("No tracking or survey data for this video - track the trial windows or run a survey first.")

    # ---- per trial x window x feeder
    win_rows, trial_out = [], []
    for r in rows:
        if r["open_s"] is None:
            continue
        o = r["open_s"]
        tap = r["tap_s"] if r["tap_s"] is not None else o - 60
        cl = r["close_s"] if r["close_s"] is not None else o + 600
        wins = dict(before=(tap - a.before_min * 60, tap), cue=(tap, o), open=(o, cl), after=(cl, cl + a.after_min * 60))
        res = {}
        for wn, (w0, w1) in wins.items():
            for f in feeders:
                m = ser.window(f, w0, w1)
                res[(wn, f)] = m
                win_rows.append(dict(trial=r["trial"], window=wn, feeder=f, opened=int(f == r["feeder"]), start_s=round(w0, 1),
                                     end_s=round(w1, 1), **{k: fmt(v) for k, v in m.items()}))
        p_left = r["chance_left"]
        rich = "" if p_left is None or p_left == 0.5 else ("feeder_left" if p_left > 0.5 else "feeder_right")
        other = next((f for f in feeders if f != r["feeder"]), "")
        L, Rr = (feeders + ["", ""])[:2]

        def tot(wn, f):
            m = res[(wn, f)]
            return (0 if np.isnan(m["at_mean"]) else m["at_mean"]) + (0 if np.isnan(m["inside_mean"]) else m["inside_mean"]) \
                if not (np.isnan(m["at_mean"]) and np.isnan(m["inside_mean"])) else np.nan
        ent_open = [t for t in ser.entries.get(r["feeder"], []) if o <= t < cl]
        prev = trial_out[-1]["opened"] if trial_out else ""
        trial_out.append(dict(
            trial=r["trial"], tap=r["tap"], open=r["open"], close=r["close"], opened=r["feeder"], chance_left=fmt(p_left),
            rich=rich, opened_rich="" if not rich else int(r["feeder"] == rich),
            tap_s=fmt(r["tap_s"], 1), open_s=fmt(o, 1), close_s=fmt(r["close_s"], 1),
            coverage_before=res[("before", L)]["coverage"], coverage_cue=res[("cue", L)]["coverage"],
            coverage_open=res[("open", L)]["coverage"], source_open=res[("open", L)]["source"],
            share_left_before=fmt(share(tot("before", L), tot("before", Rr))),
            share_left_cue=fmt(share(tot("cue", L), tot("cue", Rr))),
            share_at_opened_open=fmt(share(tot("open", r["feeder"]), tot("open", other))) if r["feeder"] in feeders else "",
            ants_inside_opened=fmt(res[("open", r["feeder"])]["inside_mean"]) if r["feeder"] in feeders else "",
            entries_opened=fmt(res[("open", r["feeder"])]["entries"]) if r["feeder"] in feeders else "",
            entries_closed=fmt(res[("open", other)]["entries"]) if other else "",
            first_entry_s=fmt(ent_open[0] - o, 1) if ent_open and res[("open", r["feeder"])]["source"] == "tracking" else "",
            ants_on_tray_open=fmt(np.mean([ser.tray[s] for s in range(int(o), int(cl)) if s in ser.tray])
                                  if any(s in ser.tray for s in range(int(o), int(cl))) else np.nan),
            share_last_opened_cue=fmt(share(tot("cue", prev), tot("cue", next(f for f in feeders if f != prev))))
            if prev in feeders else "",
            notes=r["notes"]))
    write(out / "trials.csv", trial_out)
    write(out / "trial_windows.csv", win_rows)

    # ---- per chance setting
    cond_rows = []
    groups: dict = {}
    for t in trial_out:
        groups.setdefault(t["chance_left"], []).append(t)
    for p_left, ts in sorted(groups.items(), key=lambda kv: (kv[0] == "", kv[0])):
        num = lambda k: [float(t[k]) for t in ts if t[k] not in ("", None)]  # noqa: E731
        cond_rows.append(dict(chance_left=p_left, trials=len(ts),
                              opened_left=sum(t["opened"] == "feeder_left" for t in ts),
                              opened_right=sum(t["opened"] == "feeder_right" for t in ts),
                              mean_share_left_before=fmt(np.mean(num("share_left_before")) if num("share_left_before") else np.nan),
                              mean_share_left_cue=fmt(np.mean(num("share_left_cue")) if num("share_left_cue") else np.nan),
                              mean_share_at_opened_open=fmt(np.mean(num("share_at_opened_open")) if num("share_at_opened_open") else np.nan),
                              mean_entries_opened=fmt(np.mean(num("entries_opened")) if num("entries_opened") else np.nan),
                              median_first_entry_s=fmt(float(np.median(num("first_entry_s"))) if num("first_entry_s") else np.nan, 1),
                              trials_with_data=sum(t["source_open"] != "" for t in ts)))
    write(out / "conditions.csv", cond_rows)

    # ---- time course around the opening
    edges = np.arange(-a.peri_before_min * 60, a.peri_after_min * 60 + 1e-9, a.bin_seconds)
    peri = {k: [[] for _ in range(len(edges) - 1)] for k in ("opened_at", "opened_inside", "closed_at", "tray")}
    for r in rows:
        if r["open_s"] is None or r["feeder"] not in feeders:
            continue
        other = next(f for f in feeders if f != r["feeder"])
        for i in range(len(edges) - 1):
            s0, s1 = r["open_s"] + edges[i], r["open_s"] + edges[i + 1]
            secs = [s for s in range(int(s0), int(s1)) if s in ser.src]
            if not secs:
                continue
            peri["opened_at"][i].append(np.mean([ser.at[r["feeder"]].get(s, 0) for s in secs]))
            peri["closed_at"][i].append(np.mean([ser.at[other].get(s, 0) for s in secs]))
            ins = [ser.inside[r["feeder"]][s] for s in secs if s in ser.inside[r["feeder"]]]
            if ins:
                peri["opened_inside"][i].append(np.mean(ins))
            tr = [ser.tray[s] for s in secs if s in ser.tray]
            if tr:
                peri["tray"][i].append(np.mean(tr))
    peri_rows = [dict(t_min=round((edges[i] + edges[i + 1]) / 2 / 60, 2), trials=len(peri["opened_at"][i]),
                      **{k: fmt(np.mean(v[i]) if v[i] else np.nan) for k, v in peri.items()}) for i in range(len(edges) - 1)]
    write(out / "peri_opening.csv", peri_rows)

    # ---- matching (needs >= 2 chance settings with data)
    match = None
    pts = []
    for t in trial_out:
        if t["chance_left"] in ("", None) or t["share_left_cue"] in ("", None) or not (0 < float(t["chance_left"]) < 1):
            continue
        pts.append((float(t["chance_left"]), float(t["share_left_cue"])))
    if len({p for p, _ in pts}) >= 2:
        eps = 0.02
        x = np.array([math.log(p / (1 - p)) for p, _ in pts])
        y = np.array([math.log((s + eps) / (1 - s + eps)) for _, s in pts])
        sl, ic = np.polyfit(x, y, 1)
        match = dict(sensitivity_s=round(float(sl), 3), bias_log_b=round(float(ic), 3), trials=len(pts),
                     note="Generalized matching law on the cue window: log(at left / at right) = s * log(chance left / chance right) + log b")

    rel_tap = [r["tap_s"] - r["open_s"] for r in rows if r["tap_s"] is not None and r["open_s"] is not None]
    rel_close = [r["close_s"] - r["open_s"] for r in rows if r["close_s"] is not None and r["open_s"] is not None]
    marks = dict(tap=float(np.median(rel_tap)) / 60 if rel_tap else None, close=float(np.median(rel_close)) / 60 if rel_close else None)
    plots(out, trial_out, peri_rows, pts, match, feeders, marks)
    n_cov = sum(t["source_open"] != "" for t in trial_out)
    summary = dict(log=str(a.log), video=a.video, video_start=str(start), clock_offset_s=a.clock_offset, trials=len(rows),
                   trials_with_opening=len(trial_out), trials_with_data=n_cov,
                   opened={f: sum(t["opened"] == f for t in trial_out) for f in feeders},
                   tracking_runs=[n for *_, n in ser.track_spans], survey_runs=[Path(s).name for s in a.survey],
                   conditions=cond_rows, matching=match,
                   problems=[f"trial {r['trial']}: {r['notes']}" for r in rows if r["notes"]])
    (out / "experiment_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    report(out, summary, trial_out)
    print(f"[experiment] {len(trial_out)} trials with an opening, {n_cov} with video data -> {out}", flush=True)


def fmt_t(sec) -> str:
    if sec is None:
        return "-"
    sgn = "-" if sec < 0 else ""
    sec = int(round(abs(sec)))
    return f"{sgn}{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def write(path: Path, rows: list[dict]) -> None:
    import csv
    with path.open("w", newline="") as fh:
        if not rows:
            fh.write("")
            return
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plots(out: Path, trials: list[dict], peri: list[dict], pts, match, feeders, marks: dict | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # time course
    fig, ax = plt.subplots(figsize=(9, 4.2))
    t = [r["t_min"] for r in peri]
    for k, lab, c in [("opened_at", "at the opened feeder (outside cup)", "#d1495b"), ("opened_inside", "inside the opened cup", "#8f2d56"),
                      ("closed_at", "at the closed feeder", "#00798c")]:
        y = [np.nan if r[k] == "" else r[k] for r in peri]
        ax.plot(t, y, label=lab, color=c, lw=2)
    top = ax.get_ylim()[1]
    for x, lab in [(0, "open"), ((marks or {}).get("tap"), "tap"), ((marks or {}).get("close"), "close")]:
        if x is not None:
            ax.axvline(x, color="k", lw=1, ls="-" if lab == "open" else ":")
            ax.text(x, top, f" {lab}", va="top", fontsize=9)
    ax.set_xlabel("minutes from the opening")
    ax.set_ylabel("ants (mean over trials)")
    n = max((r["trials"] for r in peri), default=0)
    ax.set_title(f"Ants around the opening ({n} trials with data)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "peri_opening.png", dpi=110); plt.close(fig)
    # per trial
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for tr in trials:
        x = tr["trial"]
        if tr["share_left_cue"] not in ("", None):
            ax.plot(x, tr["share_left_cue"], "o", color="#2a6fdb")
        if tr["share_left_before"] not in ("", None):
            ax.plot(x, tr["share_left_before"], ".", color="#9db8ec")
        ax.text(x, 1.04, "L" if tr["opened"] == "feeder_left" else ("R" if tr["opened"] == "feeder_right" else "?"),
                ha="center", fontsize=8, color="#d1495b")
        if tr["chance_left"] not in ("", None):
            ax.plot([x - 0.4, x + 0.4], [tr["chance_left"]] * 2, color="#555", lw=1)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("trial   (letters: feeder that opened)")
    ax.set_ylabel("share of ants at the LEFT feeder")
    ax.set_title("Per trial: share at left before the opening (dot: cue, light: 5 min before; line: chance left)")
    fig.tight_layout(); fig.savefig(out / "trial_shares.png", dpi=110); plt.close(fig)
    # matching
    if match:
        fig, ax = plt.subplots(figsize=(4.8, 4.5))
        ax.plot([p for p, _ in pts], [s for _, s in pts], "o", alpha=0.6)
        ax.plot([0, 1], [0, 1], "--", color="#999", label="perfect matching")
        ax.set_xlabel("chance LEFT opens")
        ax.set_ylabel("share of ants at left (cue window)")
        ax.set_title(f"Matching: s = {match['sensitivity_s']}, log b = {match['bias_log_b']}")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "matching.png", dpi=110); plt.close(fig)


def report(out: Path, s: dict, trials: list[dict]) -> None:
    L = [f"Experiment results - {Path(s['video'] or '').name}", "",
         f"Log: {s['trials']} trials, {s['trials_with_opening']} with an opening; {s['trials_with_data']} have video data "
         f"(tracking runs: {len(s['tracking_runs'])}, surveys: {len(s['survey_runs'])}).",
         "Opened: " + ", ".join(f"{k.replace('feeder_', '')} {v}" for k, v in s["opened"].items()) + "."]
    d = lambda v: "-" if v in ("", None) else v  # noqa: E731
    for c in s["conditions"]:
        L.append(f"Chance left {c['chance_left'] if c['chance_left'] not in ('', None) else '(not given)'}: {c['trials']} trials "
                 f"(left opened {c['opened_left']}, right {c['opened_right']}; {c['trials_with_data']} with video data). "
                 f"Share of ants at the left feeder before the opening: {d(c['mean_share_left_cue'])} (tap to opening), "
                 f"{d(c['mean_share_left_before'])} (before the tap). While open: share at the opened feeder "
                 f"{d(c['mean_share_at_opened_open'])}, entries into the opened cup {d(c['mean_entries_opened'])} per trial, "
                 f"first entry after {d(c['median_first_entry_s'])} s (median).")
    if s["matching"]:
        m = s["matching"]
        L.append(f"Matching law: sensitivity s = {m['sensitivity_s']} (1 = matching, < 1 under-, > 1 over-matching), "
                 f"bias log b = {m['bias_log_b']} (> 0 = left bias).")
    if s["problems"]:
        L += ["", "Check these log entries:"] + [f"  {p}" for p in s["problems"]]
    (out / "report.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
