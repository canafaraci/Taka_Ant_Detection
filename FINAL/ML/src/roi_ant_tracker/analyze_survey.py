"""Summarise a survey-video run: ant counts by area and state, feeder occupancy, heatmaps.

Reads the per-hour CSVs written by ``roi_ant_tracker.survey`` and labels every detection:

* permanent  - stays within --still-radius px of one spot for at least --permanent-hours (fixed marks,
               dead ants); short detector misses (--max-gap-seconds) are bridged;
* stationary - stays at one spot for at least --still-seconds but less than --permanent-hours;
* moving     - everything else.

Still ants are counted (Can, 2026-10-11); the state column lets them be split out.

Outputs (in --output, default <survey>/analysis):
  detections_labeled.csv  every detection + hour, state, still_id, still_seconds
  per_frame.csv           one row per sampled frame: counts per area x state
  hourly.csv              per hour means, feeder occupancy, feeder ant-seconds
  feeder_episodes.csv     runs of consecutive samples with >=1 non-permanent ant in a feeder area
  permanent_objects.csv   locations flagged permanent, with hours present
  hourly_counts.png, feeder_occupancy.png, heatmap_moving.png, heatmap_stationary.png
  summary.json

Needs pandas and matplotlib (pip install pandas matplotlib).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

STATES = ["moving", "stationary", "permanent"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--survey", required=True, help="Folder written by survey-video.")
    p.add_argument("--output", help="Default: <survey>/analysis")
    p.add_argument("--areas", help="areas.json (default: path in survey_manifest.json)")
    p.add_argument("--background", help="Image for heatmaps (default: grab a frame from the video if readable).")
    p.add_argument("--still-radius", type=float, default=6.0, help="px; centre jitter of a resting ant is ~1-2 px.")
    p.add_argument("--still-seconds", type=float, default=60.0)
    p.add_argument("--max-gap-seconds", type=float, default=60.0, help="Missed time bridged inside one still chain.")
    p.add_argument("--permanent-hours", type=float, default=2.0, help="Still this long -> permanent (debris, dead ants).")
    p.add_argument("--heatmap-bin", type=int, default=24, help="px")
    p.add_argument("--episode-gap", type=int, default=1, help="Missing samples allowed inside one feeder episode.")
    return p


# ------------------------------------------------------------------ loading

def load_survey(folder: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    man = json.loads((folder / "survey_manifest.json").read_text())
    step, n_frames, fps = int(man["step_frames"]), int(man["frames"]), float(man["fps"])
    parts, sampled = [], []
    for csv in sorted(folder.glob("*.csv")):
        m_hour, m_from = re.fullmatch(r"hour_(\d+)\.csv", csv.name), re.fullmatch(r"from_(\d+)s\.csv", csv.name)
        if not (m_hour or m_from):
            continue
        df = pd.read_csv(csv)
        parts.append(df)
        if m_hour:
            h = int(m_hour.group(1))
            f0, f1 = int(h * 3600 * fps), min(n_frames, int((h + 1) * 3600 * fps))
        else:  # pilot block: duration not recorded, use the detected span
            f0 = int(int(m_from.group(1)) * fps)
            f1 = int(df.frame.max()) + 1 if len(df) else f0
        first = ((f0 + step - 1) // step) * step
        sampled.append(np.arange(first, f1, step))
    if not parts:
        raise SystemExit(f"No finished hour_XXX.csv / from_XXXs.csv files in {folder}")
    det = pd.concat(parts, ignore_index=True)
    frames = np.unique(np.concatenate(sampled))
    det["hour"] = (det.t_sec // 3600).astype(int)
    return det, frames, man


# ------------------------------------------------------------------ states

def link_still(det: pd.DataFrame, frames: np.ndarray, radius: float, max_gap_s: float, step_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Chain detections that stay at one spot.

    A detection joins an open chain when it lies within `radius` px of the chain's mean position and the chain was
    last seen at most `max_gap_s` earlier (detector misses and occlusions are bridged). Returns chain id and
    chain duration (s) for every detection.
    """
    pos = {f: i for i, f in enumerate(frames)}
    k_all = det.frame.map(pos).values
    cx_all, cy_all = det.cx.values, det.cy.values
    order = np.argsort(k_all, kind="stable")
    gap_k = int(round(max_gap_s / step_s)) + 1
    chain = np.full(len(det), -1)
    # open chains: id -> [sum_x, sum_y, n, last_k, first_t, last_t]
    open_: dict[int, list] = {}
    t_all = det.t_sec.values
    next_id, i = 0, 0
    while i < len(order):
        k = k_all[order[i]]
        j = i
        while j < len(order) and k_all[order[j]] == k:
            j += 1
        idx = order[i:j]
        for cid in [c for c, v in open_.items() if k - v[3] > gap_k]:
            del open_[cid]
        ids = list(open_)
        if ids:
            mx = np.array([open_[c][0] / open_[c][2] for c in ids]); my = np.array([open_[c][1] / open_[c][2] for c in ids])
            d = np.hypot(cx_all[idx][:, None] - mx[None], cy_all[idx][:, None] - my[None])
            pairs = sorted(zip(*np.where(d <= radius)), key=lambda ab: d[ab])
        else:
            pairs = []
        used_d, used_c = set(), set()
        for a_, b_ in pairs:
            if a_ in used_d or b_ in used_c:
                continue
            used_d.add(a_); used_c.add(b_)
            chain[idx[a_]] = ids[b_]
        for n_, di in enumerate(idx):
            if chain[di] < 0:
                chain[di] = next_id; open_[next_id] = [0.0, 0.0, 0, k, t_all[di], t_all[di]]; next_id += 1
            v = open_[chain[di]]
            v[0] += cx_all[di]; v[1] += cy_all[di]; v[2] += 1; v[3] = k; v[5] = t_all[di]
        i = j
    s = pd.Series(chain, index=det.index)
    span = det.groupby(s).t_sec.agg(["min", "max"])
    return chain, s.map(span["max"] - span["min"]).values


def permanent_table(det: pd.DataFrame, step_s: float) -> pd.DataFrame:
    p = det[det.state == "permanent"]
    if not len(p):
        return pd.DataFrame(columns=["still_id", "cx", "cy", "area", "first_hour", "last_hour", "duration_h", "fraction_present"])
    g = p.groupby("still_id")
    t = g.agg(cx=("cx", "mean"), cy=("cy", "mean"), area=("area", lambda a: a.mode().iat[0]),
              first_sec=("t_sec", "min"), last_sec=("t_sec", "max"), n=("frame", "nunique"), score=("score", "mean"))
    # merge chains of one object that split when it was nudged a few px (or the camera shifted slightly)
    t = t.sort_values("first_sec").reset_index()
    obj = np.arange(len(t))
    for i in range(len(t)):
        for j in range(i):
            if (np.hypot(t.cx[i] - t.cx[j], t.cy[i] - t.cy[j]) <= 20 and -600 <= t.first_sec[i] - t.last_sec[j] <= 3600):
                obj[i] = obj[j]
                break
    t["object"] = obj
    t = t.groupby("object").agg(still_id=("still_id", lambda v: " ".join(map(str, v))), cx=("cx", "mean"), cy=("cy", "mean"),
                                area=("area", "first"), first_sec=("first_sec", "min"), last_sec=("last_sec", "max"),
                                n=("n", "sum"), score=("score", "mean"))
    t["duration_h"] = (t.last_sec - t.first_sec) / 3600
    t["fraction_present"] = t.n * step_s / (t.last_sec - t.first_sec + step_s)
    t["first_hour"], t["last_hour"] = t.first_sec / 3600, t.last_sec / 3600
    return t.reset_index(drop=True)[["still_id", "cx", "cy", "area", "first_hour", "last_hour", "duration_h", "fraction_present", "score"]].round(2).sort_values("first_hour")


# ------------------------------------------------------------------ summaries

def episodes(per_frame: pd.DataFrame, col: str, step_s: float, gap: int) -> list[dict]:
    on = per_frame[col].values > 0
    t = per_frame.t_sec.values
    k = per_frame.k.values
    out, start, last, peak = [], None, None, 0
    for i in range(len(on)):
        if on[i]:
            if start is not None and k[i] - k[last] > gap + 1:
                out.append((start, last, peak)); start = None
            if start is None:
                start, peak = i, 0
            last, peak = i, max(peak, per_frame[col].values[i])
    if start is not None:
        out.append((start, last, peak))
    return [dict(start_sec=t[a], end_sec=t[b], duration_sec=t[b] - t[a] + step_s, samples=int((k[b] - k[a]) + 1),
                 max_ants=int(p)) for a, b, p in out]


def grab_background(man: dict, frame: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(man["video"])
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def heatmap(det: pd.DataFrame, size: tuple[int, int], bin_px: int, bg: np.ndarray | None, title: str, path: Path, n_frames: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w, h = size
    hist, _, _ = np.histogram2d(det.cy, det.cx, bins=[h // bin_px, w // bin_px], range=[[0, h], [0, w]])
    hist = cv2.GaussianBlur(hist.astype(np.float32), (0, 0), 1.0) / max(1, n_frames)
    fig, ax = plt.subplots(figsize=(12, 6.9), dpi=130)
    if bg is not None:
        ax.imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), cmap="gray", extent=[0, w, h, 0], alpha=0.55)
    from matplotlib.colors import LogNorm
    pos_vals = hist[hist > 0]
    vmax = float(np.percentile(pos_vals, 99.5)) if len(pos_vals) else 1.0
    vmin = max(vmax / 300, float(pos_vals.min()) if len(pos_vals) else vmax / 300)
    masked = np.ma.masked_less(hist, vmin)
    im = ax.imshow(masked, cmap="inferno", extent=[0, w, h, 0], alpha=0.85, interpolation="bilinear",
                   norm=LogNorm(vmin=vmin, vmax=vmax))
    fig.colorbar(im, ax=ax, fraction=0.025, label=f"ants per sampled frame per {bin_px}x{bin_px} px")
    ax.set_title(f"{title}  ({len(det):,} detections, {n_frames:,} frames)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def charts(hourly: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"moving": "#2a6fdb", "stationary": "#e3a21a", "permanent": "#9aa0a6"}
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=130)
    bottom = np.zeros(len(hourly))
    for s in STATES:
        v = hourly[f"mean_{s}"].values
        ax.bar(hourly.hour, v, bottom=bottom, color=colors[s], label=s, width=0.85)
        bottom += v
    ax.set_xlabel("hour of recording"); ax.set_ylabel("ants per frame (mean)")
    ax.set_title("Ants detected on the tray, by state"); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "hourly_counts.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=130)
    for col, c, lab in [("pct_frames_feeder_left", "#d1495b", "left feeder"), ("pct_frames_feeder_right", "#00798c", "right feeder")]:
        if col in hourly:
            ax.plot(hourly.hour, hourly[col], marker="o", ms=3, color=c, label=lab)
    ax.set_xlabel("hour of recording"); ax.set_ylabel("% of frames with >=1 ant in feeder area")
    ax.set_title("Feeder occupancy (moving + stationary ants; permanent spots excluded)"); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "feeder_occupancy.png"); plt.close(fig)


# ------------------------------------------------------------------ main

def main() -> None:
    a = build_parser().parse_args()
    folder = Path(a.survey)
    out = Path(a.output) if a.output else folder / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    det, frames, man = load_survey(folder)
    fps, step_s = float(man["fps"]), float(man["step_seconds"])
    areas_path = a.areas or man.get("areas")
    area_names = ["tray"]
    if areas_path and Path(areas_path).exists():
        area_names = [x["id"] for x in json.loads(Path(areas_path).read_text())["areas"]] + ["tray"]
    area_names += [x for x in sorted(det.area.unique()) if x not in area_names]

    det["still_id"], det["still_seconds"] = link_still(det, frames, a.still_radius, a.max_gap_seconds, step_s)
    det["state"] = np.where(det.still_seconds >= a.permanent_hours * 3600, "permanent",
                            np.where(det.still_seconds >= a.still_seconds, "stationary", "moving"))
    det.to_csv(out / "detections_labeled.csv", index=False)
    perm = permanent_table(det, step_s)
    perm.to_csv(out / "permanent_objects.csv", index=False)

    # per frame: zero rows for frames with no detections
    counts = det.groupby(["frame", "area", "state"]).size().unstack(["area", "state"], fill_value=0)
    cols = pd.MultiIndex.from_product([area_names, STATES])
    counts = counts.reindex(index=frames, columns=cols, fill_value=0)
    counts.columns = [f"{ar}__{st}" for ar, st in counts.columns]
    pf = counts.reset_index().rename(columns={"index": "frame"})
    pf.insert(1, "t_sec", pf.frame / fps)
    pf.insert(2, "hour", (pf.t_sec // 3600).astype(int))
    pf.insert(3, "k", np.arange(len(pf)))
    for ar in area_names:
        pf[f"{ar}__ants"] = pf[f"{ar}__moving"] + pf[f"{ar}__stationary"]   # non-permanent
    pf["all__moving"] = sum(pf[f"{ar}__moving"] for ar in area_names)
    pf["all__stationary"] = sum(pf[f"{ar}__stationary"] for ar in area_names)
    pf["all__permanent"] = sum(pf[f"{ar}__permanent"] for ar in area_names)
    pf["all__ants"] = pf.all__moving + pf.all__stationary
    pf.drop(columns="k").to_csv(out / "per_frame.csv", index=False)

    g = pf.groupby("hour")
    hourly = pd.DataFrame({"frames": g.size()})
    for s in STATES:
        hourly[f"mean_{s}"] = g[f"all__{s}"].mean()
    hourly["mean_ants"] = g.all__ants.mean()
    hourly["max_ants"] = g.all__ants.max()
    for ar in area_names:
        hourly[f"mean_{ar}"] = g[f"{ar}__ants"].mean()
        if ar.startswith("feeder"):
            hourly[f"pct_frames_{ar}"] = g[f"{ar}__ants"].apply(lambda v: 100 * (v > 0).mean())
            hourly[f"ant_seconds_{ar}"] = g[f"{ar}__ants"].sum() * step_s
    hourly = hourly.reset_index().round(3)
    hourly.to_csv(out / "hourly.csv", index=False)

    eps = []
    for ar in [x for x in area_names if x.startswith("feeder")]:
        eps += [dict(area=ar, **e) for e in episodes(pf, f"{ar}__ants", step_s, a.episode_gap)]
    pd.DataFrame(eps, columns=["area", "start_sec", "end_sec", "duration_sec", "samples", "max_ants"]).to_csv(out / "feeder_episodes.csv", index=False)

    charts(hourly, out)
    bg = cv2.imread(a.background) if a.background else grab_background(man, int(frames[len(frames) // 2]))
    size = tuple(man["size"])
    heatmap(det[det.state == "moving"], size, a.heatmap_bin, bg, "Moving ants", out / "heatmap_moving.png", len(frames))
    heatmap(det[det.state == "stationary"], size, a.heatmap_bin, bg, "Stationary ants (still >= %gs)" % a.still_seconds, out / "heatmap_stationary.png", len(frames))

    summary = dict(
        survey=str(folder.resolve()), hours=sorted(map(int, hourly.hour)), sampled_frames=int(len(frames)),
        step_seconds=step_s, detections=int(len(det)), detections_by_state=det.state.value_counts().to_dict(),
        mean_ants_per_frame=round(float(pf.all__ants.mean()), 2),
        mean_by_state={s: round(float(pf[f"all__{s}"].mean()), 2) for s in STATES},
        permanent_objects=int(len(perm)),
        feeder_episodes={ar: int(sum(e["area"] == ar for e in eps)) for ar in area_names if ar.startswith("feeder")},
        settings=dict(still_radius=a.still_radius, still_seconds=a.still_seconds, max_gap_seconds=a.max_gap_seconds,
                      permanent_hours=a.permanent_hours, episode_gap=a.episode_gap),
        note="Feeder episodes = consecutive samples with >=1 non-permanent ant in the feeder rectangle; not yet a defined 'visit'.",
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[analyze] wrote {out}")


if __name__ == "__main__":
    main()
