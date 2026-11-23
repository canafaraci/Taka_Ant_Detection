"""Ant tracking app: one window to set zones, run tracking or the whole-video survey, and look at results.

Start it from the repository root (the folder that contains FINAL):
    python FINAL/pipeline.py app
"""
from __future__ import annotations

import csv
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

APP_DIR = Path(__file__).resolve().parent
FINAL = APP_DIR.parent
REPO = FINAL.parent
RUNS = FINAL / "runs"
sys.path.insert(0, str(APP_DIR))
from zones_io import DEFAULTS, load_zones, save_zones, zones_complete  # noqa: E402

SETTINGS = Path.home() / ".taka_ant_app.json"
DEFAULT_CKPT = FINAL / "checkpoints" / "ours_v2_best.pt"
DEFAULT_ZONES = RUNS / "zones"
DEVICE_SETTINGS = FINAL / "device_settings.json"   # written by check_speed.py for this computer


def measured_device() -> dict:
    try:
        return json.loads(DEVICE_SETTINGS.read_text())
    except Exception:
        return {}


BATCH_DEFAULT = str(measured_device().get("batch", 8))
DEVICE_CHOICES = ["auto", "cuda", "cpu", "mps"]


# ------------------------------------------------------------------ helpers

def load_settings() -> dict:
    try:
        return json.loads(SETTINGS.read_text())
    except Exception:
        return {}


def save_settings(d: dict) -> None:
    try:
        SETTINGS.write_text(json.dumps(d, indent=2))
    except Exception:
        pass


def parse_time(text: str) -> float:
    """'10:00:00', '10:00', '36000', '36000s', '10h', '90m' -> seconds."""
    t = text.strip().lower()
    if not t:
        return 0.0
    if ":" in t:
        parts = [float(p) for p in t.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    m = re.fullmatch(r"([\d.]+)\s*([hms]?)", t)
    if not m:
        raise ValueError(f"Cannot read time '{text}'")
    v = float(m.group(1))
    return v * {"h": 3600, "m": 60, "s": 1, "": 1}[m.group(2)]


def fmt_time(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def open_path(p: Path) -> None:
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(p)])
    elif os.name == "nt":
        os.startfile(str(p))  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", str(p)])


def video_info(path: str) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, n, w, h


def grab_frame(path: str, sec: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def rel(p: str | Path) -> str:
    """Path relative to the repository root when possible (commands run from there)."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------------ job runner (one job at a time)

class Runner:
    def __init__(self, app: "App"):
        self.app = app
        self.proc: subprocess.Popen | None = None
        self.q: queue.Queue = queue.Queue()
        self.steps: list[list[str]] = []
        self.on_done = None

    @property
    def busy(self) -> bool:
        return self.proc is not None

    def run(self, steps: list[list[str]], title: str, on_done=None) -> None:
        if self.busy:
            messagebox.showwarning("Busy", "A job is already running. Stop it first or wait for it to finish.")
            return
        self.steps, self.on_done = list(steps), on_done
        self.app.log_clear()
        self.app.log(f"=== {title} ===\n")
        self.app.set_status(title, running=True)
        self._next()

    def _next(self) -> None:
        if not self.steps:
            self.proc = None
            self.app.set_status("Finished", running=False)
            if self.on_done:
                self.app.after(100, self.on_done)
            return
        args = self.steps.pop(0)
        cmd = [sys.executable, str(FINAL / "pipeline.py"), *args]
        self.app.log("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd[1:]) + "\n")
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        self.proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     bufsize=1, env=env)
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()
        self.app.after(200, self._poll)

    def _pump(self, proc) -> None:
        for line in proc.stdout:
            self.q.put(line)
        proc.wait()
        self.q.put(("__exit__", proc.returncode))

    def _poll(self) -> None:
        try:
            while True:
                item = self.q.get_nowait()
                if isinstance(item, tuple):
                    code = item[1]
                    if code != 0:
                        self.app.log(f"\n[stopped with exit code {code}]\n")
                        self.proc, self.steps = None, []
                        self.app.set_status("Stopped / failed - see log", running=False)
                        return
                    self._next()
                    return
                self.app.log(item)
                m = re.search(r"(\d+)/(\d+) frames", item)
                if m:
                    self.app.set_progress(int(m.group(1)), int(m.group(2)))
                m = re.search(r"([\d.]+)x real time", item)
                if m:
                    self.app.set_status(f"Running  ({m.group(1)}x real time)", running=True)
        except queue.Empty:
            pass
        self.app.after(200, self._poll)

    def stop(self) -> None:
        if self.proc:
            self.steps = []
            self.proc.terminate()
            self.app.log("\n[stop requested]\n")


# ------------------------------------------------------------------ shared field widgets

class Tooltip:
    def __init__(self, widget, text):
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _e=None):
        if self.tip or not self.text:
            return
        x, y = self.widget.winfo_rootx() + 10, self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left", background="#ffffe0", relief="solid", borderwidth=1,
                 wraplength=360).pack(ipadx=4, ipady=2)

    def hide(self, _e=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None

class PathField(ttk.Frame):
    def __init__(self, master, label, var, kind="file", filetypes=None, width=60):
        super().__init__(master)
        ttk.Label(self, text=label, width=14).pack(side="left")
        ttk.Entry(self, textvariable=var, width=width).pack(side="left", fill="x", expand=True)
        ttk.Button(self, text="Browse...", command=lambda: self.browse(var, kind, filetypes)).pack(side="left", padx=4)

    @staticmethod
    def browse(var, kind, filetypes):
        start = str(Path(var.get()).parent) if var.get() else str(REPO)
        if kind == "dir":
            p = filedialog.askdirectory(initialdir=start)
        else:
            p = filedialog.askopenfilename(initialdir=start, filetypes=filetypes or [("All files", "*.*")])
        if p:
            var.set(p)


# ------------------------------------------------------------------ Zones tab

class ZonesTab(ttk.Frame):
    TOOLS = [("Tray outline", "tray"), ("Nest slide (excluded)", "slide"), ("Nest opening", "nest"),
             ("Left feeder cup", "feeder_left"), ("Right feeder cup", "feeder_right")]
    HELP = {
        "tray": "Click around the edge of the tray floor, point by point. Press 'Close outline' when done.",
        "slide": "Click two opposite corners of the nest slide. Ants are not detected on it.",
        "nest": "Click the centre of the nest opening (where ants go in and out).",
        "feeder_left": "Click the centre of the left cup, then a point on its rim.",
        "feeder_right": "Click the centre of the right cup, then a point on its rim.",
    }

    def __init__(self, master, app: "App"):
        super().__init__(master)
        self.app = app
        self.img: np.ndarray | None = None
        self.scale = 1.0
        self.z: dict = {"feeders": {}}
        self.pending: list = []
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)
        PathField(top, "Video", app.video, filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")]).pack(fill="x")
        PathField(top, "Zones folder", app.zones_dir, kind="dir").pack(fill="x", pady=2)
        row = ttk.Frame(top)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Frame at", width=14).pack(side="left")
        self.t_var = tk.StringVar(value="10:00:00")
        ttk.Entry(row, textvariable=self.t_var, width=10).pack(side="left")
        ttk.Button(row, text="Load frame", command=self.load_frame).pack(side="left", padx=4)
        ttk.Button(row, text="Load zones from folder", command=self.load_existing).pack(side="left", padx=4)
        ttk.Button(row, text="Save zones", command=self.save).pack(side="left", padx=12)
        self.info = ttk.Label(row, text="")
        self.info.pack(side="left", padx=8)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=8, pady=4)
        side = ttk.Frame(body)
        side.pack(side="left", fill="y")
        ttk.Label(side, text="Tool", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        self.tool = tk.StringVar(value="nest")
        for label, key in self.TOOLS:
            ttk.Radiobutton(side, text=label, value=key, variable=self.tool, command=self.tool_changed).pack(anchor="w")
        ttk.Button(side, text="Close outline", command=self.close_outline).pack(fill="x", pady=(8, 2))
        ttk.Button(side, text="Clear this zone", command=self.clear_zone).pack(fill="x", pady=2)
        self.help = ttk.Label(side, text="", wraplength=200, foreground="#444")
        self.help.pack(anchor="w", pady=8)
        ttk.Separator(side).pack(fill="x", pady=6)
        ttk.Label(side, text="Sizes (px, full frame)", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        self.size_vars = {}
        for key, label in [("nest_clearance", "Ignore circle at opening"), ("nest_enter", "Counts as going in"),
                           ("nest_birth", "New ants may appear"), ("hide_margin", "Cup: hidden margin"),
                           ("exit_margin", "Cup: coming-out margin")]:
            f = ttk.Frame(side)
            f.pack(fill="x")
            ttk.Label(f, text=label, width=22).pack(side="left")
            v = tk.StringVar(value=str(DEFAULTS[key]))
            ttk.Entry(f, textvariable=v, width=6).pack(side="left")
            v.trace_add("write", lambda *_: self.redraw())
            self.size_vars[key] = v
        self.canvas = tk.Canvas(body, bg="#222", width=1100, height=620, highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.tool_changed()

    # -- data
    def sizes(self) -> dict:
        out = {}
        for k, v in self.size_vars.items():
            try:
                out[k] = int(float(v.get()))
            except ValueError:
                out[k] = DEFAULTS[k]
        return out

    def load_frame(self):
        v = self.app.video.get()
        if not v or not Path(v).exists():
            messagebox.showerror("Video", "Choose a video file first.")
            return
        try:
            t = parse_time(self.t_var.get())
        except ValueError as e:
            messagebox.showerror("Time", str(e))
            return
        fps, n, w, h = video_info(v)
        if n and t * fps >= n:                         # past the end: use the middle of the video
            t = n / fps / 2
            self.t_var.set(fmt_time(t))
        img = grab_frame(v, t)
        if img is None:
            messagebox.showerror("Video", "Could not read a frame at that time.")
            return
        self.img = img
        self.z["frame_h"], self.z["frame_w"] = img.shape[:2]
        self.info.config(text=f"{w}x{h}, {fps:.0f} fps, {fmt_time(n / fps)} long")
        self.redraw()

    def load_existing(self):
        d = Path(self.app.zones_dir.get() or DEFAULT_ZONES)
        if not d.exists():
            messagebox.showinfo("Zones", f"No folder {d}")
            return
        z = load_zones(d)
        if not z.get("nest") and not z.get("feeders") and not z.get("tray"):
            messagebox.showinfo("Zones", "No zone files found in that folder.")
            return
        self.z.update({k: v for k, v in z.items() if v})
        for k in self.size_vars:
            if k in z:
                self.size_vars[k].set(str(z[k]))
        if self.img is None:
            self.load_frame()
        self.redraw()

    def save(self):
        if not self.z.get("frame_w"):
            messagebox.showerror("Zones", "Load a frame first.")
            return
        missing = [n for n, ok in [("tray outline", self.z.get("tray")), ("nest opening", self.z.get("nest")),
                                   ("left feeder", self.z["feeders"].get("feeder_left")),
                                   ("right feeder", self.z["feeders"].get("feeder_right"))] if not ok]
        d = Path(self.app.zones_dir.get() or DEFAULT_ZONES)
        if missing and not messagebox.askyesno("Zones", "Not set yet: " + ", ".join(missing) + ".\nSave anyway?"):
            return
        if d.resolve() == DEFAULT_ZONES.resolve() and not messagebox.askyesno(
                "Zones", f"This overwrites the zones used for the main video:\n{d}\nContinue?"):
            return
        written = save_zones(d, dict(self.z, **self.sizes()), Path(self.app.video.get()).name)
        self.app.save()
        messagebox.showinfo("Zones", "Saved in " + str(d) + ":\n" + "\n".join(written))

    # -- interaction
    def tool_changed(self):
        self.pending = []
        self.help.config(text=self.HELP[self.tool.get()])
        self.redraw()

    def clear_zone(self):
        k = self.tool.get()
        if k == "tray":
            self.z.pop("tray", None)
        elif k == "slide":
            self.z.pop("nest_slide", None)
        elif k == "nest":
            self.z.pop("nest", None)
        else:
            self.z["feeders"].pop(k, None)
        self.pending = []
        self.redraw()

    def close_outline(self):
        if self.tool.get() == "tray" and len(self.pending) >= 3:
            self.z["tray"] = [[int(x), int(y)] for x, y in self.pending]
            self.pending = []
            self.redraw()

    def click(self, e):
        if self.img is None:
            return
        x, y = (e.x - self.ox) / self.scale, (e.y - self.oy) / self.scale
        if not (0 <= x < self.img.shape[1] and 0 <= y < self.img.shape[0]):
            return
        k = self.tool.get()
        self.pending.append((x, y))
        if k == "nest":
            self.z["nest"] = dict(cx=int(x), cy=int(y))
            self.pending = []
        elif k == "slide" and len(self.pending) == 2:
            (x1, y1), (x2, y2) = self.pending
            self.z["nest_slide"] = [int(min(x1, x2)), int(min(y1, y2)), int(abs(x2 - x1)), int(abs(y2 - y1))]
            self.pending = []
        elif k.startswith("feeder") and len(self.pending) == 2:
            (cx, cy), (ex, ey) = self.pending
            self.z["feeders"][k] = dict(cx=int(cx), cy=int(cy), r=int(np.hypot(ex - cx, ey - cy)))
            self.pending = []
        self.redraw()

    def redraw(self):
        c = self.canvas
        c.delete("all")
        if self.img is None:
            c.create_text(20, 20, anchor="nw", fill="#ddd",
                          text="Choose a video and press 'Load frame'. Then pick a tool on the left and click on the picture.")
            return
        cw, ch = max(c.winfo_width(), 200), max(c.winfo_height(), 200)
        H, W = self.img.shape[:2]
        self.scale = min(cw / W, ch / H)
        self.ox, self.oy = (cw - W * self.scale) / 2, (ch - H * self.scale) / 2
        small = cv2.resize(self.img, (int(W * self.scale), int(H * self.scale)), interpolation=cv2.INTER_AREA)
        self._photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        c.create_image(self.ox, self.oy, anchor="nw", image=self._photo)
        S = lambda x, y: (self.ox + x * self.scale, self.oy + y * self.scale)  # noqa: E731
        circ = lambda x, y, r, **kw: c.create_oval(*S(x - r, y - r), *S(x + r, y + r), **kw)  # noqa: E731
        sz = self.sizes()
        if self.z.get("tray"):
            pts = [v for p in self.z["tray"] for v in S(*p)]
            c.create_polygon(*pts, outline="#33ff66", fill="", width=2)
        if self.z.get("nest_slide"):
            x, y, w, h = self.z["nest_slide"]
            c.create_rectangle(*S(x, y), *S(x + w, y + h), outline="#ff4444", width=2, dash=(4, 2))
            c.create_text(*S(x, y - 12), anchor="sw", text="nest slide (excluded)", fill="#ff6666")
        if self.z.get("nest"):
            n = self.z["nest"]
            circ(n["cx"], n["cy"], sz["nest_birth"], outline="#ff66ff", dash=(3, 3))
            circ(n["cx"], n["cy"], sz["nest_enter"], outline="#ff66ff", width=2)
            circ(n["cx"], n["cy"], sz["nest_clearance"], outline="#aa00aa", fill="", width=1)
            c.create_text(*S(n["cx"], n["cy"] + sz["nest_birth"] + 10), anchor="n", text="nest opening", fill="#ff88ff")
        for k, f in self.z.get("feeders", {}).items():
            circ(f["cx"], f["cy"], f["r"], outline="#ff9900", width=2)
            circ(f["cx"], f["cy"], f["r"] + sz["hide_margin"], outline="#ff9900", dash=(2, 2))
            circ(f["cx"], f["cy"], f["r"] + sz["exit_margin"], outline="#ffcc66", dash=(4, 4))
            c.create_text(*S(f["cx"], f["cy"] + f["r"] + sz["exit_margin"] + 10), anchor="n", text=k.replace("_", " "), fill="#ffbb55")
        for p in self.pending:
            x, y = S(*p)
            c.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#00ffff", outline="")
        if self.tool.get() == "tray" and len(self.pending) > 1:
            c.create_line(*[v for p in self.pending for v in S(*p)], fill="#00ffff", width=2)


# ------------------------------------------------------------------ Track tab

def device_picker(parent, app: "App") -> ttk.Frame:
    """'Run on' choice shared by the Track and Survey tabs."""
    fr = ttk.Frame(parent)
    m = measured_device()
    lab = ttk.Label(fr, text="Run on:")
    lab.pack(side="left")
    ttk.Combobox(fr, textvariable=app.device, values=DEVICE_CHOICES, width=6, state="readonly").pack(side="left", padx=4)
    hint = f"(auto = {m['device'].upper()}, measured)" if m.get("device") else "(auto = GPU if available)"
    ttk.Label(fr, text=hint, foreground="#555").pack(side="left")
    Tooltip(lab, "auto: what the speed test chose for this computer (or the GPU if no test was run). "
                 "cuda = NVIDIA GPU, cpu = processor only, mps = Apple GPU. Re-run the speed test with "
                 ".venv\\Scripts\\python.exe FINAL\\check_speed.py after changing hardware.")
    return fr


# (argument, label, default, help). Defaults must match track_ants.build_parser().
TRACK_SETTINGS = [
    ("Detection", None, None, None),
    ("sweep-seconds", "Full-tray search every (s)", "2.0", "How often the whole tray is searched for ants. Between searches only the areas around known ants are checked. Shorter = new ants found sooner, but slower."),
    ("crop", "Check area around each ant (px)", "128", "Size of the square checked around each known ant between full searches."),
    ("score-threshold", "Keep detections with score >=", "0.3", "Weakest detection that can continue an ant that is already tracked."),
    ("birth-threshold", "New ant needs score >=", "0.5", "Weakest detection that can start a new ID (at the nest or at a cup)."),
    ("track-fps", "Track at (frames/s, 0 = all)", "0", "Speed setting. 0 tracks every frame. A lower number skips frames: 7.5 on 15 fps video uses every 2nd frame (about 2x faster), 5 every 3rd. All other settings are in seconds, so they keep working. Below ~5 fast ants are lost more often."),
    ("batch", "Areas per detector call", BATCH_DEFAULT, "Speed setting. How many image pieces the model checks at once. Default = what the speed test (check_speed.py) found best on this computer. Lower it if you get 'out of memory'."),
    ("Movement", None, None, None),
    ("max-speed", "Fastest ant speed (px/s)", "250", "How far a visible ant may move between frames (divided by the frame rate)."),
    ("merge-radius", "Touching distance (px)", "25", "An ant that vanishes this close to another ant is treated as touching it (both IDs kept)."),
    ("Lost ants", None, None, None),
    ("lost-after-seconds", "Unseen before 'lost' (s)", "0.3", "An ant not detected for this long is lost (or, at a cup / the nest, has gone in)."),
    ("lost-speed", "Lost search grows by (px/s)", "40", "The area in which a lost ant can be found again grows this fast while it is lost."),
    ("lost-max-radius", "Largest lost search (px)", "600", "Upper limit of that search area."),
    ("orphan-seconds", "Unexplained ant: follow for (s)", "1.5", "A detection no ID explains must be followed this long (and move) before it becomes an ID."),
    ("orphan-move", "Unexplained ant: must move (px)", "20", "...and must move at least this far (still specks never qualify)."),
    ("Still and dead", None, None, None),
    ("pin-seconds", "Still before pinned (s)", "3", "An ant that has not moved for this long is held to its spot, so a passing ant cannot take its ID."),
    ("still-radius", "Counts as not moving within (px)", "8", "Movement smaller than this does not count as moving."),
    ("dead-minutes", "Dead after not moving for (min)", "5", "Then the ID is dead for good: never moves, never counted as an ant again."),
    ("static-minutes", "Fixed mark after (min)", "5", "A spot found in nearly every full search for this long is a fixed mark (debris, a mark on a cup rim, a dead ant) and is ignored from then on."),
    ("static-fraction", "...found in this share of searches", "0.8", "How often the spot must be found (0-1) to count as a fixed mark."),
    ("Feeders", None, None, None),
    ("feeder-min-seconds", "Out of sight in cup at least (s)", "3", "An ant must stay hidden in a cup this long to count as going in. Shorter disappearances at the rim are the detector missing an ant on the sleeve or rim."),
    ("Nest", None, None, None),
    ("nest-grace", "Back out of nest within (s)", "10", "An ant that just went into the nest and comes straight out keeps its ID."),
]

class TrackTab(ttk.Frame):
    def __init__(self, master, app: "App"):
        super().__init__(master)
        self.app = app
        f = ttk.Frame(self)
        f.pack(fill="x", padx=10, pady=10)
        PathField(f, "Video", app.video, filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")]).pack(fill="x", pady=2)
        PathField(f, "Model", app.ckpt, filetypes=[("Checkpoints", "*.pt"), ("All", "*.*")]).pack(fill="x", pady=2)
        PathField(f, "Zones folder", app.zones_dir, kind="dir").pack(fill="x", pady=2)
        g = ttk.Frame(f)
        g.pack(fill="x", pady=8)
        self.start = tk.StringVar(value="10:00:00")
        self.dur = tk.StringVar(value="10")
        self.name = tk.StringVar(value="")
        self.overlay = tk.BooleanVar(value=True)
        self.trace = tk.StringVar(value="1080p")
        for i, (lab, w) in enumerate([("Start (h:mm:ss)", ttk.Entry(g, textvariable=self.start, width=10)),
                                      ("Length (minutes)", ttk.Entry(g, textvariable=self.dur, width=8)),
                                      ("Run name (optional)", ttk.Entry(g, textvariable=self.name, width=28))]):
            ttk.Label(g, text=lab).grid(row=0, column=2 * i, sticky="w", padx=(0, 4))
            w.grid(row=0, column=2 * i + 1, sticky="w", padx=(0, 16))
        o = ttk.Frame(f)
        o.pack(fill="x")
        ttk.Checkbutton(o, text="Quick check video while running (overlay.mp4, 1080p; costs some speed)", variable=self.overlay).pack(side="left")
        ttk.Label(o, text="   Trace video afterwards:").pack(side="left")
        ttk.Combobox(o, textvariable=self.trace, values=["none", "1080p", "4K"], width=7, state="readonly").pack(side="left")
        device_picker(o, app).pack(side="left", padx=(16, 0))
        self.settings_vars: dict[str, tk.StringVar] = {}
        saved = app.settings.get("track_settings", {})
        sf = ttk.LabelFrame(f, text="Tracking settings (hover a name for an explanation; times in seconds, distances in full-frame pixels)")
        sf.pack(fill="x", pady=(10, 0))
        col, row = 0, 0
        for arg, label, default, helptext in TRACK_SETTINGS:
            if label is None:
                if row:
                    col, row = col + 1, 0
                ttk.Label(sf, text=arg, font=("TkDefaultFont", 10, "bold")).grid(row=0, column=2 * col, columnspan=2, sticky="w", padx=8, pady=(4, 2))
                row = 1
                continue
            v = tk.StringVar(value=str(saved.get(arg, default)))
            self.settings_vars[arg] = v
            lab = ttk.Label(sf, text=label)
            lab.grid(row=row, column=2 * col, sticky="w", padx=(8, 4), pady=1)
            ttk.Entry(sf, textvariable=v, width=7).grid(row=row, column=2 * col + 1, sticky="w", padx=(0, 14), pady=1)
            Tooltip(lab, helptext)
            row += 1
        ttk.Button(sf, text="Reset to defaults", command=self.reset_settings).grid(row=7, column=0, columnspan=2, sticky="w", padx=8, pady=6)
        b = ttk.Frame(f)
        b.pack(fill="x", pady=10)
        ttk.Button(b, text="Run tracking", command=self.run).pack(side="left")
        ttk.Label(b, text="Speed: the log shows 'x real time'. Lab PC (GTX 1050): about 0.27x on 10 fps video with every frame tracked. "
                          "Faster: 'Track at' 7.5 or 5, overlay off.", foreground="#555").pack(side="left", padx=12)
        ttk.Label(self, justify="left", foreground="#333", text=(
            "What the tracker does: follows every ant in every frame (any frame rate). Ants never vanish in the open field; new ants only appear\n"
            "at the nest opening; touching ants keep both IDs; ants going into a cup are queued and the first one out gets the\n"
            "first ID in; anything that has not moved for 5 minutes is marked dead for good.\n"
            "Outputs (in FINAL/runs/<run name>): counts.csv (ants on tray, in each feeder, dead - every second), feeder_events.csv,\n"
            "feeder_visits.csv, tracks.csv, tracks_summary.csv, overlay.mp4 / trace video.")).pack(anchor="w", padx=10)

    def reset_settings(self):
        for arg, label, default, _ in TRACK_SETTINGS:
            if label is not None:
                self.settings_vars[arg].set(default)

    def setting_args(self) -> list[str] | None:
        out = []
        for arg, label, default, _ in TRACK_SETTINGS:
            if label is None:
                continue
            txt = self.settings_vars[arg].get().strip()
            try:
                float(txt)
            except ValueError:
                messagebox.showerror("Track", f"'{label}' must be a number (got '{txt}').")
                return None
            out += [f"--{arg}", txt]
        self.app.settings["track_settings"] = {k: v.get() for k, v in self.settings_vars.items()}
        return out

    def run(self):
        a = self.app
        v, ck, zd = a.video.get(), a.ckpt.get(), Path(a.zones_dir.get() or DEFAULT_ZONES)
        if not Path(v).exists():
            return messagebox.showerror("Track", "Video not found.")
        extra = self.setting_args()
        if extra is None:
            return
        if not Path(ck).exists():
            return messagebox.showerror("Track", "Model checkpoint not found.")
        miss = zones_complete(zd)
        if miss:
            return messagebox.showerror("Track", f"Zones folder is missing: {', '.join(miss)}.\nSet and save zones in the Zones tab.")
        try:
            s, d = parse_time(self.start.get()), float(self.dur.get()) * 60
        except ValueError as e:
            return messagebox.showerror("Track", str(e))
        name = self.name.get().strip() or f"track_{Path(v).stem[:20].replace(' ', '_')}_{fmt_time(s).replace(':', '-')}_{int(d // 60)}min"
        out = RUNS / name
        if out.exists() and any(out.iterdir()) and not messagebox.askyesno("Track", f"{out.name} already exists. Overwrite?"):
            return
        a.save()
        steps = [["track-ants", "--", "--video", rel(v), "--checkpoint", rel(ck), "--output", rel(out),
                  "--mask", rel(zd / "tray_mask.png"), "--zones", rel(zd / "nest_clearance.json"),
                  "--areas", rel(zd / "areas.json"), "--tracking-zones", rel(zd / "tracking_zones.json"),
                  "--start-seconds", str(s), "--duration-seconds", str(d), "--device", a.device.get(),
                  *(["--overlay"] if self.overlay.get() else []), *extra]]
        if self.trace.get() != "none":
            steps.append(["trace-video", "--", "--run", rel(out), "--scale", "1.0" if self.trace.get() == "4K" else "0.5"])
        a.runner.run(steps, f"Tracking {fmt_time(s)} + {d / 60:g} min -> {name}",
                     on_done=lambda: a.show_results(out))


# ------------------------------------------------------------------ Survey tab

class SurveyTab(ttk.Frame):
    def __init__(self, master, app: "App"):
        super().__init__(master)
        self.app = app
        f = ttk.Frame(self)
        f.pack(fill="x", padx=10, pady=10)
        PathField(f, "Video", app.video, filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")]).pack(fill="x", pady=2)
        PathField(f, "Model", app.ckpt, filetypes=[("Checkpoints", "*.pt"), ("All", "*.*")]).pack(fill="x", pady=2)
        PathField(f, "Zones folder", app.zones_dir, kind="dir").pack(fill="x", pady=2)
        g = ttk.Frame(f)
        g.pack(fill="x", pady=8)
        self.h0, self.h1, self.step, self.name = tk.StringVar(value="0"), tk.StringVar(value=""), tk.StringVar(value="5"), tk.StringVar(value="")
        for i, (lab, var, w) in enumerate([("From hour", self.h0, 6), ("To hour (blank = end)", self.h1, 6),
                                           ("One frame every (s)", self.step, 6), ("Run name (optional)", self.name, 24)]):
            ttk.Label(g, text=lab).grid(row=0, column=2 * i, sticky="w", padx=(0, 4))
            ttk.Entry(g, textvariable=var, width=w).grid(row=0, column=2 * i + 1, sticky="w", padx=(0, 14))
        b = ttk.Frame(f)
        b.pack(fill="x", pady=10)
        ttk.Button(b, text="Run survey + analysis", command=self.run).pack(side="left")
        device_picker(b, app).pack(side="left", padx=(12, 0))
        ttk.Label(b, foreground="#555", text="About 0.6 s (Mac) to 1.9 s (lab PC) per sampled frame; every 5 s keeps up with recording. "
                                              "Finished hours are kept, so a stopped survey can be restarted.").pack(side="left", padx=12)
        ttk.Label(self, justify="left", foreground="#333", text=(
            "The survey counts ants (no IDs) on frames spaced through the whole recording, then the analysis makes:\n"
            "hourly counts (moving / stationary / permanent), feeder occupancy, heatmaps of where ants go and rest,\n"
            "and a list of objects that never move. The analysis needs: pip install pandas matplotlib")).pack(anchor="w", padx=10)

    def run(self):
        a = self.app
        v, ck, zd = a.video.get(), a.ckpt.get(), Path(a.zones_dir.get() or DEFAULT_ZONES)
        if not Path(v).exists() or not Path(ck).exists():
            return messagebox.showerror("Survey", "Check the video and model paths.")
        miss = [m for m in zones_complete(zd) if m != "tracking_zones.json"]
        if miss:
            return messagebox.showerror("Survey", f"Zones folder is missing: {', '.join(miss)}.")
        try:
            h0 = int(float(self.h0.get() or 0))
            h1 = int(float(self.h1.get())) if self.h1.get().strip() else None
            step = float(self.step.get())
        except ValueError:
            return messagebox.showerror("Survey", "Hours and step must be numbers.")
        name = self.name.get().strip() or f"survey_{Path(v).stem[:20].replace(' ', '_')}_{step:g}s"
        out = RUNS / name
        a.save()
        steps = [["survey-video", "--", "--video", rel(v), "--checkpoint", rel(ck), "--output", rel(out),
                  "--mask", rel(zd / "tray_mask.png"), "--zones", rel(zd / "nest_clearance.json"), "--areas", rel(zd / "areas.json"),
                  "--step-seconds", str(step), "--device", a.device.get(), "--start-hour", str(h0), *(["--end-hour", str(h1)] if h1 is not None else [])],
                 ["analyze-survey", "--", "--survey", rel(out)]]
        a.runner.run(steps, f"Survey hours {h0}-{h1 if h1 is not None else 'end'}, every {step:g}s -> {name}",
                     on_done=lambda: a.show_results(out))


# ------------------------------------------------------------------ Results tab

class ResultsTab(ttk.Frame):
    def __init__(self, master, app: "App"):
        super().__init__(master)
        self.app = app
        left = ttk.Frame(self)
        left.pack(side="left", fill="y", padx=8, pady=8)
        ttk.Label(left, text="Runs", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        self.lb = tk.Listbox(left, width=34, height=28, exportselection=False)
        self.lb.pack(fill="y", expand=True)
        self.lb.bind("<<ListboxSelect>>", lambda e: self.select())
        ttk.Button(left, text="Refresh", command=self.refresh).pack(fill="x", pady=2)
        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        self.title = ttk.Label(right, text="", font=("TkDefaultFont", 12, "bold"))
        self.title.pack(anchor="w")
        self.summary = tk.Text(right, height=7, wrap="word", relief="flat", background="#f6f6f6", font=("TkDefaultFont", 11))
        self.summary.pack(fill="x", pady=4)
        self.btns = ttk.Frame(right)
        self.btns.pack(fill="x", pady=4)
        self.chart = tk.Canvas(right, bg="white", height=330, highlightthickness=1, highlightbackground="#ccc")
        self.chart.pack(fill="both", expand=True, pady=4)
        self.chart.bind("<Configure>", lambda e: self.draw_chart())
        self.runs: list[Path] = []
        self.current: Path | None = None
        self.series = None
        self.image_paths: list[Path] = []
        self.refresh()

    def refresh(self):
        self.runs = sorted([p for p in RUNS.glob("*") if p.is_dir() and ((p / "tracks.csv").exists() or (p / "survey_manifest.json").exists())],
                           key=lambda p: p.stat().st_mtime, reverse=True)
        self.lb.delete(0, "end")
        for p in self.runs:
            kind = "track " if (p / "tracks.csv").exists() else "survey"
            self.lb.insert("end", f"{kind}  {p.name}")

    def show(self, path: Path):
        self.refresh()
        for i, p in enumerate(self.runs):
            if p.resolve() == Path(path).resolve():
                self.lb.selection_clear(0, "end")
                self.lb.selection_set(i)
                self.lb.see(i)
                self.select()

    def button(self, text, cmd):
        ttk.Button(self.btns, text=text, command=cmd).pack(side="left", padx=(0, 6))

    def select(self):
        sel = self.lb.curselection()
        if not sel:
            return
        p = self.runs[sel[0]]
        self.current = p
        for w in self.btns.winfo_children():
            w.destroy()
        self.summary.delete("1.0", "end")
        self.series = None
        self.image_paths = []
        self.title.config(text=p.name)
        self.button("Open folder", lambda: open_path(p))
        if (p / "tracks.csv").exists():
            self.show_track(p)
        else:
            self.show_survey(p)
        self.draw_chart()

    # -- tracking run
    def show_track(self, p: Path):
        man = json.loads((p / "track_manifest.json").read_text()) if (p / "track_manifest.json").exists() else {}
        counts = read_csv(p / "counts.csv")
        ev = read_csv(p / "feeder_events.csv")
        visits = [v for v in read_csv(p / "feeder_visits.csv") if v.get("exit_sec")]
        lines = []
        if man:
            lines.append(f"Video {Path(man.get('video', '')).name}, from {fmt_time(float(man.get('start_seconds', 0)))} "
                         f"for {float(man.get('duration_seconds', 0)) / 60:g} min.")
        if counts:
            tray = [int(r["ants_on_tray"]) for r in counts]
            dead = [int(r.get("dead", 0)) for r in counts]
            lines.append(f"Ants on the tray: mean {np.mean(tray):.1f}, min {min(tray)}, max {max(tray)}.   Dead at the end: {dead[-1]}.")
            feeders = [k[3:] for k in counts[0] if k.startswith("in_feeder")]
            for fid in feeders:
                n_in = sum(e["feeder"] == fid and e["event"] == "in" for e in ev)
                n_out = sum(e["feeder"] == fid and e["event"] == "out" for e in ev)
                durs = [float(v["duration_sec"]) for v in visits if v["feeder"] == fid]
                med = f", median visit {np.median(durs):.0f} s" if durs else ""
                lines.append(f"{fid.replace('_', ' ').capitalize()}: {n_in} in, {n_out} out, {len(durs)} complete visits{med}.")
            t = [float(r["t_sec"]) for r in counts]
            self.series = dict(t=t, lines=[("ants on tray", tray, "#2a6fdb")] +
                               [(f"inside {fid.replace('feeder_', '')} cup", [int(r[f'in_{fid}']) for r in counts], c)
                                for fid, c in zip(feeders, ["#d1495b", "#00798c"])] + [("dead", dead, "#555555")])
        else:
            lines.append("No counts.csv yet (run may still be going or was stopped).")
        self.summary.insert("1.0", "\n".join(lines))
        for name, label in [("trace_4k.mp4", "Play trace video (4K)"), ("trace.mp4", "Play trace video"), ("overlay.mp4", "Play quick video")]:
            if (p / name).exists():
                self.button(label, lambda n=name: open_path(p / n))
        self.button("Make trace video (1080p)", lambda: self.app.runner.run(
            [["trace-video", "--", "--run", rel(p), "--scale", "0.5"]], f"Trace video for {p.name}", on_done=lambda: self.show(p)))
        self.button("Make trace video (4K)", lambda: self.app.runner.run(
            [["trace-video", "--", "--run", rel(p), "--scale", "1.0"]], f"4K trace video for {p.name}", on_done=lambda: self.show(p)))
        if (p / "feeder_visits.csv").exists():
            self.button("Feeder visits table", lambda: open_path(p / "feeder_visits.csv"))

    # -- survey run
    def show_survey(self, p: Path):
        an = p / "analysis"
        if (an / "summary.json").exists():
            s = json.loads((an / "summary.json").read_text())
            mb = s.get("mean_by_state", {})
            self.summary.insert("1.0", (
                f"{len(s.get('hours', []))} hours, {s.get('sampled_frames', 0):,} frames (every {s.get('step_seconds')} s), "
                f"{s.get('detections', 0):,} detections.\n"
                f"Ants per frame: {mb.get('moving', 0)} moving, {mb.get('stationary', 0)} stationary, {mb.get('permanent', 0)} permanent.\n"
                f"Feeder episodes: {s.get('feeder_episodes')}.  Objects that never move: {s.get('permanent_objects')}."))
            self.image_paths = [an / n for n in ("hourly_counts.png", "feeder_occupancy.png", "heatmap_moving.png", "heatmap_stationary.png")
                                if (an / n).exists()]
            for ip in self.image_paths:
                self.button(ip.stem.replace("_", " "), lambda ip=ip: self.show_image(ip))
            if self.image_paths:
                self.after(50, lambda: self.show_image(self.image_paths[0]))
        else:
            done = len(list(p.glob("hour_*.csv")))
            self.summary.insert("1.0", f"{done} hours surveyed so far. No analysis yet: run 'Analyse again' (needs pandas + matplotlib).")
        self.button("Analyse again", lambda: self.app.runner.run([["analyze-survey", "--", "--survey", rel(p)]],
                                                                 f"Analysis of {p.name}", on_done=lambda: self.show(p)))

    def show_image(self, ip: Path):
        self.series = None
        c = self.chart
        c.delete("all")
        w, h = max(c.winfo_width(), 300), max(c.winfo_height(), 200)
        im = Image.open(ip)
        im.thumbnail((w, h))
        self._img = ImageTk.PhotoImage(im)
        c.create_image(w // 2, h // 2, image=self._img)

    def draw_chart(self):
        c = self.chart
        if self.series is None:
            if not self.image_paths:
                c.delete("all")
            return
        c.delete("all")
        w, h = max(c.winfo_width(), 300), max(c.winfo_height(), 200)
        L, R, T, B = 50, 140, 20, 35
        t = self.series["t"]
        if len(t) < 2:
            return
        ymax = max(max(v) for _, v, _ in self.series["lines"]) or 1
        ymax = int(np.ceil(ymax * 1.1))
        X = lambda x: L + (x - t[0]) / (t[-1] - t[0]) * (w - L - R)  # noqa: E731
        Y = lambda y: h - B - y / ymax * (h - T - B)  # noqa: E731
        c.create_line(L, h - B, w - R, h - B, fill="#999")
        c.create_line(L, T, L, h - B, fill="#999")
        for k in range(0, ymax + 1, max(1, ymax // 6)):
            c.create_line(L - 4, Y(k), w - R, Y(k), fill="#eee")
            c.create_text(L - 8, Y(k), text=str(k), anchor="e", fill="#555")
        span = t[-1] - t[0]
        step = next(s for s in (10, 30, 60, 120, 300, 600, 1800, 3600, 7200, 14400) if span / s <= 8)
        x0 = t[0] - t[0] % step + step
        while x0 <= t[-1]:
            c.create_text(X(x0), h - B + 12, text=fmt_time(x0), fill="#555")
            x0 += step
        for i, (label, ys, col) in enumerate(self.series["lines"]):
            pts = [v for x, y in zip(t, ys) for v in (X(x), Y(y))]
            c.create_line(*pts, fill=col, width=2)
            c.create_line(w - R + 10, T + 18 * i + 6, w - R + 30, T + 18 * i + 6, fill=col, width=3)
            c.create_text(w - R + 36, T + 18 * i + 6, text=label, anchor="w", fill="#333")


# ------------------------------------------------------------------ main window

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ant tracking - Taka Lab")
        self.geometry("1400x900")
        s = load_settings()
        self.settings = s
        guess_video = next(iter(sorted(REPO.parent.glob("*.mp4"))), None)
        self.video = tk.StringVar(value=s.get("video") or (str(guess_video) if guess_video else ""))
        self.ckpt = tk.StringVar(value=s.get("ckpt") or str(DEFAULT_CKPT))
        self.zones_dir = tk.StringVar(value=s.get("zones_dir") or str(DEFAULT_ZONES))
        self.device = tk.StringVar(value=s.get("device") if s.get("device") in DEVICE_CHOICES else "auto")
        self.runner = Runner(self)
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True)
        self.zones = ZonesTab(self.nb, self)
        self.track = TrackTab(self.nb, self)
        self.survey = SurveyTab(self.nb, self)
        self.results = ResultsTab(self.nb, self)
        for tab, name in [(self.zones, "1  Zones"), (self.track, "2  Track ants"), (self.survey, "3  Whole-video survey"), (self.results, "4  Results")]:
            self.nb.add(tab, text=name)
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", side="bottom")
        bar = ttk.Frame(bottom)
        bar.pack(fill="x", padx=8, pady=4)
        self.status = ttk.Label(bar, text="Ready", width=60)
        self.status.pack(side="left")
        self.pb = ttk.Progressbar(bar, length=400, mode="determinate")
        self.pb.pack(side="left", padx=8)
        ttk.Button(bar, text="Stop", command=self.runner.stop).pack(side="left")
        ttk.Button(bar, text="Show/hide log", command=self.toggle_log).pack(side="right")
        self.logbox = tk.Text(bottom, height=10, wrap="none", font=("Courier", 10), background="#111", foreground="#ddd")
        self.log_visible = True
        self.logbox.pack(fill="x", padx=8, pady=(0, 6))
        self.protocol("WM_DELETE_WINDOW", self.quit_app)
        if Path(self.zones_dir.get()).exists() and self.video.get() and Path(self.video.get()).exists():
            self.after(300, self.zones.load_existing)

    def save(self):
        self.settings.update(video=self.video.get(), ckpt=self.ckpt.get(), zones_dir=self.zones_dir.get(), device=self.device.get())
        if hasattr(self, "track"):
            self.settings["track_settings"] = {k: v.get() for k, v in self.track.settings_vars.items()}
        save_settings(self.settings)

    def log(self, text: str):
        self.logbox.insert("end", text)
        self.logbox.see("end")

    def log_clear(self):
        self.logbox.delete("1.0", "end")
        self.pb["value"] = 0

    def toggle_log(self):
        if self.log_visible:
            self.logbox.pack_forget()
        else:
            self.logbox.pack(fill="x", padx=8, pady=(0, 6))
        self.log_visible = not self.log_visible

    def set_status(self, text: str, running: bool):
        self.status.config(text=("Running: " if running and not text.startswith("Running") else "") + text)

    def set_progress(self, done: int, total: int):
        self.pb["maximum"] = max(total, 1)
        self.pb["value"] = done

    def show_results(self, path: Path):
        self.nb.select(self.results)
        self.results.show(path)

    def quit_app(self):
        if self.runner.busy and not messagebox.askyesno("Quit", "A job is running. Stop it and quit?"):
            return
        self.runner.stop()
        self.save()
        self.destroy()


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
