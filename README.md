# Taka Lab ant tracker

Detects and tracks ants on the foraging tray in 4K video, counts ants on the tray and in the feeders,
and records feeder visits. Built on Aloha's detector pipeline (`FINAL/`, see `FINAL/README.md`), extended by
Can (Taka Lab) in 2026.

## What it does
- **Detector:** Faster R-CNN (MobileNetV3-FPN) retrained on this setup (`ours_v2_best.pt`): 97-98 % of ants
  found, F1 0.96 on a held-out test set.
- **Ant tracking** (`track-ants`): every frame, conserved counts - ants never vanish on the open tray, new ants
  only appear at the nest opening, touching ants keep both IDs, ants going into a cup are queued
  (first in, first out), objects that never move are marked dead.
- **Whole-video survey** (`survey-video` + `analyze-survey`): one frame every few seconds through a long
  recording; hourly counts, feeder occupancy, heatmaps.
- **App** (`FINAL/pipeline.py app`): a window to set zones, run tracking or the survey, and view results.
  See `FINAL/app/README.md`.

## Set up on Windows (lab computer)
No administrator rights are needed; everything installs for the current user.

1. Install Git (`winget install -e --id Git.Git --scope user`), close and reopen PowerShell, then clone:
   ```
   cd $HOME
   git clone https://github.com/canafaraci/Taka_Ant_Detection.git
   cd Taka_Ant_Detection
   ```
2. Run the setup. It installs Python 3.12 and ffmpeg, PyTorch (a CUDA build matched to the NVIDIA card,
   with fallbacks for older cards such as the GTX 10xx series, or the CPU build), the other packages,
   downloads the model, and finally runs a speed test:
   ```
   powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
   ```
   The speed test (`FINAL\check_speed.py`) compares GPU and CPU and writes the faster one, with a batch size
   that fits in GPU memory, to `FINAL\device_settings.json`. Survey and tracking use it automatically.
   Re-run it after changing hardware or drivers: `.venv\Scripts\python.exe FINAL\check_speed.py`.
3. Double-click **Ant Tracker.bat**.

To update later: `git pull` in the folder (re-run the setup only if it says packages changed).

## Set up on macOS
Python 3.12 with Tk (`brew install python@3.12 python-tk@3.12 ffmpeg`), then
`python3.12 -m venv .venv && source .venv/bin/activate && pip install torch torchvision -r FINAL/requirements.txt pandas matplotlib`,
download `ours_v2_best.pt` from the latest release into `FINAL/checkpoints/`, and double-click `Ant Tracker.command`.

## Command line
```
python FINAL/pipeline.py track-ants -- --video <video.mp4> --checkpoint FINAL/checkpoints/ours_v2_best.pt \
  --output FINAL/runs/<name> --mask FINAL/runs/zones/<setup>/tray_mask.png \
  --zones FINAL/runs/zones/<setup>/nest_clearance.json --areas FINAL/runs/zones/<setup>/areas.json \
  --tracking-zones FINAL/runs/zones/<setup>/tracking_zones.json --start-seconds 0 --duration-seconds 600 --overlay
python FINAL/pipeline.py trace-video -- --run FINAL/runs/<name> --scale 0.5
python FINAL/pipeline.py survey-video -- --help
```

## Zones
`FINAL/runs/zones/` holds the zones of the 51.6 h recording; `FINAL/runs/zones/pilot/` those of the pilot
experiment (Nov 2026). Make a new folder per setup in the app's Zones tab whenever the camera, tray, nest
or cups move.

## Outputs of a tracking run (`FINAL/runs/<name>/`)
`counts.csv` (ants on the tray, inside each feeder, dead - every second), `feeder_events.csv`, `feeder_visits.csv`,
`tracks.csv` (every ant, every frame, with detection confidence), `tracks_summary.csv`, `events.csv`,
`overlay.mp4`, and a trace video on request. Videos and run outputs are not stored in git.
