# Ant tracking app

One window to set up zones, track ants, run the whole-video survey and look at the results.

## Start it

From the repository folder (`Taka_Ant_Detection`), with the Python environment active:

```
source ~/TakaLab/taka-env/bin/activate        # or your own environment
python FINAL/pipeline.py app
```

On a Mac you can also double-click `Ant Tracker.command` in the repository folder.

## The four tabs

1. **Zones** - for each new recording. Load a frame, then click:
   - *Tray outline*: points around the tray floor, then *Close outline*.
   - *Nest slide (excluded)*: two corners of the nest slide (no detection there).
   - *Nest opening*: the notch where ants go in and out.
   - *Left / right feeder cup*: centre, then a point on the rim.
   Save into a new folder per recording (e.g. `FINAL/runs/zones/<recording>`). The circles show the
   margins; the sizes on the left can be changed.
2. **Track ants** - choose start time and length; follows every ant in every frame (10 or 15 fps both work).
   Results: `counts.csv` (ants on the tray, inside each feeder, dead - every second), `feeder_events.csv`
   (every in/out), `feeder_visits.csv` (first in, first out), `tracks.csv`, plus a trace video if chosen.
   Speed: about 6-7 frames per second on an Apple-silicon Mac (10 min of 15 fps video take ~23 min).
3. **Whole-video survey** - counts ants (no IDs) every few seconds through the whole recording, then makes
   hourly counts, feeder occupancy and heatmaps. Every 5 s over 52 h took about 9 h. Stopped surveys
   continue where they left off.
4. **Results** - pick a run: summary numbers, a chart of the counts, buttons to play or make the trace video.

Only one job runs at a time; the log at the bottom shows progress, and *Stop* ends the job.

## Setting up on another computer

1. Python 3.12 with Tk (Mac: `brew install python@3.12 python-tk@3.12 ffmpeg`).
2. `python3.12 -m venv taka-env && source taka-env/bin/activate`
3. `pip install -r FINAL/requirements.txt pandas matplotlib`
4. The model file `FINAL/checkpoints/ours_v2_best.pt` (not in git; copy it from Can).
5. The recordings can live anywhere; choose them with *Browse...*.

An Apple-silicon Mac uses its GPU automatically; an NVIDIA GPU is used if present; otherwise it runs on the
CPU (much slower: tracking roughly 10x slower).
