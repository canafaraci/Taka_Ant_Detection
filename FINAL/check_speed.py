"""Measure how fast the ant detector runs on this computer and pick GPU or CPU.

Run:  .venv\\Scripts\\python.exe FINAL\\check_speed.py      (Windows)
      .venv/bin/python FINAL/check_speed.py                 (macOS)

Tries the NVIDIA GPU (if PyTorch can use it) at batch sizes 8, 4, 2 and 1, then the CPU,
and saves the fastest working choice to FINAL/device_settings.json. The app, survey and
tracking read that file when the device is left on "auto". Delete the file to go back
to the default (GPU if available).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ML" / "src"))

import torch  # noqa: E402

from roi_ant_tracker.model import SETTINGS_PATH, load_detector, mps_available  # noqa: E402

CKPT = ROOT / "checkpoints" / "ours_v2_best.pt"
TILE = 320  # tile size used by survey and tracking (the model resizes it to 768)


def fake_tiles(n: int, device: torch.device) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    out = []
    for _ in range(n):
        img = torch.rand(3, TILE, TILE, generator=g) * 0.15 + 0.6
        for _ in range(6):  # a few dark ant-sized blobs
            x, y = torch.randint(10, TILE - 30, (2,), generator=g).tolist()
            img[:, y:y + 8, x:x + 20] = 0.15
        out.append(img.to(device))
    return out


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def time_batch(model, device, batch: int, reps: int) -> float:
    tiles = fake_tiles(batch, device)
    with torch.inference_mode():
        for _ in range(2):
            model(tiles)
        sync(device)
        t0 = time.perf_counter()
        for _ in range(reps):
            model(tiles)
        sync(device)
    return (time.perf_counter() - t0) / (reps * batch)


def is_oom(e: Exception) -> bool:
    return "out of memory" in str(e).lower() or e.__class__.__name__ == "OutOfMemoryError"


def main() -> int:
    if not CKPT.exists():
        print(f"Model not found: {CKPT}")
        return 1
    print(f"torch {torch.__version__}, CPU threads {torch.get_num_threads()}")
    results: dict[str, dict] = {}

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"GPU: {name} ({total:.1f} GB)")
        try:
            model, _ = load_detector(str(CKPT), dev)
            for b in (8, 4, 2, 1):
                try:
                    torch.cuda.reset_peak_memory_stats()
                    s = time_batch(model, dev, b, reps=5)
                    peak = torch.cuda.max_memory_allocated() / 2**30
                    print(f"  GPU batch {b}: {s * 1000:6.1f} ms per tile (peak memory {peak:.2f} GB)")
                    if "cuda" not in results or s < results["cuda"]["sec_per_tile"]:
                        results["cuda"] = dict(sec_per_tile=s, batch=b, peak_gb=round(peak, 2))
                except Exception as e:  # noqa: BLE001
                    if is_oom(e):
                        print(f"  GPU batch {b}: out of GPU memory")
                        torch.cuda.empty_cache()
                        continue
                    raise
            del model
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"  GPU failed: {str(e).splitlines()[0]}")
    elif mps_available():
        dev = torch.device("mps")
        model, _ = load_detector(str(CKPT), dev)
        s = time_batch(model, dev, 8, reps=5)
        print(f"  Apple GPU batch 8: {s * 1000:6.1f} ms per tile")
        results["mps"] = dict(sec_per_tile=s, batch=8)
        del model
    else:
        print("GPU: none usable by PyTorch")

    dev = torch.device("cpu")
    model, _ = load_detector(str(CKPT), dev)
    s = time_batch(model, dev, 4, reps=3)
    print(f"  CPU batch 4: {s * 1000:6.1f} ms per tile")
    results["cpu"] = dict(sec_per_tile=s, batch=4)

    best = min(results, key=lambda k: results[k]["sec_per_tile"])
    choice = dict(device=best, batch=results[best]["batch"], measured=results,
                  torch=torch.__version__, measured_at=time.strftime("%Y-%m-%d %H:%M"))
    SETTINGS_PATH.write_text(json.dumps(choice, indent=2))
    other = [k for k in results if k != best]
    ratio = f" ({results[other[0]]['sec_per_tile'] / results[best]['sec_per_tile']:.1f}x faster than {other[0].upper()})" if other else ""
    print(f"\nChosen: {best.upper()}, batch {choice['batch']}{ratio}")
    print(f"  {1 / results[best]['sec_per_tile']:.1f} tiles per second")
    print(f"Saved to {SETTINGS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
