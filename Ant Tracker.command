#!/bin/bash
# Double-click to open the ant tracking app (macOS).
cd "$(dirname "$0")"
for env in "$HOME/TakaLab/taka-env" "../taka-env" "./taka-env"; do
  if [ -f "$env/bin/activate" ]; then source "$env/bin/activate"; break; fi
done
python FINAL/pipeline.py app
