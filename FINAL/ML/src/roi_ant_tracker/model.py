from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


ARCHITECTURES = ("mobilenet_v3_fpn", "resnet50_fpn_v2")

# Per-computer choice written by FINAL/check_speed.py (not in git).
SETTINGS_PATH = Path(__file__).resolve().parents[3] / "device_settings.json"


def local_settings() -> dict[str, Any]:
    try:
        return json.loads(SETTINGS_PATH.read_text())
    except Exception:
        return {}


def default_batch(fallback: int = 8) -> int:
    """Batch size measured for this computer by check_speed.py, else the fallback."""
    try:
        return max(1, int(local_settings().get("batch", fallback)))
    except Exception:
        return fallback


def build_detector(
    architecture: str = "mobilenet_v3_fpn",
    num_classes: int = 2,
    min_size: int = 768,
    max_size: int = 1280,
    pretrained_backbone: bool = False,
    pretrained_detector: bool = False,
    box_detections_per_img: int = 300,
) -> FasterRCNN:
    """Build a small-object Faster R-CNN detector.

    MobileNet is the default because the ROI workflow should keep inference
    cheap. ResNet50 remains available for difficult footage.
    """

    if pretrained_backbone and pretrained_detector:
        raise ValueError("Choose either pretrained_detector or pretrained_backbone, not both.")
    common: dict[str, Any] = {
        "min_size": min_size,
        "max_size": max_size,
        "box_detections_per_img": box_detections_per_img,
    }
    if architecture == "mobilenet_v3_fpn":
        model = fasterrcnn_mobilenet_v3_large_fpn(
            **common,
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained_detector else None,
            weights_backbone=MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None,
            num_classes=None if pretrained_detector else num_classes,
        )
        # MobileNet's RPN head expects 15 anchors per location.
        sizes = ((4, 8, 16, 32, 64),) * 3
        ratios = ((0.5, 1.0, 2.0),) * 3
    elif architecture == "resnet50_fpn_v2":
        model = fasterrcnn_resnet50_fpn_v2(
            **common,
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained_detector else None,
            weights_backbone=ResNet50_Weights.DEFAULT if pretrained_backbone else None,
            num_classes=None if pretrained_detector else num_classes,
        )
        # FPN levels have strides 4-64; these anchors retain tiny ant proposals.
        sizes = ((4,), (8,), (16,), (32,), (64,))
        ratios = ((0.5, 1.0, 2.0),) * len(sizes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    model.rpn.anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=ratios)
    if pretrained_detector:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def choose_device(requested: str = "auto") -> torch.device:
    """auto picks CUDA, then Apple-silicon GPU (MPS), then CPU."""
    if requested == "auto":
        preferred = local_settings().get("device")
        if preferred == "cpu":
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps" and not mps_available():
        raise RuntimeError("MPS (Apple GPU) was requested but is not available.")
    return device


def load_detector(checkpoint_path: str, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config", {})
    model = build_detector(**model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, model_config

