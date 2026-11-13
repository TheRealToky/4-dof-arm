"""Locates the MediaPipe model files, downloading them once on first use.

Files come from Google's official MediaPipe model bucket and are stored in
the git-ignored models/ folder.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_BASE = "https://storage.googleapis.com/mediapipe-models"

MODELS = {
    "pose": ("pose_landmarker_full.task",
             f"{_BASE}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"),
    "hand": ("hand_landmarker.task",
             f"{_BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"),
}


def model_path(name: str) -> Path:
    filename, url = MODELS[name]
    path = MODEL_DIR / filename
    if not path.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {filename} ...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path
