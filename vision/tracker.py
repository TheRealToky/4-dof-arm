"""MediaPipe wrapper: finds one arm (shoulder, elbow, wrist) and its hand.

Pose gives the elbow, which hand tracking alone cannot see; Hands gives the
knuckles, which pose tracking places poorly. Both run on every frame.

Axes for every 3D value here follow the camera: x right, y down, z away
from the camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from .models import model_path

# Pose landmark indices, by the person's own left/right.
POSE_IDX = {
    "left":  {"shoulder": 11, "elbow": 13, "wrist": 15},
    "right": {"shoulder": 12, "elbow": 14, "wrist": 16},
}
# Hand landmark indices.
HAND_IDX = {"hand_wrist": 0, "index_mcp": 5, "index_tip": 8, "middle_mcp": 9, "pinky_mcp": 17}

MIN_VISIBILITY = 0.5
# A detected hand belongs to the arm only if its wrist is this close to the
# pose wrist, as a fraction of the forearm's length in the image.
MAX_HAND_WRIST_GAP = 0.6


@dataclass
class TrackResult:
    arm_ok: bool = False            # elbow and wrist found
    hand_ok: bool = False           # hand found on that arm
    px: dict = field(default_factory=dict)      # name -> (x, y) in pixels
    forearm: np.ndarray | None = None   # elbow -> wrist, MediaPipe's own 3D estimate (m)
    hand: np.ndarray | None = None      # wrist -> middle knuckle, MediaPipe's own 3D estimate (m)
    hand_px_all: np.ndarray | None = None       # all 21 hand landmarks, pixels (21, 2)
    hand_world_all: np.ndarray | None = None    # all 21, MediaPipe's metric hand model (21, 3)


class ArmTracker:
    def __init__(self, arm: str = "right"):
        if arm not in POSE_IDX:
            raise ValueError("arm must be 'left' or 'right'")
        self.arm = arm
        mode = vision.RunningMode.VIDEO
        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=model_path("pose").read_bytes()),
                running_mode=mode, num_poses=1))
        self._hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=model_path("hand").read_bytes()),
                running_mode=mode, num_hands=2))
        self._last_ts = -1

    def close(self):
        self._pose.close()
        self._hands.close()

    def process(self, frame_bgr, timestamp_ms: int) -> TrackResult:
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(timestamp_ms), self._last_ts + 1)   # must strictly increase
        self._last_ts = ts

        result = TrackResult()

        pose = self._pose.detect_for_video(image, ts)
        if pose.pose_landmarks:
            lms = pose.pose_landmarks[0]
            world = pose.pose_world_landmarks[0]
            idx = POSE_IDX[self.arm]
            for name, i in idx.items():
                result.px[name] = (lms[i].x * w, lms[i].y * h)
            visible = all((lms[idx[n]].visibility or 0.0) >= MIN_VISIBILITY
                          for n in ("elbow", "wrist"))
            if visible:
                result.arm_ok = True
                e, wr = world[idx["elbow"]], world[idx["wrist"]]
                result.forearm = np.array([wr.x - e.x, wr.y - e.y, wr.z - e.z])

        hands = self._hands.detect_for_video(image, ts)
        if result.arm_ok and hands.hand_landmarks:
            wrist_px = np.array(result.px["wrist"])
            forearm_px = np.linalg.norm(wrist_px - np.array(result.px["elbow"]))
            best, best_gap = None, None
            for i, lms in enumerate(hands.hand_landmarks):
                gap = np.linalg.norm(np.array([lms[0].x * w, lms[0].y * h]) - wrist_px)
                if best_gap is None or gap < best_gap:
                    best, best_gap = i, gap
            if best is not None and best_gap <= MAX_HAND_WRIST_GAP * max(forearm_px, 1.0):
                lms = hands.hand_landmarks[best]
                world = hands.hand_world_landmarks[best]
                for name, i in HAND_IDX.items():
                    result.px[name] = (lms[i].x * w, lms[i].y * h)
                a, b = world[HAND_IDX["hand_wrist"]], world[HAND_IDX["middle_mcp"]]
                result.hand = np.array([b.x - a.x, b.y - a.y, b.z - a.z])
                result.hand_px_all = np.array([[l.x * w, l.y * h] for l in lms])
                result.hand_world_all = np.array([[l.x, l.y, l.z] for l in world])
                result.hand_ok = True

        return result


def draw(frame_bgr, result: TrackResult):
    """Draws the tracked arm on the frame in place."""
    def pt(name):
        p = result.px.get(name)
        return (int(p[0]), int(p[1])) if p is not None else None

    arm_colour = (0, 220, 0) if result.arm_ok else (0, 0, 220)
    chain = [pt("shoulder"), pt("elbow"), pt("wrist")]
    for a, b in zip(chain, chain[1:]):
        if a and b:
            cv2.line(frame_bgr, a, b, arm_colour, 3)
    if result.hand_ok:
        for name in ("index_mcp", "middle_mcp", "pinky_mcp"):
            cv2.line(frame_bgr, pt("hand_wrist"), pt(name), (255, 180, 0), 2)
    for name in result.px:
        cv2.circle(frame_bgr, pt(name), 5, (255, 255, 255), -1)
