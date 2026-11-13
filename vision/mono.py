"""Metric hand position from ONE camera.

MediaPipe's hand model gives the 21 hand landmarks as a small 3D shape in
metres (its learned idea of how big a hand is), with no position in space.
Fitting that shape to where the landmarks appear in the image (PnP) gives
the hand's position in the camera frame. Depth here rests entirely on the
assumed hand size, which is exactly the weakness stereo is meant to remove.
"""

from __future__ import annotations

import cv2
import numpy as np

from .tracker import HAND_IDX


def hand_points_mono(result, f: float, cx: float, cy: float, names=("hand_wrist", "middle_mcp")) -> dict:
    """Landmark name -> XYZ in metres (camera frame), or None.

    f, cx, cy: focal length and principal point of the (undistorted) image
    the landmarks were found in.
    """
    out = {name: None for name in names}
    if result.hand_px_all is None or result.hand_world_all is None:
        return out
    k = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=float)
    try:
        ok, rvec, tvec = cv2.solvePnP(
            result.hand_world_all.astype(np.float64).reshape(-1, 1, 3),
            result.hand_px_all.astype(np.float64).reshape(-1, 1, 2),
            k, None, flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        return out
    if not ok:
        return out
    rot, _ = cv2.Rodrigues(rvec)
    cam = (rot @ result.hand_world_all.T).T + tvec.reshape(1, 3)
    if not np.all(np.isfinite(cam)) or np.median(cam[:, 2]) <= 0:
        return out
    for name in names:
        out[name] = cam[HAND_IDX[name]]
    return out
