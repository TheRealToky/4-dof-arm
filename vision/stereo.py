"""Stereo depth: rectification, disparity map, and metric 3D at tracked landmarks.

All 3D points are in metres in the rectified left camera's frame:
x right, y down, z away from the camera.
"""

from __future__ import annotations

import cv2
import numpy as np

from calib.calibrate import STEREO_FILE, load, ordered, rectify_maps

# Landmarks that get a 3D position. Forearm = elbow -> wrist (from pose),
# hand = hand_wrist -> middle_mcp (from the hand model).
POINT_NAMES = ("elbow", "wrist", "hand_wrist", "middle_mcp")

MAX_ROW_GAP_PX = 12.0       # triangulation: same landmark must be on (nearly) the same row
MIN_DEPTH_M, MAX_DEPTH_M = 0.15, 5.0


class StereoDepth:
    """Defaults were tuned on real pairs from two different webcams:

    equalise - local contrast equalisation of both views before matching.
               Two different cameras expose and sharpen differently; without
               this most of the image finds no match.
    scale    - matching runs on images shrunk by this factor. Half size hides
               the ~1 px of leftover row misalignment, removes most false
               matches and is several times faster. Disparities are scaled
               back, so results are still in full-resolution pixels.
    num_disparities - search range in full-resolution pixels; sets the
               nearest measurable distance (focal_px * baseline / range).
    """

    def __init__(self, calib: dict | None = None, num_disparities: int = 192,
                 block_size: int = 9, patch: int = 9, scale: float = 0.5,
                 equalise: bool = True):
        if calib is None:
            if not STEREO_FILE.exists():
                raise FileNotFoundError(
                    f"{STEREO_FILE} not found. Calibrate first: python -m calib.calibrate")
            calib = load()
        self.calib = calib
        self.size = tuple(int(v) for v in calib["rect_size"])       # (width, height)
        p1, p2 = calib["P1"], calib["P2"]
        self.f = float(p1[0, 0])
        self.cx, self.cy = float(p1[0, 2]), float(p1[1, 2])
        self.baseline = abs(float(p2[0, 3])) / self.f
        self.patch = patch
        (self._m1l, self._m2l), (self._m1r, self._m2r) = rectify_maps(calib)

        self.scale = scale
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if equalise else None
        scaled_range = max(16, int(num_disparities * scale) // 16 * 16)
        self.num_disparities = scaled_range / scale
        self.min_depth = self.f * self.baseline / self.num_disparities
        self._sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=scaled_range, blockSize=block_size,
            P1=8 * block_size * block_size, P2=32 * block_size * block_size,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    # ---- images ----------------------------------------------------------

    def check_sizes(self, left, right):
        """Raises if the frames are not the sizes the rig was calibrated at."""
        left, right = ordered(self.calib, left, right)
        for frame, key in ((left, "size_left"), (right, "size_right")):
            want = tuple(int(v) for v in self.calib[key])
            got = (frame.shape[1], frame.shape[0])
            if got != want:
                raise ValueError(f"camera gives {got[0]}x{got[1]} but was calibrated at "
                                 f"{want[0]}x{want[1]}; recalibrate or fix the resolution")

    def rectify(self, left, right):
        """Takes frames in config.json order; returns the rectified true (left, right)."""
        left, right = ordered(self.calib, left, right)
        return (cv2.remap(left, self._m1l, self._m2l, cv2.INTER_LINEAR),
                cv2.remap(right, self._m1r, self._m2r, cv2.INTER_LINEAR))

    def disparity(self, rect_left, rect_right):
        """Disparity in pixels as float32; NaN where no match was found."""
        gl = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY) if rect_left.ndim == 3 else rect_left
        gr = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY) if rect_right.ndim == 3 else rect_right
        full_size = (gl.shape[1], gl.shape[0])
        if self._clahe is not None:
            gl, gr = self._clahe.apply(gl), self._clahe.apply(gr)
        if self.scale != 1.0:
            gl = cv2.resize(gl, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            gr = cv2.resize(gr, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        disp = self._sgbm.compute(gl, gr).astype(np.float32) / (16.0 * self.scale)
        disp[disp <= 0] = np.nan
        if self.scale != 1.0:
            disp = cv2.resize(disp, full_size, interpolation=cv2.INTER_NEAREST)
        return disp

    def colourise(self, disp):
        """Disparity map as a BGR picture (near = warm, no data = black)."""
        scaled = np.nan_to_num(disp, nan=0.0) * (255.0 / self.num_disparities)
        colour = cv2.applyColorMap(np.clip(scaled, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        colour[np.isnan(disp)] = 0
        return colour

    # ---- 3D --------------------------------------------------------------

    def point_from_disparity_value(self, x: float, y: float, d: float):
        """Pixel (x, y) in the rectified left image with disparity d -> XYZ in metres."""
        if d is None or not np.isfinite(d) or d <= 0:
            return None
        z = self.f * self.baseline / d
        if not (MIN_DEPTH_M <= z <= MAX_DEPTH_M):
            return None
        return np.array([(x - self.cx) * z / self.f, (y - self.cy) * z / self.f, z])

    def point_at(self, disp, x: float, y: float):
        """3D point at a pixel of the rectified left image, from the disparity map.

        Uses the median disparity in a small patch, which rejects holes and
        most background pixels around a landmark.
        """
        h, w = disp.shape
        xi, yi, r = int(round(x)), int(round(y)), self.patch // 2
        if not (0 <= xi < w and 0 <= yi < h):
            return None
        window = disp[max(0, yi - r):yi + r + 1, max(0, xi - r):xi + r + 1]
        valid = window[np.isfinite(window)]
        if valid.size < max(3, window.size // 4):
            return None
        return self.point_from_disparity_value(x, y, float(np.median(valid)))

    def triangulate(self, px_left, px_right):
        """3D point from the same landmark seen in both rectified images."""
        if px_left is None or px_right is None:
            return None
        if abs(px_left[1] - px_right[1]) > MAX_ROW_GAP_PX:
            return None
        return self.point_from_disparity_value(px_left[0], px_left[1], px_left[0] - px_right[0])


def points_from_disparity(depth: StereoDepth, disp, result) -> dict:
    """Landmark name -> XYZ (or None), sampling the disparity map at each landmark."""
    out = {}
    for name in POINT_NAMES:
        p = result.px.get(name)
        out[name] = depth.point_at(disp, p[0], p[1]) if p is not None else None
    return out


def points_from_triangulation(depth: StereoDepth, result_left, result_right) -> dict:
    """Landmark name -> XYZ (or None), matching landmarks tracked in both views."""
    return {name: depth.triangulate(result_left.px.get(name), result_right.px.get(name))
            for name in POINT_NAMES}


def arm_vectors(points: dict):
    """(forearm, hand) direction vectors from 3D points; None where a point is missing."""
    def diff(a, b):
        pa, pb = points.get(a), points.get(b)
        return pb - pa if pa is not None and pb is not None else None
    return diff("elbow", "wrist"), diff("hand_wrist", "middle_mcp")
