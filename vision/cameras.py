"""Camera capture."""

from __future__ import annotations

import sys
import time

import cv2


class Camera:
    """One webcam. Use as a context manager."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480,
                 lock_focus: bool = False):
        self.index = index
        self.width = width
        self.height = height
        self.lock_focus = lock_focus
        self.cap = None

    def open(self) -> "Camera":
        # DirectShow opens much faster than the default backend on Windows
        # and lets resolution requests take effect.
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always the newest frame
        if self.lock_focus:
            # Autofocus changes the lens geometry and silently invalidates a
            # calibration. Ignored by cameras that have fixed focus.
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return self

    def read(self):
        """Returns a BGR frame, or None if the camera gave nothing."""
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()


class StereoCapture:
    """Two webcams read as a pair. Use as a context manager.

    USB webcams cannot be hardware-synchronised. grab() is called on both
    first (fast, just latches a frame) and the slower decode happens after,
    which keeps the two exposures as close together as the drivers allow.
    """

    def __init__(self, left_index: int, right_index: int, width: int = 640, height: int = 480):
        if left_index == right_index:
            raise ValueError("left and right camera indices must differ")
        self.left = Camera(left_index, width, height, lock_focus=True)
        self.right = Camera(right_index, width, height, lock_focus=True)
        self.skew_ms = 0.0      # time between the two grab() calls, last pair

    def open(self) -> "StereoCapture":
        self.left.open()
        try:
            self.right.open()
        except Exception:
            self.left.close()
            raise
        return self

    def read(self):
        """Returns (left, right) BGR frames, or (None, None) if either failed."""
        ok_l = self.left.cap.grab()
        t_left = time.perf_counter()
        ok_r = self.right.cap.grab()
        self.skew_ms = (time.perf_counter() - t_left) * 1000
        if not (ok_l and ok_r):
            return None, None
        ok_l, left = self.left.cap.retrieve()
        ok_r, right = self.right.cap.retrieve()
        if not (ok_l and ok_r):
            return None, None
        return left, right

    def close(self):
        self.left.close()
        self.right.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
