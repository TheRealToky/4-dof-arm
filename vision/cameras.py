"""Camera capture."""

from __future__ import annotations

import sys

import cv2


class Camera:
    """One webcam. Use as a context manager."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self.index = index
        self.width = width
        self.height = height
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
