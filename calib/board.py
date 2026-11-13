"""The ChArUco calibration board, built from the "board" section of config.json."""

from __future__ import annotations

import cv2
import numpy as np

DEFAULT_BOARD = {
    "squares_x": 7,
    "squares_y": 5,
    "square_mm": 35.0,
    "marker_mm": 26.0,
    "dictionary": "DICT_5X5_50",
}


def board_config(config: dict) -> dict:
    return {**DEFAULT_BOARD, **config.get("board", {})}


def make_board(cfg: dict):
    """Board with lengths in metres, so calibration results come out in metres."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg["dictionary"]))
    return cv2.aruco.CharucoBoard(
        (cfg["squares_x"], cfg["squares_y"]),
        cfg["square_mm"] / 1000.0, cfg["marker_mm"] / 1000.0, dictionary)


class BoardDetector:
    def __init__(self, cfg: dict):
        self.board = make_board(cfg)
        self._detector = cv2.aruco.CharucoDetector(self.board)
        self.corners_3d = np.asarray(self.board.getChessboardCorners(), dtype=np.float32)

    def detect(self, image):
        """Returns (corners Nx2 float32, ids N int) for the inner corners found."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners, ids, _, _ = self._detector.detectBoard(gray)
        if ids is None or len(ids) == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0,), int)
        return corners.reshape(-1, 2).astype(np.float32), ids.reshape(-1).astype(int)

    def object_points(self, ids):
        return self.corners_3d[ids]
