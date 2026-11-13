"""Calibrate the stereo rig from the pairs saved by capture_pairs.

    python -m calib.calibrate

Steps: intrinsics for each camera on its own (each at its own resolution,
which is what lets two different cameras work together), then the rotation
and translation between them, then rectification. Writes calib/stereo.npz
and calib/rectified_check.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from arm.joints import load_config
from calib.board import BoardDetector, board_config

CALIB_DIR = Path(__file__).parent
PAIR_DIR = CALIB_DIR / "pairs"
STEREO_FILE = CALIB_DIR / "stereo.npz"

MIN_CORNERS = 6     # per view, for intrinsics
MIN_COMMON = 6      # per pair, for extrinsics


def calibrate_single(detections, corners_3d, image_size):
    """detections: list of (corners Nx2, ids N). Returns (rms, K, D, views_used)."""
    obj, img = [], []
    for corners, ids in detections:
        if len(ids) >= MIN_CORNERS:
            obj.append(corners_3d[ids].reshape(-1, 1, 3).astype(np.float32))
            img.append(corners.reshape(-1, 1, 2).astype(np.float32))
    if len(obj) < 5:
        raise ValueError(f"only {len(obj)} usable views; need at least 5 (aim for 20+)")
    rms, k, d, _, _ = cv2.calibrateCamera(obj, img, tuple(image_size), None, None)
    return rms, k, d, len(obj)


def rectify(k1, d1, k2, d2, r, t, size_left):
    """Rectifying rotations plus projection matrices for a common output image.

    Both views are rectified onto an image with the left camera's size,
    focal length and field of view. cv2.stereoRectify is used only for the
    rotations: its own projection matrices assume both cameras have the same
    resolution and zoom, which is wrong for two different webcams and ends
    up cropping the picture heavily.
    """
    rect_size = tuple(int(v) for v in size_left)
    r1, r2, _, _, _, _, _ = cv2.stereoRectify(
        k1, d1, k2, d2, rect_size, r, t, flags=cv2.CALIB_ZERO_DISPARITY)

    f = float(k1[0, 0] + k1[1, 1]) / 2.0
    cx, cy = rect_size[0] / 2.0, rect_size[1] / 2.0
    tx = float((r2 @ np.asarray(t).reshape(3))[0])      # right camera offset, negative
    p1 = np.array([[f, 0, cx, 0], [0, f, cy, 0], [0, 0, 1, 0]], dtype=float)
    p2 = p1.copy()
    p2[0, 3] = f * tx
    q = np.array([[1, 0, 0, -cx],
                  [0, 1, 0, -cy],
                  [0, 0, 0, f],
                  [0, 0, -1.0 / tx, 0]], dtype=float)
    return r1, r2, p1, p2, q, rect_size


def calibrate_stereo(det_left, det_right, corners_3d, size_left, size_right, _swapped=False):
    """Full calibration from per-pair detections. Returns a dict of results.

    det_left / det_right: lists of (corners Nx2, ids N), same length, where
    index i of each is the same moment in time.

    If the camera given as "left" turns out to be the right-hand one (as
    seen from behind the cameras), the calibration is redone with the two
    roles exchanged and the result is marked "swapped". Everything in the
    result (K1, size_left, ...) then describes the true left camera, and
    ordered() puts incoming frames in the matching order.
    """
    rms_l, k1, d1, n_l = calibrate_single(det_left, corners_3d, size_left)
    rms_r, k2, d2, n_r = calibrate_single(det_right, corners_3d, size_right)

    obj, img_l, img_r = [], [], []
    for (cl, il), (cr, ir) in zip(det_left, det_right):
        common, idx_l, idx_r = np.intersect1d(il, ir, return_indices=True)
        if len(common) >= MIN_COMMON:
            obj.append(corners_3d[common].reshape(-1, 1, 3).astype(np.float32))
            img_l.append(cl[idx_l].reshape(-1, 1, 2).astype(np.float32))
            img_r.append(cr[idx_r].reshape(-1, 1, 2).astype(np.float32))
    if len(obj) < 5:
        raise ValueError(f"only {len(obj)} pairs where both cameras see the board; need at least 5")

    rms_s, k1, d1, k2, d2, r, t, _, _ = cv2.stereoCalibrate(
        obj, img_l, img_r, k1, d1, k2, d2, tuple(size_left),
        flags=cv2.CALIB_FIX_INTRINSIC)

    # The right camera must sit at negative x in the left camera's frame.
    if float(np.asarray(t).ravel()[0]) > 0:
        if _swapped:
            raise ValueError("could not determine which camera is on the left")
        return calibrate_stereo(det_right, det_left, corners_3d, size_right, size_left, _swapped=True)

    r1, r2, p1, p2, q, rect_size = rectify(k1, d1, k2, d2, r, t, size_left)

    # After rectification the same point must sit on the same row in both views.
    row_errors = []
    for pl, pr in zip(img_l, img_r):
        yl = cv2.undistortPoints(pl, k1, d1, R=r1, P=p1).reshape(-1, 2)[:, 1]
        yr = cv2.undistortPoints(pr, k2, d2, R=r2, P=p2).reshape(-1, 2)[:, 1]
        row_errors.append(np.abs(yl - yr))
    row_error = float(np.mean(np.concatenate(row_errors)))

    return {
        "K1": k1, "D1": d1, "K2": k2, "D2": d2, "R": r, "T": t,
        "R1": r1, "R2": r2, "P1": p1, "P2": p2, "Q": q,
        "size_left": np.array(size_left), "size_right": np.array(size_right),
        "rect_size": np.array(rect_size),
        "rms_left": rms_l, "rms_right": rms_r, "rms_stereo": rms_s,
        "row_error_px": row_error, "baseline_m": float(np.linalg.norm(t)),
        "views_left": n_l, "views_right": n_r, "pairs_used": len(obj),
        "swapped": bool(_swapped),
    }


def ordered(calib, left, right):
    """Puts a (config-left, config-right) pair into true (left, right) order."""
    return (right, left) if bool(calib.get("swapped", False)) else (left, right)


def rectify_maps(calib):
    """Returns ((map1_l, map2_l), (map1_r, map2_r)) for cv2.remap."""
    size = tuple(int(v) for v in calib["rect_size"])
    left = cv2.initUndistortRectifyMap(calib["K1"], calib["D1"], calib["R1"], calib["P1"], size, cv2.CV_16SC2)
    right = cv2.initUndistortRectifyMap(calib["K2"], calib["D2"], calib["R2"], calib["P2"], size, cv2.CV_16SC2)
    return left, right


def save(calib, path=STEREO_FILE):
    np.savez(path, **calib)


def load(path=STEREO_FILE):
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def report(c):
    def verdict(value, good, okay):
        return "good" if value <= good else ("acceptable" if value <= okay else "POOR - recapture")
    lines = [
        f"Left camera   : {c['views_left']} views, RMS {c['rms_left']:.3f} px  ({verdict(c['rms_left'], 0.5, 1.0)})",
        f"Right camera  : {c['views_right']} views, RMS {c['rms_right']:.3f} px  ({verdict(c['rms_right'], 0.5, 1.0)})",
        f"Stereo        : {c['pairs_used']} pairs, RMS {c['rms_stereo']:.3f} px  ({verdict(c['rms_stereo'], 1.0, 2.0)})",
        f"Row alignment : {c['row_error_px']:.3f} px mean after rectification  ({verdict(c['row_error_px'], 0.5, 1.0)})",
        f"Baseline      : {c['baseline_m'] * 100:.2f} cm  (compare with a ruler, lens centre to lens centre)",
    ]
    if bool(c.get("swapped", False)):
        lines.append('Note          : the camera set as "left" in config.json is physically the right-hand one. '
                     "This is handled automatically; nothing to change. "
                     '"Left camera" above means the true left one.')
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="path to config.json")
    parser.add_argument("--pairs", default=str(PAIR_DIR))
    args = parser.parse_args()

    detector = BoardDetector(board_config(load_config(args.config)))
    pair_dir = Path(args.pairs)
    left_files = sorted(pair_dir.glob("left_*.png"))
    if not left_files:
        raise SystemExit(f"No pairs in {pair_dir}. Run: python -m calib.capture_pairs")

    det_l, det_r, first = [], [], None
    size_l = size_r = None
    for lf in left_files:
        rf = lf.with_name(lf.name.replace("left_", "right_"))
        left, right = cv2.imread(str(lf)), cv2.imread(str(rf))
        if left is None or right is None:
            print(f"Skipping {lf.name}: missing or unreadable partner")
            continue
        size_l, size_r = (left.shape[1], left.shape[0]), (right.shape[1], right.shape[0])
        det_l.append(detector.detect(left))
        det_r.append(detector.detect(right))
        first = first or (left, right)

    calib = calibrate_stereo(det_l, det_r, detector.corners_3d, size_l, size_r)
    save(calib)
    print(report(calib))
    print(f"Saved {STEREO_FILE}")

    # Visual check: rectified pair with horizontal lines.
    (m1l, m2l), (m1r, m2r) = rectify_maps(calib)
    true_left, true_right = ordered(calib, *first)
    rl = cv2.remap(true_left, m1l, m2l, cv2.INTER_LINEAR)
    rr = cv2.remap(true_right, m1r, m2r, cv2.INTER_LINEAR)
    check = np.hstack([rl, rr])
    for y in range(20, check.shape[0], 40):
        cv2.line(check, (0, y), (check.shape[1], y), (0, 255, 0), 1)
    out = CALIB_DIR / "rectified_check.png"
    cv2.imwrite(str(out), check)
    print(f"Wrote {out} - each green line should cross the same features in both halves.")


if __name__ == "__main__":
    main()
