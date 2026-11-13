import cv2
import numpy as np
import pytest

from calib.board import DEFAULT_BOARD, BoardDetector
from calib.calibrate import calibrate_stereo, load, rectify_maps, save


def test_board_is_detected_in_its_own_image():
    det = BoardDetector(DEFAULT_BOARD)
    image = det.board.generateImage((1400, 1000), marginSize=40)
    corners, ids = det.detect(image)
    expected = (DEFAULT_BOARD["squares_x"] - 1) * (DEFAULT_BOARD["squares_y"] - 1)
    assert len(ids) == expected
    assert corners.shape == (expected, 2)
    # Lengths are in metres.
    spacing = np.linalg.norm(det.corners_3d[1] - det.corners_3d[0])
    assert spacing == pytest.approx(DEFAULT_BOARD["square_mm"] / 1000)


def synthetic_rig(n_views=30, noise_px=0.1, seed=1):
    """Two deliberately different cameras looking at the board in random poses."""
    rng = np.random.default_rng(seed)
    det = BoardDetector(DEFAULT_BOARD)
    pts = det.corners_3d

    size_l, size_r = (640, 480), (1280, 720)
    k_l = np.array([[520.0, 0, 322], [0, 518, 238], [0, 0, 1]])
    k_r = np.array([[930.0, 0, 655], [0, 934, 350], [0, 0, 1]])
    d_l = np.array([0.08, -0.15, 0.001, -0.001, 0.0])
    d_r = np.array([-0.05, 0.06, 0.0, 0.002, 0.0])
    # Right camera 9 cm to the right, slightly toed in and rolled.
    r_true, _ = cv2.Rodrigues(np.array([0.01, -0.04, 0.015]))
    t_true = np.array([-0.09, 0.004, 0.003])

    det_l, det_r = [], []
    centre = pts.mean(axis=0)
    for _ in range(n_views):
        rvec = rng.uniform(-0.5, 0.5, 3)
        tvec = np.array([rng.uniform(-0.12, 0.12), rng.uniform(-0.08, 0.08), rng.uniform(0.35, 0.8)])
        rot, _ = cv2.Rodrigues(rvec)
        tvec = tvec - rot @ centre            # put the board's centre at tvec
        r2 = r_true @ rot
        t2 = r_true @ tvec + t_true
        views = []
        for k, d, r, t, size in ((k_l, d_l, rot, tvec, size_l), (k_r, d_r, r2, t2, size_r)):
            proj, _ = cv2.projectPoints(pts, cv2.Rodrigues(r)[0], t, k, d)
            proj = proj.reshape(-1, 2) + rng.normal(0, noise_px, (len(pts), 2))
            inside = (proj[:, 0] > 5) & (proj[:, 0] < size[0] - 5) & (proj[:, 1] > 5) & (proj[:, 1] < size[1] - 5)
            ids = np.flatnonzero(inside)
            views.append((proj[ids].astype(np.float32), ids))
        det_l.append(views[0])
        det_r.append(views[1])
    return det_l, det_r, pts, size_l, size_r, r_true, t_true, k_l, k_r


def test_stereo_calibration_recovers_a_mismatched_rig(tmp_path):
    det_l, det_r, pts, size_l, size_r, r_true, t_true, k_l, k_r = synthetic_rig()
    c = calibrate_stereo(det_l, det_r, pts, size_l, size_r)

    assert c["rms_left"] < 0.3 and c["rms_right"] < 0.3 and c["rms_stereo"] < 0.3
    assert c["row_error_px"] < 0.3
    # Rectified view keeps the left camera's zoom, despite the mismatched pair.
    assert c["P1"][0, 0] == pytest.approx(k_l[0, 0], rel=0.03)
    # A point 0.5 m straight ahead must come back out of Q at 0.5 m.
    f, b = c["P1"][0, 0], c["baseline_m"]
    disparity = f * b / 0.5
    x, y, z, w = c["Q"] @ np.array([c["P1"][0, 2], c["P1"][1, 2], disparity, 1.0])
    assert z / w == pytest.approx(0.5, rel=1e-3)
    assert c["baseline_m"] == pytest.approx(np.linalg.norm(t_true), abs=0.002)
    assert np.allclose(c["T"].ravel(), t_true, atol=0.003)
    assert np.allclose(c["R"], r_true, atol=0.01)
    assert c["K1"][0, 0] == pytest.approx(k_l[0, 0], rel=0.02)
    assert c["K2"][0, 0] == pytest.approx(k_r[0, 0], rel=0.02)

    # Round-trips through the file, and the rectification maps build.
    path = tmp_path / "stereo.npz"
    save(c, path)
    loaded = load(path)
    assert np.allclose(loaded["Q"], c["Q"])
    (m1, _), (m2, _) = rectify_maps(loaded)
    assert m1.shape[:2] == (480, 640) and m2.shape[:2] == (480, 640)


def test_too_few_views_is_reported():
    det_l, det_r, pts, size_l, size_r, *_ = synthetic_rig(n_views=3)
    with pytest.raises(ValueError):
        calibrate_stereo(det_l, det_r, pts, size_l, size_r)
