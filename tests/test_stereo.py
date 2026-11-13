import cv2
import numpy as np
import pytest

from calib.calibrate import calibrate_stereo, ordered
from tests.test_calib import synthetic_rig
from vision.stereo import (StereoDepth, arm_vectors, points_from_disparity,
                           points_from_triangulation)
from vision.tracker import TrackResult

F, B, W, H = 500.0, 0.08, 640, 480


def ideal_calib():
    """A perfect, already-rectified rig: focal 500 px, baseline 8 cm."""
    k = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
    p1 = np.hstack([k, np.zeros((3, 1))])
    p2 = p1.copy()
    p2[0, 3] = -F * B
    return {
        "K1": k, "D1": np.zeros(5), "K2": k, "D2": np.zeros(5),
        "R1": np.eye(3), "R2": np.eye(3), "P1": p1, "P2": p2,
        "rect_size": np.array([W, H]), "size_left": np.array([W, H]),
        "size_right": np.array([W, H]), "swapped": False,
    }


def textured_pair(shift):
    """Random texture seen with a uniform disparity of `shift` pixels."""
    rng = np.random.default_rng(0)
    left = cv2.GaussianBlur(rng.integers(0, 255, (H, W), dtype=np.uint8), (5, 5), 0)
    right = np.roll(left, -shift, axis=1)
    return left, right


def test_disparity_and_depth_of_a_flat_scene():
    depth = StereoDepth(ideal_calib())
    left, right = textured_pair(40)
    disp = depth.disparity(left, right)
    centre = disp[100:380, 200:540]
    assert np.nanmedian(centre) == pytest.approx(40, abs=0.5)

    p = depth.point_at(disp, 400, 300)
    z = F * B / 40
    assert p[2] == pytest.approx(z, rel=0.02)
    assert p[0] == pytest.approx((400 - W / 2) * z / F, rel=0.03)
    assert p[1] == pytest.approx((300 - H / 2) * z / F, rel=0.03)


def test_point_at_rejects_holes_and_out_of_frame():
    depth = StereoDepth(ideal_calib())
    disp = np.full((H, W), np.nan, np.float32)
    assert depth.point_at(disp, 320, 240) is None
    assert depth.point_at(disp, -5, 240) is None
    disp[:] = 40
    assert depth.point_at(disp, 320, 240) is not None


def test_patch_median_ignores_a_few_background_pixels():
    depth = StereoDepth(ideal_calib())
    disp = np.full((H, W), 40, np.float32)
    disp[236:240, 316:325] = 5          # a strip of far background inside the patch
    assert depth.point_at(disp, 320, 240)[2] == pytest.approx(F * B / 40)


def test_triangulate():
    depth = StereoDepth(ideal_calib())
    p = depth.triangulate((400, 300), (350, 301))
    assert p[2] == pytest.approx(F * B / 50)
    assert depth.triangulate((400, 300), (350, 340)) is None     # rows disagree
    assert depth.triangulate((400, 300), (420, 300)) is None     # negative disparity
    assert depth.triangulate(None, (350, 300)) is None


def test_landmark_points_and_vectors():
    depth = StereoDepth(ideal_calib())
    disp = np.full((H, W), 40, np.float32)
    res = TrackResult(arm_ok=True, hand_ok=True, px={
        "elbow": (300, 400), "wrist": (300, 200), "hand_wrist": (300, 200), "middle_mcp": (300, 150)})
    pts = points_from_disparity(depth, disp, res)
    forearm, hand = arm_vectors(pts)
    assert forearm[1] < 0 and abs(forearm[0]) < 1e-9 and abs(forearm[2]) < 1e-9   # straight up
    assert hand[1] < 0

    res_r = TrackResult(px={k: (x - 40, y) for k, (x, y) in res.px.items()})
    tri = points_from_triangulation(depth, res, res_r)
    for name in pts:
        assert np.allclose(tri[name], pts[name])

    missing = TrackResult(px={"elbow": (300, 400)})
    assert arm_vectors(points_from_disparity(depth, disp, missing)) == (None, None)


def test_swapped_cameras_are_detected_and_handled():
    det_l, det_r, pts, size_l, size_r, _, t_true, k_l, k_r = synthetic_rig()
    normal = calibrate_stereo(det_l, det_r, pts, size_l, size_r)
    swapped = calibrate_stereo(det_r, det_l, pts, size_r, size_l)   # roles given the wrong way round
    assert not normal["swapped"] and swapped["swapped"]
    assert swapped["baseline_m"] == pytest.approx(normal["baseline_m"], abs=1e-6)
    assert np.allclose(swapped["K1"], normal["K1"])                 # K1 is the true left camera
    assert swapped["P2"][0, 3] < 0
    assert ordered(swapped, "cfg_left", "cfg_right") == ("cfg_right", "cfg_left")
    assert ordered(normal, "cfg_left", "cfg_right") == ("cfg_left", "cfg_right")
