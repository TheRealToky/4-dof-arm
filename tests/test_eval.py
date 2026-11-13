import cv2
import numpy as np
import pytest

from eval import DEFAULT_TARGETS_CM, JOINTS, METHODS, POINTS, columns
from eval.analyze import analyse, fit, make_plots, residuals, table_text, write_summary
from vision.mono import hand_points_mono
from vision.tracker import HAND_IDX, TrackResult


# ---- fitting ----------------------------------------------------------------

def test_fit_recovers_a_rigid_move():
    rng = np.random.default_rng(0)
    truth = rng.uniform(-0.2, 0.2, (9, 3))
    rot, _ = cv2.Rodrigues(np.array([0.3, -0.5, 0.2]))
    est = (rot @ truth.T).T + np.array([0.1, -0.05, 0.6])
    err, s = residuals(est, truth)
    assert np.allclose(err, 0, atol=1e-9) and s == 1.0


def test_fit_accepts_a_mirrored_layout():
    rng = np.random.default_rng(1)
    truth = rng.uniform(-0.2, 0.2, (9, 3))
    est = truth * np.array([-1, 1, 1]) + 0.3
    assert np.allclose(residuals(est, truth)[0], 0, atol=1e-9)


def test_scale_is_recovered_only_when_asked():
    rng = np.random.default_rng(2)
    truth = rng.uniform(-0.2, 0.2, (9, 3))
    est = truth / 1.25
    err, _ = residuals(est, truth)
    assert np.abs(err).max() > 0.01
    err, s = residuals(est, truth, with_scale=True)
    assert s == pytest.approx(1.25) and np.allclose(err, 0, atol=1e-9)


def test_residuals_are_in_camera_axes():
    """An error added along the camera's z must be reported as depth, not lateral."""
    truth = np.array([[x, 0.0, z] for x in (0, 0.1, 0.2) for z in (0, 0.1, 0.2)])
    rot, _ = cv2.Rodrigues(np.array([0.9, 0.2, -0.1]))      # camera tilted w.r.t. the table
    est = (rot @ truth.T).T + np.array([0.0, 0.1, 0.6])
    bump = np.zeros_like(est)
    bump[::2, 2] = 0.01
    bump[1::2, 2] = -0.01
    err, _ = residuals(est + bump, truth)
    assert np.abs(err[:, 2]).mean() > 5 * np.abs(err[:, :2]).mean()


# ---- mono PnP ---------------------------------------------------------------

def test_mono_hand_position_from_pnp():
    rng = np.random.default_rng(3)
    world = rng.uniform(-0.08, 0.08, (21, 3))
    world[:, 2] *= 0.2                                      # a hand is fairly flat
    rot, _ = cv2.Rodrigues(np.array([0.2, -0.3, 0.1]))
    cam = (rot @ world.T).T + np.array([0.05, -0.02, 0.55])
    f, cx, cy = 700.0, 320.0, 240.0
    px = np.stack([f * cam[:, 0] / cam[:, 2] + cx, f * cam[:, 1] / cam[:, 2] + cy], axis=1)
    res = TrackResult(hand_ok=True, hand_px_all=px, hand_world_all=world)
    out = hand_points_mono(res, f, cx, cy, POINTS)
    for name in POINTS:
        assert np.allclose(out[name], cam[HAND_IDX[name]], atol=1e-3)
    assert hand_points_mono(TrackResult(), f, cx, cy, POINTS) == {p: None for p in POINTS}


# ---- whole analysis on a synthetic session ------------------------------------

def synthetic_rows(frames=40, seed=4):
    """mono: 20 % too small and noisy in depth; disparity: 25 % dropouts; triangulate: clean."""
    rng = np.random.default_rng(seed)
    rot, _ = cv2.Rodrigues(np.array([0.8, 0.1, 0.0]))
    shift = np.array([-0.1, 0.1, 0.6])
    spec = {
        "mono": dict(scale=0.8, noise=(0.001, 0.001, 0.012), drop=0.0),
        "disparity": dict(scale=1.0, noise=(0.001, 0.001, 0.004), drop=0.25),
        "triangulate": dict(scale=1.0, noise=(0.0005, 0.0005, 0.002), drop=0.0),
    }
    rows, n = [], 0
    for name, pos in DEFAULT_TARGETS_CM.items():
        true_cam = rot @ (np.array(pos) / 100) + shift
        for _ in range(frames):
            row = {c: "" for c in columns()}
            row.update(t=n / 15, target=name, frame=n, arm_ok=1, hand_ok=1,
                       ms_track_left=40, ms_track_right=40, ms_disparity=5, ms_mono=1)
            for m, s in spec.items():
                dropped = rng.random() < s["drop"]
                for p in POINTS:
                    if dropped:
                        continue
                    v = true_cam * s["scale"] + rng.normal(0, s["noise"])
                    for a, val in zip("xyz", v):
                        row[f"{m}_{p}_{a}"] = f"{val:.5f}"
                for j in JOINTS:
                    row[f"{m}_{j}"] = f"{90 + rng.normal(0, s['noise'][2] * 300):.3f}"
            rows.append(row)
            n += 1
    return rows


def test_analysis_ranks_methods_as_constructed(tmp_path):
    results, layouts = analyse(synthetic_rows(), DEFAULT_TARGETS_CM)
    mono, disp, tri = (results[m] for m in METHODS)

    assert mono["scale"] == pytest.approx(1.25, rel=0.03)
    assert tri["scale"] == pytest.approx(1.0, abs=0.02)
    assert mono["error_mm"] > 10 > tri["error_mm"]
    assert mono["error_after_scale_mm"] < mono["error_mm"] / 2
    assert tri["error_mm"] < 2 and tri["pair_distance_error_mm"] < 2

    assert disp["dropout_pct"] == pytest.approx(25, abs=6)
    assert mono["dropout_pct"] == 0 and tri["dropout_pct"] == 0
    assert mono["jitter_index_tip_mm"] > disp["jitter_index_tip_mm"] > tri["jitter_index_tip_mm"]
    assert mono["jitter_depth_index_tip_mm"] == pytest.approx(12 * 0.8 / 0.8, rel=0.2)
    assert mono["jitter_joints_deg"] > tri["jitter_joints_deg"]

    assert mono["latency_ms"] == 41 and disp["latency_ms"] == 45 and tri["latency_ms"] == 80
    assert all(r["targets_used"] == 9 for r in results.values())

    text = table_text(results)
    assert "Position error" in text and "triangulate" in text
    write_summary(results, tmp_path / "summary.csv")
    make_plots(results, layouts, tmp_path)
    for name in ("summary.csv", "accuracy.png", "jitter.png", "layout.png"):
        assert (tmp_path / name).stat().st_size > 0


def test_method_with_no_readings_is_reported_not_crashed():
    rows = synthetic_rows()
    for r in rows:
        for c in list(r):
            if c.startswith("disparity_"):
                r[c] = ""
    results, layouts = analyse(rows, DEFAULT_TARGETS_CM)
    assert results["disparity"]["dropout_pct"] == 100
    assert "error_mm" not in results["disparity"] and "disparity" not in layouts
    assert "-" in table_text(results)
