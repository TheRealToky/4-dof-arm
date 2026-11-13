"""Mono vs stereo comparison: shared names for the recorder and the analysis."""

from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
SESSION_DIR = EVAL_DIR / "sessions"
RESULT_DIR = EVAL_DIR / "results"

# Depth methods compared, in fixed order (also their colour order in plots).
METHODS = ("mono", "disparity", "triangulate")
METHOD_LABELS = {
    "mono": "Mono (one camera)",
    "disparity": "Stereo, disparity map",
    "triangulate": "Stereo, triangulated",
}
# Points logged per method. index_tip is the probe placed on each target;
# the other two are the ones the arm control actually uses.
POINTS = ("index_tip", "hand_wrist", "middle_mcp")
PROBE = "index_tip"
JOINTS = ("j0_pan", "j0_tilt", "j1_pan", "j1_tilt")

# 3 x 3 grid taped on the table, 10 cm apart: x across, y up, z away from you.
DEFAULT_TARGETS_CM = {
    f"{row}{col + 1}": [col * 10.0, 0.0, r * 10.0]
    for r, row in enumerate("ABC") for col in range(3)
}


def eval_config(config: dict) -> dict:
    cfg = {"hold_s": 2.0, "settle_s": 1.5, "targets_cm": DEFAULT_TARGETS_CM}
    cfg.update(config.get("eval", {}))
    return cfg


def columns() -> list[str]:
    cols = ["t", "target", "frame", "arm_ok", "hand_ok",
            "ms_track_left", "ms_track_right", "ms_disparity", "ms_mono"]
    for m in METHODS:
        for p in POINTS:
            cols += [f"{m}_{p}_{axis}" for axis in "xyz"]
        cols += [f"{m}_{j}" for j in JOINTS]
    return cols
