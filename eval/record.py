"""Record a known-target session for the mono vs stereo comparison.

    python -m eval.record

Preparation: tape marks on the table at the positions listed under
"eval" -> "targets_cm" in config.json (default: a 3 x 3 grid, 10 cm apart;
x across, y up, z away from you - any origin, the layout is what matters).
Measure them with a ruler; their accuracy is the ground truth.

For each target the window names it. Put the tip of your index finger on
the mark, keep your elbow, forearm and hand in view, press SPACE and hold
still until it moves on. Every frame is measured three ways at once (one
camera, stereo disparity map, stereo triangulation), so all methods see
exactly the same motion.

Keys:  SPACE  record the current target
       B      go back and redo the previous target
       S      skip this target
       Q/ESC  finish and save

Then:  python -m eval.analyze
"""

import argparse
import csv
import json
import time
from datetime import datetime

import cv2
import numpy as np

from arm.joints import load_config
from eval import JOINTS, METHODS, POINTS, PROBE, SESSION_DIR, columns, eval_config
from vision.cameras import StereoCapture
from vision.mapping import ArmMapper
from vision.mono import hand_points_mono
from vision.stereo import (StereoDepth, arm_vectors, points_from_disparity,
                           points_from_triangulation)
from vision.tracker import ArmTracker, draw

WINDOW = "Mono vs stereo - recording"
STEREO_NAMES = ("elbow", "wrist") + POINTS
METHOD_COLOURS = {"mono": (214, 120, 42), "disparity": (52, 104, 235), "triangulate": (122, 175, 27)}  # BGR


class Estimators:
    """Runs every depth method on one stereo frame and returns one CSV row."""

    def __init__(self, config):
        self.depth = StereoDepth(**config.get("stereo", {}))
        arm = config.get("mapping", {}).get("arm", "right")
        self.tracker_left = ArmTracker(arm)
        self.tracker_right = ArmTracker(arm)
        self.mapper = ArmMapper()       # default neutral; raw, unfiltered angles
        self._checked = False

    def close(self):
        self.tracker_left.close()
        self.tracker_right.close()

    def measure(self, left, right, timestamp_ms):
        """Returns (row dict, picture, per-method probe positions)."""
        d = self.depth
        if not self._checked:
            d.check_sizes(left, right)
            self._checked = True
        rect_left, rect_right = d.rectify(left, right)

        t0 = time.perf_counter()
        res_l = self.tracker_left.process(rect_left, timestamp_ms)
        t1 = time.perf_counter()
        res_r = self.tracker_right.process(rect_right, timestamp_ms)
        t2 = time.perf_counter()
        disp = d.disparity(rect_left, rect_right)
        pts_disp = points_from_disparity(d, disp, res_l, STEREO_NAMES)
        t3 = time.perf_counter()
        pts_mono = hand_points_mono(res_l, d.f, d.cx, d.cy, POINTS)
        t4 = time.perf_counter()
        pts_tri = points_from_triangulation(d, res_l, res_r, STEREO_NAMES)

        row = {
            "arm_ok": int(res_l.arm_ok), "hand_ok": int(res_l.hand_ok),
            "ms_track_left": (t1 - t0) * 1000, "ms_track_right": (t2 - t1) * 1000,
            "ms_disparity": (t3 - t2) * 1000, "ms_mono": (t4 - t3) * 1000,
        }
        # Joint angles each method would command.
        vectors = {
            "mono": (res_l.forearm if res_l.arm_ok else None,
                     res_l.hand if res_l.hand_ok else None),
            "disparity": arm_vectors(pts_disp),
            "triangulate": arm_vectors(pts_tri),
        }
        points = {"mono": pts_mono, "disparity": pts_disp, "triangulate": pts_tri}
        for m in METHODS:
            for p in POINTS:
                xyz = points[m].get(p)
                for i, axis in enumerate("xyz"):
                    row[f"{m}_{p}_{axis}"] = "" if xyz is None else f"{xyz[i]:.5f}"
            angles = self.mapper.angles(*vectors[m])
            for j, a in zip(JOINTS, angles):
                row[f"{m}_{j}"] = "" if a is None else f"{a:.3f}"

        draw(rect_left, res_l)
        probe = {m: points[m].get(PROBE) for m in METHODS}
        return row, rect_left, probe


def text(frame, message, row, colour=(255, 255, 255), scale=0.6):
    pos = (10, 26 + row * 24)
    cv2.putText(frame, message, (pos[0] + 1, pos[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, message, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def save(rows, targets, config, stamp):
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SESSION_DIR / f"{stamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns())
        writer.writeheader()
        writer.writerows(rows)
    meta = {"targets_cm": targets, "stereo": config.get("stereo", {}),
            "arm": config.get("mapping", {}).get("arm", "right")}
    (SESSION_DIR / f"{stamp}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="path to config.json")
    args = parser.parse_args()

    config = load_config(args.config)
    ev = eval_config(config)
    targets = ev["targets_cm"]
    names = list(targets)
    cam = config.get("cameras", {})

    try:
        est = Estimators(config)
    except FileNotFoundError as e:
        raise SystemExit(str(e))

    rows = []
    index = 0
    state, state_until = "wait", 0.0
    frame_no = 0
    start = time.monotonic()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with StereoCapture(cam.get("left", 0), cam.get("right", 1),
                       cam.get("width", 640), cam.get("height", 480)) as capture:
        try:
            while index < len(names):
                left, right = capture.read()
                if left is None:
                    print("A camera stopped delivering frames.")
                    break
                now = time.monotonic()
                row, picture, probe = est.measure(left, right, int((now - start) * 1000))
                name = names[index]

                if state == "settle" and now >= state_until:
                    state, state_until = "record", now + ev["hold_s"]
                if state == "record":
                    row.update({"t": f"{now - start:.4f}", "target": name, "frame": frame_no})
                    rows.append(row)
                    frame_no += 1
                    if now >= state_until:
                        state = "wait"
                        index += 1

                view = cv2.flip(picture, 1)
                if index < len(names):
                    x, y, z = targets[names[index]]
                    text(view, f"Target {index + 1}/{len(names)}:  {names[index]}   "
                               f"(x {x:g}, y {y:g}, z {z:g} cm)", 0, (0, 255, 255), 0.7)
                if state == "wait":
                    text(view, "Index fingertip on the mark, then SPACE", 1)
                elif state == "settle":
                    text(view, f"Hold still...  {state_until - now:.1f}", 1, (0, 200, 255))
                else:
                    text(view, f"RECORDING  {state_until - now:.1f}", 1, (0, 0, 255))
                for i, m in enumerate(METHODS):
                    p = probe[m]
                    reading = "no reading" if p is None else f"{p[2] * 100:6.1f} cm away"
                    text(view, f"{m:<12}{reading}", 3 + i, METHOD_COLOURS[m], 0.55)
                text(view, "B back   S skip   Q finish", 7, (200, 200, 200), 0.5)
                cv2.imshow(WINDOW, view)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if state == "wait":
                    if key == ord(" "):
                        state, state_until = "settle", now + ev["settle_s"]
                    elif key == ord("s"):
                        index += 1
                    elif key == ord("b") and index > 0:
                        index -= 1
                        rows = [r for r in rows if r["target"] != names[index]]
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            est.close()
            cv2.destroyAllWindows()

    if rows:
        path = save(rows, targets, config, stamp)
        done = len({r["target"] for r in rows})
        print(f"Saved {len(rows)} frames over {done}/{len(names)} targets to {path}")
        print("Next: python -m eval.analyze")
    else:
        print("Nothing recorded.")


if __name__ == "__main__":
    main()
