"""Drive the arm from your forearm and hand, seen by a camera.

    python vision_control.py --no-arm                    # one camera, overlay only
    python vision_control.py                             # one camera, with the arm
    python vision_control.py --mode stereo --no-arm      # two cameras, depth from the disparity map
    python vision_control.py --mode stereo --depth triangulate
    python vision_control.py --port COM3 --camera 1

Keys:  SPACE  engage / pause arm output (starts paused)
       N      set the current pose as neutral (= 90 deg on every servo)
       H      send the arm home and pause
       Q/ESC  quit
"""

import argparse
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from arm.joints import JointController, load_config
from arm.link import ArmLink
from vision.cameras import Camera, StereoCapture
from vision.mapping import ArmMapper
from vision.tracker import ArmTracker, draw

WINDOW = "4-DOF arm - vision control"
JOINT_LABELS = ["J0 pan", "J0 tilt", "J1 pan", "J1 tilt"]


@dataclass
class Observation:
    """What one frame told us, whichever way depth was obtained."""
    view: np.ndarray                    # camera picture with the arm drawn on it
    forearm: np.ndarray | None = None   # elbow -> wrist direction, None if unknown
    hand: np.ndarray | None = None      # wrist -> knuckle direction, None if unknown
    seen: bool = False                  # arm found in the picture at all
    status: str = ""
    panel: np.ndarray | None = None     # optional second picture (disparity map)
    labels: dict = field(default_factory=dict)   # pixel (x, y) -> text, drawn on the view


class MonoSource:
    """One camera; 3D directions are MediaPipe's own learned estimate."""

    def __init__(self, config, camera_index=None):
        cam = config.get("cameras", {})
        index = camera_index if camera_index is not None else cam.get("mono", 0)
        self.camera = Camera(index, cam.get("width", 640), cam.get("height", 480))
        self.tracker = ArmTracker(config.get("mapping", {}).get("arm", "right"))

    def open(self):
        self.camera.open()

    def close(self):
        self.camera.close()
        self.tracker.close()

    def step(self, timestamp_ms) -> Observation | None:
        frame = self.camera.read()
        if frame is None:
            return None
        result = self.tracker.process(frame, timestamp_ms)
        draw(frame, result)
        obs = Observation(view=frame, seen=result.arm_ok)
        if result.arm_ok:
            obs.forearm = result.forearm
            obs.hand = result.hand if result.hand_ok else None
            obs.status = "arm+hand" if result.hand_ok else "arm only"
        else:
            obs.status = "NOT TRACKED"
        return obs


class StereoSource:
    """Two calibrated cameras; 3D directions come from measured depth.

    depth="disparity":   track in the left view, read each landmark's depth
                         from the dense disparity map.
    depth="triangulate": track in both views and intersect matching landmarks.
    """

    def __init__(self, config, depth_method="disparity"):
        from vision.stereo import StereoDepth

        cam = config.get("cameras", {})
        self.capture = StereoCapture(cam.get("left", 0), cam.get("right", 1),
                                     cam.get("width", 640), cam.get("height", 480))
        self.depth = StereoDepth(**config.get("stereo", {}))
        self.method = depth_method
        arm = config.get("mapping", {}).get("arm", "right")
        self.tracker = ArmTracker(arm)
        self.tracker_right = ArmTracker(arm) if depth_method == "triangulate" else None
        self.frames = 0
        self.dropouts = 0       # arm seen, but no usable depth for the forearm
        self._checked = False

    def open(self):
        self.capture.open()

    def close(self):
        self.capture.close()
        self.tracker.close()
        if self.tracker_right is not None:
            self.tracker_right.close()

    def step(self, timestamp_ms) -> Observation | None:
        from vision.stereo import arm_vectors, points_from_disparity, points_from_triangulation

        left, right = self.capture.read()
        if left is None:
            return None
        if not self._checked:
            self.depth.check_sizes(left, right)
            self._checked = True
        rect_left, rect_right = self.depth.rectify(left, right)
        result = self.tracker.process(rect_left, timestamp_ms)
        disp = self.depth.disparity(rect_left, rect_right)

        if self.method == "triangulate":
            result_right = self.tracker_right.process(rect_right, timestamp_ms)
            points = points_from_triangulation(self.depth, result, result_right)
        else:
            points = points_from_disparity(self.depth, disp, result)

        draw(rect_left, result)
        obs = Observation(view=rect_left, seen=result.arm_ok, panel=self.depth.colourise(disp))
        for name, p in points.items():
            if p is not None and name in result.px and name != "hand_wrist":
                obs.labels[result.px[name]] = f"{p[2] * 100:.0f} cm"

        if result.arm_ok:
            self.frames += 1
            forearm, hand = arm_vectors(points)
            obs.forearm = forearm
            obs.hand = hand if result.hand_ok else None
            if forearm is None:
                self.dropouts += 1
                obs.status = "arm seen, NO DEPTH"
            else:
                obs.status = "arm+hand" if obs.hand is not None else "arm only"
        else:
            obs.status = "NOT TRACKED"
        rate = 100.0 * self.dropouts / self.frames if self.frames else 0.0
        obs.status += f"   [{self.method}, depth dropouts {rate:.0f}%, skew {self.capture.skew_ms:.0f} ms]"
        return obs


def shadow_text(frame, text, pos, colour, scale=0.55):
    """Text with a 1 px drop shadow so it stays readable on any background."""
    x, y = int(pos[0]), int(pos[1])
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def put_text(frame, text, row, colour=(255, 255, 255)):
    shadow_text(frame, text, (10, 24 + row * 22), colour)


def render(obs, engaged, link, fps, targets, controller):
    """Mirrored picture (reads like a mirror) with status text on top."""
    view = cv2.flip(obs.view, 1)
    width = view.shape[1]
    for (x, y), text in obs.labels.items():
        shadow_text(view, text, (width - 1 - x + 8, y - 8), (0, 255, 255), 0.5)

    state = "ENGAGED" if engaged else "PAUSED (space to engage)"
    if link.dry_run:
        state += "  [no arm]"
    put_text(view, f"{state}   {fps:4.1f} fps", 0, (0, 255, 0) if engaged else (0, 200, 255))
    usable = obs.forearm is not None
    put_text(view, f"tracking: {obs.status}", 1, (255, 255, 255) if usable else (0, 0, 255))
    for i, label in enumerate(JOINT_LABELS):
        target = "  --" if targets[i] is None else f"{targets[i]:5.1f}"
        put_text(view, f"{label}: target {target}   servo {controller.servo_deg[i]:5.1f}", 2 + i)
    if not link.dry_run:
        put_text(view, f"sent {link.sent_count}  ok {link.ok_count}  err {link.error_count}", 6)

    if obs.panel is not None:
        view = np.hstack([view, cv2.flip(obs.panel, 1)])
    return view


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["mono", "stereo"], default="mono")
    parser.add_argument("--depth", choices=["disparity", "triangulate"], default="disparity",
                        help="stereo mode only: how landmark depth is measured")
    parser.add_argument("--no-arm", action="store_true", help="do not open the serial port")
    parser.add_argument("--port", help="serial port, e.g. COM3")
    parser.add_argument("--camera", type=int, help="mono mode: camera index (overrides config.json)")
    parser.add_argument("--config", help="path to config.json")
    args = parser.parse_args()

    config = load_config(args.config)
    serial_cfg = config.get("serial", {})
    map_cfg = config.get("mapping", {})
    hold_s = map_cfg.get("hold_s", 1.0)

    controller = JointController(config)
    mapper = ArmMapper(map_cfg.get("gains", [1, 1, 1, 1]))
    try:
        if args.mode == "stereo":
            source = StereoSource(config, args.depth)
        else:
            source = MonoSource(config, args.camera)
    except FileNotFoundError as e:
        raise SystemExit(str(e))
    link = ArmLink(port=args.port or serial_cfg.get("port"),
                   baud=serial_cfg.get("baud", 115200), dry_run=args.no_arm)

    engaged = False
    targets = controller.home_logical()
    last_seen = time.monotonic()
    last_time = time.monotonic()
    fps = 0.0
    start = time.monotonic()

    source.open()
    try:
        with link:
            link.send(controller.counts(), force=True)
            while True:
                now = time.monotonic()
                obs = source.step(int((now - start) * 1000))
                if obs is None:
                    print("A camera stopped delivering frames.")
                    break
                dt = max(now - last_time, 1e-3)
                last_time = now
                fps = 0.9 * fps + 0.1 / dt if fps else 1.0 / dt

                if obs.forearm is not None:
                    last_seen = now
                    targets = mapper.angles(obs.forearm, obs.hand)
                elif now - last_seen > hold_s:
                    targets = controller.home_logical()     # lost for a while: ease home
                else:
                    targets = [None] * 4                    # brief dropout: hold

                if engaged:
                    link.send(controller.update(targets, dt))

                cv2.imshow(WINDOW, render(obs, engaged, link, fps, targets, controller))

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    engaged = not engaged
                    controller.reset_filters()
                elif key == ord("n") and obs.forearm is not None:
                    mapper.set_neutral(obs.forearm, obs.hand)
                    controller.reset_filters()
                elif key == ord("h"):
                    engaged = False
                    home = controller.home_logical()
                    for _ in range(100):                    # ease home at the slew limit
                        link.send(controller.update(home, 0.02), force=True)
                        time.sleep(0.02)
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
