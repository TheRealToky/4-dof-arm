"""Drive the arm from your forearm and hand, seen by a camera.

    python vision_control.py --no-arm          # tracking + overlay only
    python vision_control.py                   # with the arm (port auto-detected)
    python vision_control.py --port COM3 --camera 1

Keys:  SPACE  engage / pause arm output (starts paused)
       N      set the current pose as neutral (= 90 deg on every servo)
       H      send the arm home and pause
       Q/ESC  quit
"""

import argparse
import time

import cv2

from arm.joints import JointController, load_config
from arm.link import ArmLink
from vision.cameras import Camera
from vision.mapping import ArmMapper
from vision.tracker import ArmTracker, draw

WINDOW = "4-DOF arm - vision control"
JOINT_LABELS = ["J0 pan", "J0 tilt", "J1 pan", "J1 tilt"]


def put_text(frame, text, row, colour=(255, 255, 255)):
    pos = (10, 24 + row * 22)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["mono", "stereo"], default="mono")
    parser.add_argument("--no-arm", action="store_true", help="do not open the serial port")
    parser.add_argument("--port", help="serial port, e.g. COM3")
    parser.add_argument("--camera", type=int, help="camera index (overrides config.json)")
    parser.add_argument("--config", help="path to config.json")
    args = parser.parse_args()

    if args.mode == "stereo":
        parser.error("stereo mode is not implemented yet (needs the calibration stage first)")

    config = load_config(args.config)
    serial_cfg = config.get("serial", {})
    cam_cfg = config.get("cameras", {})
    map_cfg = config.get("mapping", {})
    hold_s = map_cfg.get("hold_s", 1.0)

    controller = JointController(config)
    mapper = ArmMapper(map_cfg.get("gains", [1, 1, 1, 1]))
    tracker = ArmTracker(map_cfg.get("arm", "right"))
    camera = Camera(args.camera if args.camera is not None else cam_cfg.get("mono", 0),
                    cam_cfg.get("width", 640), cam_cfg.get("height", 480))
    link = ArmLink(port=args.port or serial_cfg.get("port"),
                   baud=serial_cfg.get("baud", 115200), dry_run=args.no_arm)

    engaged = False
    targets = controller.home_logical()
    last_seen = time.monotonic()
    last_time = time.monotonic()
    fps = 0.0
    start = time.monotonic()

    with camera, link:
        link.send(controller.counts(), force=True)
        try:
            while True:
                frame = camera.read()
                if frame is None:
                    print("Camera stopped delivering frames.")
                    break

                now = time.monotonic()
                dt = max(now - last_time, 1e-3)
                last_time = now
                fps = 0.9 * fps + 0.1 / dt if fps else 1.0 / dt

                result = tracker.process(frame, int((now - start) * 1000))

                if result.arm_ok:
                    last_seen = now
                    targets = mapper.angles(result.forearm, result.hand if result.hand_ok else None)
                elif now - last_seen > hold_s:
                    targets = controller.home_logical()     # lost for a while: ease home
                else:
                    targets = [None] * 4                    # brief dropout: hold

                if engaged:
                    link.send(controller.update(targets, dt))

                # ---- overlay (mirrored, so it reads like a mirror) ----
                draw(frame, result)
                view = cv2.flip(frame, 1)
                state = "ENGAGED" if engaged else "PAUSED (space to engage)"
                if link.dry_run:
                    state += "  [no arm]"
                put_text(view, f"{state}   {fps:4.1f} fps", 0,
                         (0, 255, 0) if engaged else (0, 200, 255))
                status = ("arm+hand" if result.hand_ok else "arm only") if result.arm_ok else "NOT TRACKED"
                put_text(view, f"tracking: {status}", 1,
                         (255, 255, 255) if result.arm_ok else (0, 0, 255))
                for i, label in enumerate(JOINT_LABELS):
                    target = "  --" if targets[i] is None else f"{targets[i]:5.1f}"
                    put_text(view, f"{label}: target {target}   servo {controller.servo_deg[i]:5.1f}", 2 + i)
                if not link.dry_run:
                    put_text(view, f"sent {link.sent_count}  ok {link.ok_count}  err {link.error_count}", 6)
                cv2.imshow(WINDOW, view)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    engaged = not engaged
                    controller.reset_filters()
                elif key == ord("n") and result.arm_ok:
                    mapper.set_neutral(result.forearm, result.hand if result.hand_ok else None)
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
            tracker.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
