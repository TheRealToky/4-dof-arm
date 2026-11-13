"""Capture image pairs of the ChArUco board for stereo calibration.

    python -m calib.capture_pairs

Hold the board so both cameras see it, press SPACE to save a pair. Aim for
about 30 pairs: near and far, every corner of the image, and tilted in
different directions (not just flat-on). Keep the board still when saving.

Keys:  SPACE  save pair (only when both views share enough corners)
       Q/ESC  quit
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from arm.joints import load_config
from calib.board import BoardDetector, board_config
from vision.cameras import StereoCapture

PAIR_DIR = Path(__file__).parent / "pairs"
MIN_COMMON = 8
WINDOW = "Stereo calibration capture (left | right)"


def annotate(frame, corners, label):
    view = frame.copy()
    for x, y in corners:
        cv2.circle(view, (int(x), int(y)), 4, (0, 255, 0), -1)
    cv2.putText(view, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(view, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return view


def side_by_side(left, right):
    h = min(left.shape[0], right.shape[0])
    def fit(img):
        return img if img.shape[0] == h else cv2.resize(img, (round(img.shape[1] * h / img.shape[0]), h))
    return np.hstack([fit(left), fit(right)])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="path to config.json")
    parser.add_argument("--out", default=str(PAIR_DIR))
    args = parser.parse_args()

    config = load_config(args.config)
    cam = config.get("cameras", {})
    detector = BoardDetector(board_config(config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    count = len(list(out.glob("left_*.png")))

    with StereoCapture(cam.get("left", 0), cam.get("right", 1),
                       cam.get("width", 640), cam.get("height", 480)) as stereo:
        print(f"Left camera  {stereo.left.width}x{stereo.left.height}")
        print(f"Right camera {stereo.right.width}x{stereo.right.height}")
        while True:
            left, right = stereo.read()
            if left is None:
                print("A camera stopped delivering frames.")
                break
            cl, il = detector.detect(left)
            cr, ir = detector.detect(right)
            common = len(np.intersect1d(il, ir))
            ready = common >= MIN_COMMON

            view = side_by_side(annotate(left, cl, f"LEFT  {len(il)} corners"),
                                annotate(right, cr, f"RIGHT {len(ir)} corners"))
            status = f"saved {count}   shared corners {common}   skew {stereo.skew_ms:.0f} ms"
            status += "   SPACE to save" if ready else f"   need {MIN_COMMON} shared"
            cv2.putText(view, status, (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0) if ready else (0, 0, 255), 2)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" ") and ready:
                cv2.imwrite(str(out / f"left_{count:03d}.png"), left)
                cv2.imwrite(str(out / f"right_{count:03d}.png"), right)
                count += 1
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()
    print(f"{count} pairs in {out}")


if __name__ == "__main__":
    main()
