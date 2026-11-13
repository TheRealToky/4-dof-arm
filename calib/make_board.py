"""Writes a printable A4 ChArUco board to calib/charuco_board.pdf.

    python -m calib.make_board

Print it at 100 % ("actual size", no "fit to page"), glue it to something
flat and rigid, then measure one black square with a ruler. If it is not
the size given in config.json ("board" -> "square_mm"), put the measured
value there and scale "marker_mm" by the same ratio.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arm.joints import load_config
from calib.board import board_config, make_board

A4_MM = (297.0, 210.0)      # landscape
PX_PER_MM = 12              # ~300 dpi


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="path to config.json")
    parser.add_argument("--out", default=str(Path(__file__).parent / "charuco_board.pdf"))
    args = parser.parse_args()

    cfg = board_config(load_config(args.config))
    width_mm = cfg["squares_x"] * cfg["square_mm"]
    height_mm = cfg["squares_y"] * cfg["square_mm"]
    if width_mm > A4_MM[0] - 20 or height_mm > A4_MM[1] - 20:
        raise SystemExit(f"Board is {width_mm:.0f} x {height_mm:.0f} mm, too big for A4 with margins.")

    image = make_board(cfg).generateImage(
        (round(width_mm * PX_PER_MM), round(height_mm * PX_PER_MM)), marginSize=0)

    fig = plt.figure(figsize=(A4_MM[0] / 25.4, A4_MM[1] / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, A4_MM[0])
    ax.set_ylim(A4_MM[1], 0)
    ax.axis("off")
    x0 = (A4_MM[0] - width_mm) / 2
    y0 = (A4_MM[1] - height_mm) / 2
    ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest",
              extent=(x0, x0 + width_mm, y0 + height_mm, y0))
    ax.text(A4_MM[0] / 2, y0 + height_mm + 6,
            f"ChArUco {cfg['squares_x']}x{cfg['squares_y']}  square {cfg['square_mm']:g} mm  "
            f"marker {cfg['marker_mm']:g} mm  {cfg['dictionary']}  - print at 100 %",
            ha="center", va="center", fontsize=8)
    fig.savefig(args.out)
    print(f"Wrote {args.out}  (board {width_mm:.0f} x {height_mm:.0f} mm)")


if __name__ == "__main__":
    main()
