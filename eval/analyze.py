"""Score a recorded session: does stereo track position better than one camera?

    python -m eval.analyze                      # newest session
    python -m eval.analyze eval/sessions/20261113_210000.csv

Writes eval/results/<session>/summary.csv plus accuracy.png, jitter.png
and layout.png, and prints the table.

How accuracy is scored: each method's average fingertip position per target
is fitted onto the measured target layout with the best rigid move (rotation
+ shift, no stretching), so you never have to know where the camera sits.
What is left over is the error, split into "depth" (along the camera's line
of sight) and "lateral" (across the picture).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from eval import (JOINTS, METHOD_LABELS, METHODS, POINTS, PROBE, RESULT_DIR,
                  SESSION_DIR)

MIN_FRAMES = 5      # a target needs this many valid frames to count for a method


# ---- geometry ---------------------------------------------------------------

def fit(est, truth, with_scale=False):
    """Best transform taking est (N,3) onto truth (N,3): truth ~ s * R @ est + t.

    Mirror images are allowed, so it does not matter which way round the
    target layout's axes were written down. Returns (s, R, t).
    """
    mu_e, mu_t = est.mean(axis=0), truth.mean(axis=0)
    e, g = est - mu_e, truth - mu_t
    u, sing, vt = np.linalg.svd(e.T @ g)
    r = vt.T @ u.T
    s = float(sing.sum() / (e ** 2).sum()) if with_scale else 1.0
    return s, r, mu_t - s * r @ mu_e


def residuals(est, truth, with_scale=False):
    """Per-target error after the best fit, expressed in the camera frame (N,3)."""
    s, r, t = fit(est, truth, with_scale)
    in_truth_frame = (s * (r @ est.T)).T + t - truth
    return (r.T @ in_truth_frame.T).T, s


# ---- loading ----------------------------------------------------------------

def load_session(csv_path):
    csv_path = Path(csv_path)
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    meta = json.loads(csv_path.with_suffix(".json").read_text(encoding="utf-8"))
    return rows, meta["targets_cm"]


def column(rows, name):
    return np.array([float(r[name]) if r.get(name, "") != "" else np.nan for r in rows])


def xyz(rows, method, point):
    return np.stack([column(rows, f"{method}_{point}_{a}") for a in "xyz"], axis=1)


# ---- metrics ----------------------------------------------------------------

def analyse(rows, targets_cm):
    """Returns {method: {metric: value}} plus per-method fitted layouts for plotting."""
    target_of = np.array([r["target"] for r in rows])
    names = [n for n in targets_cm if np.any(target_of == n)]
    truth_all = {n: np.array(targets_cm[n], dtype=float) / 100.0 for n in names}

    ms = {k: column(rows, k) for k in ("ms_track_left", "ms_track_right", "ms_disparity", "ms_mono")}
    latency = {
        "mono": ms["ms_track_left"] + ms["ms_mono"],
        "disparity": ms["ms_track_left"] + ms["ms_disparity"],
        "triangulate": ms["ms_track_left"] + ms["ms_track_right"],
    }

    out, layouts = {}, {}
    for m in METHODS:
        probe = xyz(rows, m, PROBE)
        valid = np.all(np.isfinite(probe), axis=1)
        res = {"frames": len(rows), "dropout_pct": 100.0 * (1.0 - valid.mean())}

        # Accuracy: mean probe position per target vs the measured layout.
        used, means = [], []
        for n in names:
            sel = (target_of == n) & valid
            if sel.sum() >= MIN_FRAMES:
                used.append(n)
                means.append(np.median(probe[sel], axis=0))
        res["targets_used"] = len(used)
        if len(used) >= 3:
            est, truth = np.array(means), np.array([truth_all[n] for n in used])
            err, _ = residuals(est, truth)
            res["error_mm"] = float(np.sqrt((err ** 2).sum(axis=1).mean()) * 1000)
            res["depth_error_mm"] = float(np.sqrt((err[:, 2] ** 2).mean()) * 1000)
            res["lateral_error_mm"] = float(np.sqrt((err[:, :2] ** 2).sum(axis=1).mean()) * 1000)
            err_s, scale = residuals(est, truth, with_scale=True)
            res["scale"] = scale
            res["error_after_scale_mm"] = float(np.sqrt((err_s ** 2).sum(axis=1).mean()) * 1000)
            # Simple cross-check that needs no fitting: distances between targets.
            i, j = np.triu_indices(len(used), 1)
            d_est = np.linalg.norm(est[i] - est[j], axis=1)
            d_true = np.linalg.norm(truth[i] - truth[j], axis=1)
            res["pair_distance_error_mm"] = float(np.abs(d_est - d_true).mean() * 1000)
            s, r, t = fit(est, truth)
            layouts[m] = {"names": used, "fitted": (r @ est.T).T + t, "truth": truth}

        # Jitter: spread while holding still, averaged over targets.
        for p in POINTS:
            pts = xyz(rows, m, p)
            ok = np.all(np.isfinite(pts), axis=1)
            total, depth = [], []
            for n in names:
                sel = (target_of == n) & ok
                if sel.sum() >= MIN_FRAMES:
                    sd = pts[sel].std(axis=0)
                    total.append(np.linalg.norm(sd))
                    depth.append(sd[2])
            if total:
                res[f"jitter_{p}_mm"] = float(np.mean(total) * 1000)
                res[f"jitter_depth_{p}_mm"] = float(np.mean(depth) * 1000)
        for j in JOINTS:
            a = column(rows, f"{m}_{j}")
            sds = [a[(target_of == n) & np.isfinite(a)].std() for n in names
                   if ((target_of == n) & np.isfinite(a)).sum() >= MIN_FRAMES]
            if sds:
                res[f"jitter_{j}_deg"] = float(np.mean(sds))
        joint_vals = [res[f"jitter_{j}_deg"] for j in JOINTS if f"jitter_{j}_deg" in res]
        if joint_vals:
            res["jitter_joints_deg"] = float(np.mean(joint_vals))

        res["latency_ms"] = float(np.nanmean(latency[m]))
        res["max_fps"] = 1000.0 / res["latency_ms"] if res["latency_ms"] > 0 else float("nan")
        out[m] = res
    return out, layouts


# Rows of the printed table: (key, label, lower is better?)
TABLE = [
    ("error_mm", "Position error, RMS (mm)", True),
    ("depth_error_mm", "  of which depth (mm)", True),
    ("lateral_error_mm", "  of which lateral (mm)", True),
    ("pair_distance_error_mm", "Target-to-target distance error (mm)", True),
    ("scale", "Size scale (1.00 = true size)", None),
    ("error_after_scale_mm", "Position error if rescaled (mm)", True),
    ("jitter_index_tip_mm", "Jitter, fingertip (mm)", True),
    ("jitter_hand_wrist_mm", "Jitter, wrist (mm)", True),
    ("jitter_middle_mcp_mm", "Jitter, knuckle (mm)", True),
    ("jitter_joints_deg", "Jitter, commanded joint angles (deg)", True),
    ("dropout_pct", "Frames with no reading (%)", True),
    ("targets_used", "Targets with enough readings", False),
    ("latency_ms", "Processing time per frame (ms)", True),
    ("max_fps", "Frame rate that allows (fps)", False),
]


def table_text(results):
    width = max(len(label) for _, label, _ in TABLE)
    lines = [" " * width + "".join(f"{m:>14}" for m in METHODS) + "   best"]
    for key, label, lower in TABLE:
        vals = [results[m].get(key, np.nan) for m in METHODS]
        cells = "".join(f"{v:14.2f}" if np.isfinite(v) else f"{'-':>14}" for v in vals)
        best = ""
        if lower is not None and np.any(np.isfinite(vals)):
            top = np.nanmin(vals) if lower else np.nanmax(vals)
            winners = [m for m, v in zip(METHODS, vals) if np.isfinite(v) and np.isclose(v, top)]
            best = winners[0] if len(winners) == 1 else "tie"
        lines.append(f"{label:<{width}}{cells}   {best}")
    return "\n".join(lines)


def write_summary(results, path):
    keys = sorted({k for r in results.values() for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + list(METHODS))
        for k in keys:
            w.writerow([k] + [f"{results[m][k]:.4f}" if k in results[m] else "" for m in METHODS])


# ---- plots ------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLOURS = {"mono": "#2a78d6", "disparity": "#eb6834", "triangulate": "#1baf7a"}


def _style(ax, ylabel):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, length=0, labelsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)


def _grouped_bars(ax, results, keys, labels, ylabel, title):
    width = 0.24
    for i, m in enumerate(METHODS):
        vals = [results[m].get(k, np.nan) for k in keys]
        xs = np.arange(len(keys)) + (i - 1) * (width + 0.03)
        ax.bar(xs, np.nan_to_num(vals), width, color=COLOURS[m], label=METHOD_LABELS[m])
        for x, v in zip(xs, vals):
            ax.text(x, (v if np.isfinite(v) else 0), f"{v:.1f}" if np.isfinite(v) else "n/a",
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(np.arange(len(keys)), labels)
    _style(ax, ylabel)
    ax.set_title(title, loc="left", color=INK, fontsize=11)


def make_plots(results, layouts, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def legend(fig):
        handles = [plt.Rectangle((0, 0), 1, 1, color=COLOURS[m]) for m in METHODS]
        fig.legend(handles, [METHOD_LABELS[m] for m in METHODS], loc="lower center",
                   ncol=3, frameon=False, fontsize=9, labelcolor=INK_MUTED)

    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    _grouped_bars(ax, results, ["error_mm", "lateral_error_mm", "depth_error_mm"],
                  ["Total", "Lateral", "Depth"], "RMS error (mm)",
                  "Fingertip position error against the measured targets (lower is better)")
    legend(fig)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / "accuracy.png", dpi=150)
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE,
                                 gridspec_kw={"width_ratios": [3, 4]})
    _grouped_bars(a1, results, [f"jitter_{p}_mm" for p in POINTS], ["Fingertip", "Wrist", "Knuckle"],
                  "Spread while holding still (mm)", "Position jitter (lower is better)")
    _grouped_bars(a2, results, [f"jitter_{j}_deg" for j in JOINTS],
                  ["Joint 0 pan", "Joint 0 tilt", "Joint 1 pan", "Joint 1 tilt"],
                  "Spread while holding still (deg)", "Jitter of the commanded joint angles")
    legend(fig)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / "jitter.png", dpi=150)
    plt.close(fig)

    if layouts:
        fig, ax = plt.subplots(figsize=(6.5, 6), facecolor=SURFACE)
        truth = {}
        for lay in layouts.values():
            truth.update(dict(zip(lay["names"], lay["truth"])))
        t = np.array(list(truth.values())) * 100
        ax.scatter(t[:, 0], t[:, 2], s=260, facecolors="none", edgecolors=INK_MUTED,
                   linewidths=1.2, label="Measured target")
        for name, p in truth.items():
            ax.annotate(name, (p[0] * 100, p[2] * 100), xytext=(11, 9), textcoords="offset points",
                        fontsize=8, color=INK_MUTED)
        for m in METHODS:
            if m in layouts:
                f = layouts[m]["fitted"] * 100
                g = layouts[m]["truth"] * 100
                for a, b in zip(f, g):
                    ax.plot([a[0], b[0]], [a[2], b[2]], color=COLOURS[m], linewidth=1, alpha=0.6)
                ax.scatter(f[:, 0], f[:, 2], s=42, color=COLOURS[m], edgecolors=SURFACE,
                           linewidths=1.5, label=METHOD_LABELS[m], zorder=3)
        ax.set_aspect("equal", adjustable="datalim")
        _style(ax, "z, away from you (cm)")
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_xlabel("x, across the table (cm)", color=INK_MUTED, fontsize=9)
        ax.set_title("Where each method put the targets (top view, after best fit)",
                     loc="left", color=INK, fontsize=11)
        fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=INK_MUTED)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        fig.savefig(out_dir / "layout.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", nargs="?", help="session CSV (default: newest)")
    args = parser.parse_args()

    if args.session:
        path = Path(args.session)
    else:
        sessions = sorted(SESSION_DIR.glob("*.csv"))
        if not sessions:
            raise SystemExit(f"No sessions in {SESSION_DIR}. Run: python -m eval.record")
        path = sessions[-1]

    rows, targets = load_session(path)
    if not rows:
        raise SystemExit(f"{path} has no frames.")
    results, layouts = analyse(rows, targets)

    out_dir = RESULT_DIR / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    write_summary(results, out_dir / "summary.csv")
    make_plots(results, layouts, out_dir)

    print(f"Session {path.name}: {len(rows)} frames, {len({r['target'] for r in rows})} targets\n")
    print(table_text(results))
    if len(layouts) < len(METHODS):
        missing = [m for m in METHODS if m not in layouts]
        print(f"\nNo accuracy score for {', '.join(missing)}: fewer than 3 targets had enough readings.")
    print(f"\nWrote summary.csv, accuracy.png, jitter.png, layout.png to {out_dir}")


if __name__ == "__main__":
    main()
