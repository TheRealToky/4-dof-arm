"""Joint maths: wire counts <-> degrees, per-servo limits, smoothing, slew limit.

Two kinds of angle are used:

* servo angle  - what the physical servo is told (0-180 deg), after the
                 per-servo invert / offset / limits from config.json.
* logical angle - what the control code thinks in: 0-180 deg with 90 as
                 neutral, same sign convention for every joint.

The wire protocol carries "pot counts" (0-1023), which the firmware maps
inverted onto the servo range: 0 -> 180 deg, 1023 -> 0 deg.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

SERVO_COUNT = 4
COUNT_MIN = 0
COUNT_MAX = 1023
COUNT_HOME = 512
SERVO_RANGE_DEG = 180.0

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


def clamp(value, low, high):
    return max(low, min(high, value))


def deg_to_counts(servo_deg: float) -> int:
    """Servo angle in degrees -> wire value (inverted, as the firmware expects)."""
    counts = round((SERVO_RANGE_DEG - servo_deg) * COUNT_MAX / SERVO_RANGE_DEG)
    return int(clamp(counts, COUNT_MIN, COUNT_MAX))


def counts_to_deg(counts: int) -> float:
    """Wire value -> servo angle in degrees."""
    return SERVO_RANGE_DEG - counts * SERVO_RANGE_DEG / COUNT_MAX


@dataclass
class ServoConfig:
    role: str = ""
    invert: bool = False
    offset_deg: float = 0.0
    min_deg: float = 0.0
    max_deg: float = 180.0
    home_deg: float = 90.0
    max_speed_dps: float = 180.0

    def logical_to_servo(self, logical_deg: float) -> float:
        """Apply direction, offset and soft limits."""
        delta = logical_deg - 90.0
        if self.invert:
            delta = -delta
        return clamp(90.0 + delta + self.offset_deg, self.min_deg, self.max_deg)


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def servo_configs(config: dict) -> list[ServoConfig]:
    servos = [ServoConfig(**entry) for entry in config.get("servos", [])]
    if len(servos) != SERVO_COUNT:
        raise ValueError(f"config must list exactly {SERVO_COUNT} servos, found {len(servos)}")
    return servos


class OneEuroFilter:
    """One-Euro filter: heavy smoothing when still, light smoothing when moving.

    min_cutoff_hz - lower = smoother but laggier at rest.
    beta          - higher = less lag during fast motion.
    """

    def __init__(self, min_cutoff_hz: float = 1.5, beta: float = 0.02, d_cutoff_hz: float = 1.0):
        self.min_cutoff = min_cutoff_hz
        self.beta = beta
        self.d_cutoff = d_cutoff_hz
        self._x = None
        self._dx = 0.0

    @staticmethod
    def _alpha(cutoff_hz: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff_hz)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self._x = None
        self._dx = 0.0

    def __call__(self, x: float, dt: float) -> float:
        if self._x is None or dt <= 0:
            self._x = x
            return x
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        return self._x


class JointController:
    """Turns noisy logical target angles into safe wire counts.

    Per joint: One-Euro smoothing -> invert/offset/limits -> slew limit.
    """

    def __init__(self, config: dict | None = None):
        config = config if config is not None else load_config()
        self.servos = servo_configs(config)
        f = config.get("filter", {})
        self._filters = [
            OneEuroFilter(f.get("min_cutoff_hz", 1.5), f.get("beta", 0.02))
            for _ in self.servos
        ]
        # Current commanded servo angles; start at home.
        self.servo_deg = [s.logical_to_servo(s.home_deg) for s in self.servos]

    def home_logical(self) -> list[float]:
        return [s.home_deg for s in self.servos]

    def reset_filters(self):
        for f in self._filters:
            f.reset()

    def update(self, logical_targets_deg, dt: float) -> list[int]:
        """Advance one time step towards the targets; returns the 4 wire counts."""
        if len(logical_targets_deg) != len(self.servos):
            raise ValueError("expected one target per servo")
        for i, (servo, target) in enumerate(zip(self.servos, logical_targets_deg)):
            if target is None or not math.isfinite(target):
                continue  # hold this joint
            smoothed = self._filters[i](float(target), dt)
            goal = servo.logical_to_servo(smoothed)
            max_step = servo.max_speed_dps * max(dt, 0.0)
            step = clamp(goal - self.servo_deg[i], -max_step, max_step)
            self.servo_deg[i] += step
        return self.counts()

    def counts(self) -> list[int]:
        return [deg_to_counts(d) for d in self.servo_deg]
