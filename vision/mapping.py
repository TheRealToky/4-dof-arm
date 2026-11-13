"""Turns a forearm direction and a hand direction into the four joint angles.

Input vectors are in camera axes (x right, y down, z away from the camera)
and can come from any source: MediaPipe's own 3D estimate (mono) or metric
points from the stereo rig. Only their direction matters.

Everything is measured relative to a neutral pose, which maps to 90 deg on
all four servos:

    joint 0 pan / tilt  - how far the forearm has swung from its neutral
                          direction, sideways / front-back.
    joint 1 pan / tilt  - how far the hand is bent away from the forearm,
                          compared with the neutral pose.

Forearm roll (turning the palm over) is not tracked, so the hand's two axes
are referenced to the camera, not to the palm.
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_FOREARM = np.array([0.0, -1.0, 0.0])   # forearm pointing straight up


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def rotation_between(a, b):
    """Smallest rotation matrix taking unit vector a onto unit vector b."""
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.9999:
        return np.eye(3)    # opposite directions: far outside the servo range anyway
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1.0 + c)


class ArmMapper:
    def __init__(self, gains=(1.0, 1.0, 1.0, 1.0)):
        self.gains = list(gains)
        self.set_neutral(DEFAULT_FOREARM, None)

    def set_neutral(self, forearm, hand=None):
        """Record the pose that should correspond to 90 deg everywhere."""
        n = unit(forearm)
        if n is None:
            return False
        # Sideways axis: camera x made perpendicular to the forearm (camera z
        # if the forearm happens to point along x).
        u = None
        for ref in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])):
            cand = ref - np.dot(ref, n) * n
            if np.linalg.norm(cand) > 0.3:
                u = unit(cand)
                break
        self._n0, self._u0, self._w0 = n, u, np.cross(n, u)
        self._hand0 = (0.0, 0.0)
        if hand is not None:
            local = self._hand_local(n, hand)
            if local is not None:
                self._hand0 = local
        return True

    def _hand_local(self, forearm_unit, hand):
        """Hand bend (pan, tilt) in degrees, in a frame carried along with the forearm."""
        h = unit(hand)
        if h is None:
            return None
        r = rotation_between(self._n0, forearm_unit)
        along = float(np.dot(h, forearm_unit))
        pan = math.degrees(math.atan2(float(np.dot(h, r @ self._u0)), along))
        tilt = math.degrees(math.atan2(float(np.dot(h, r @ self._w0)), along))
        return pan, tilt

    def angles(self, forearm, hand=None):
        """Returns [j0_pan, j0_tilt, j1_pan, j1_tilt] as logical servo degrees.

        An entry is None when it cannot be computed; the caller holds that joint.
        """
        out = [None, None, None, None]
        f = unit(forearm) if forearm is not None else None
        if f is None:
            return out
        along = float(np.dot(f, self._n0))
        raw = [
            math.degrees(math.atan2(float(np.dot(f, self._u0)), along)),
            math.degrees(math.atan2(float(np.dot(f, self._w0)), along)),
            None, None,
        ]
        if hand is not None:
            local = self._hand_local(f, hand)
            if local is not None:
                raw[2] = local[0] - self._hand0[0]
                raw[3] = local[1] - self._hand0[1]
        for i, deg in enumerate(raw):
            if deg is not None:
                out[i] = max(0.0, min(180.0, 90.0 + self.gains[i] * deg))
        return out
