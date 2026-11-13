import math

import numpy as np
import pytest

from vision.mapping import ArmMapper, rotation_between, unit

UP = np.array([0.0, -1.0, 0.0])


def rot(axis, deg):
    """Rotation matrix about a camera axis."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


def lean(pan_deg=0.0, tilt_deg=0.0):
    """Forearm that started straight up, leaned sideways (pan) and/or away (tilt)."""
    v = np.array([math.tan(math.radians(pan_deg)), -1.0, math.tan(math.radians(tilt_deg))])
    return v / np.linalg.norm(v)


def test_rotation_between():
    a, b = unit([1, 2, 3]), unit([-2, 0.5, 1])
    r = rotation_between(a, b)
    assert np.allclose(r @ a, b)
    assert np.allclose(r @ r.T, np.eye(3))


def test_default_neutral_is_forearm_up():
    m = ArmMapper()
    assert m.angles(UP, UP) == pytest.approx([90, 90, 90, 90])


def test_forearm_lean_drives_joint0_only():
    m = ArmMapper()
    f = lean(pan_deg=30)                         # towards image right
    out = m.angles(f, f)                         # hand in line with forearm
    assert out == pytest.approx([120, 90, 90, 90], abs=1e-6)

    f = lean(tilt_deg=-25)                       # towards the camera
    out = m.angles(f, f)
    assert out == pytest.approx([90, 65, 90, 90], abs=1e-6)


def test_vector_length_does_not_matter():
    m = ArmMapper()
    f = lean(pan_deg=20, tilt_deg=10)
    assert m.angles(f * 0.25, f * 7) == pytest.approx(m.angles(f, f))


def test_hand_bend_drives_joint1_only():
    m = ArmMapper()
    hand = rot("z", 20) @ UP                     # bent 20 deg towards image right
    out = m.angles(UP, hand)
    assert out[:2] == pytest.approx([90, 90])
    assert out[2] == pytest.approx(110, abs=1e-6)
    assert out[3] == pytest.approx(90, abs=1e-6)


def test_hand_bend_is_relative_to_forearm():
    """Same wrist bend must give the same joint 1 angles wherever the forearm points."""
    m = ArmMapper()
    bent = rot("z", 20) @ UP
    for f in (lean(pan_deg=35), lean(tilt_deg=-30), lean(pan_deg=-20, tilt_deg=25)):
        carry = rotation_between(UP, f)
        out = m.angles(f, carry @ bent)
        assert out[2] == pytest.approx(110, abs=1e-6)
        assert out[3] == pytest.approx(90, abs=1e-6)


def test_custom_neutral():
    m = ArmMapper()
    f0 = lean(pan_deg=15, tilt_deg=-10)
    h0 = rot("x", 12) @ f0
    m.set_neutral(f0, h0)
    assert m.angles(f0, h0) == pytest.approx([90, 90, 90, 90], abs=1e-6)


def test_gain_and_clamp():
    m = ArmMapper(gains=(2, 1, 1, 1))
    assert m.angles(lean(pan_deg=20), None)[0] == pytest.approx(130, abs=1e-6)
    m = ArmMapper(gains=(10, 1, 1, 1))
    assert m.angles(lean(pan_deg=20), None)[0] == 180


def test_missing_inputs_give_none():
    m = ArmMapper()
    assert m.angles(None, None) == [None] * 4
    out = m.angles(UP, None)
    assert out[:2] == pytest.approx([90, 90]) and out[2:] == [None, None]
    assert m.angles(np.zeros(3), UP) == [None] * 4
