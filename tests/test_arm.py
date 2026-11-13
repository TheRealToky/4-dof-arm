import pytest

from arm.joints import (JointController, OneEuroFilter, ServoConfig,
                        counts_to_deg, deg_to_counts)
from arm.link import READY_BANNER, ArmLink


def arduino_map(x, in_min, in_max, out_min, out_max):
    """Arduino's integer map(), truncating toward zero like C."""
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min)) + out_min


CONFIG = {
    "servos": [
        {"role": "a"},
        {"role": "b", "invert": True},
        {"role": "c", "offset_deg": 10, "min_deg": 30, "max_deg": 150},
        {"role": "d", "max_speed_dps": 90},
    ],
    "filter": {"min_cutoff_hz": 1000, "beta": 0},  # effectively no smoothing
}


# ---- conversions -----------------------------------------------------------

def test_endpoints_and_home():
    assert deg_to_counts(180) == 0
    assert deg_to_counts(0) == 1023
    assert deg_to_counts(90) in (511, 512)
    assert counts_to_deg(0) == 180
    assert counts_to_deg(1023) == 0


def test_round_trip_within_one_count():
    for counts in range(0, 1024):
        assert abs(deg_to_counts(counts_to_deg(counts)) - counts) <= 1


def test_matches_old_firmware_mapping_within_a_degree():
    for counts in range(0, 1024, 7):
        assert abs(counts_to_deg(counts) - arduino_map(counts, 0, 1023, 180, 0)) <= 1


def test_out_of_range_degrees_are_clamped():
    assert deg_to_counts(-20) == 1023
    assert deg_to_counts(400) == 0


# ---- per-servo transform ---------------------------------------------------

def test_invert_offset_limits():
    assert ServoConfig().logical_to_servo(120) == 120
    assert ServoConfig(invert=True).logical_to_servo(120) == 60
    assert ServoConfig(offset_deg=10).logical_to_servo(90) == 100
    assert ServoConfig(min_deg=30, max_deg=150).logical_to_servo(0) == 30
    assert ServoConfig(min_deg=30, max_deg=150).logical_to_servo(180) == 150


# ---- controller ------------------------------------------------------------

def test_controller_starts_at_home():
    jc = JointController(CONFIG)
    assert jc.servo_deg == [90, 90, 100, 90]


def test_slew_limit():
    jc = JointController(CONFIG)
    jc.update([90, 90, 90, 180], dt=0.1)      # servo 3: 90 dps -> 9 deg in 0.1 s
    assert jc.servo_deg[3] == pytest.approx(99, abs=0.2)


def test_reaches_target_and_respects_limits():
    jc = JointController(CONFIG)
    for _ in range(200):
        jc.update([120, 120, 180, 60], dt=0.02)
    assert jc.servo_deg[0] == pytest.approx(120, abs=0.5)
    assert jc.servo_deg[1] == pytest.approx(60, abs=0.5)    # inverted
    assert jc.servo_deg[2] == pytest.approx(150, abs=0.5)   # clamped
    assert jc.servo_deg[3] == pytest.approx(60, abs=0.5)


def test_none_or_nan_holds_joint():
    jc = JointController(CONFIG)
    before = list(jc.servo_deg)
    jc.update([None, float("nan"), None, None], dt=0.02)
    assert jc.servo_deg == before


def test_one_euro_reduces_noise():
    import random
    random.seed(0)
    f = OneEuroFilter(min_cutoff_hz=1.0, beta=0.0)
    raw = [90 + random.uniform(-3, 3) for _ in range(300)]
    out = [f(x, 1 / 30) for x in raw]
    spread = lambda xs: max(xs) - min(xs)
    # 1 Hz cutoff at 30 fps should cut white-noise amplitude to roughly a third.
    assert spread(out[100:]) < spread(raw[100:]) / 2


# ---- link ------------------------------------------------------------------

class FakeSerial:
    def __init__(self, lines=()):
        self.incoming = [l.encode() + b"\r\n" for l in lines]
        self.written = []
        self.closed = False

    def readline(self):
        return self.incoming.pop(0) if self.incoming else b""

    def write(self, data):
        self.written.append(data)

    def close(self):
        self.closed = True


def test_command_format_matches_firmware():
    assert ArmLink.format_command([512, 0, 1023, 7]) == "512,0,1023,7\n"


def test_dry_run_records_without_port():
    with ArmLink(dry_run=True, max_rate_hz=0) as link:
        assert link.send([1, 2, 3, 4]) is True
        assert link.send([1, 2, 3, 4]) is False     # unchanged
        assert link.send([1, 2, 3, 5]) is True
        assert link.sent_count == 2
        assert link.last_sent == [1, 2, 3, 5]


def test_rate_limit_and_force():
    with ArmLink(dry_run=True, max_rate_hz=1) as link:
        assert link.send([1, 2, 3, 4]) is True
        assert link.send([5, 6, 7, 8]) is False     # too soon
        assert link.send([5, 6, 7, 8], force=True) is True


def test_rejects_bad_values():
    with ArmLink(dry_run=True) as link:
        with pytest.raises(ValueError):
            link.send([1, 2, 3])
        with pytest.raises(ValueError):
            link.send([1, 2, 3, 1024])
        with pytest.raises(ValueError):
            link.send([-1, 2, 3, 4])


def test_waits_for_banner_then_writes():
    fake = FakeSerial(["", READY_BANNER])
    link = ArmLink(serial_obj=fake, ready_timeout_s=1.0)
    link.open()
    try:
        assert link.saw_banner
        link.send([512, 512, 512, 512], force=True)
        assert fake.written == [b"512,512,512,512\n"]
    finally:
        link.close()
    assert fake.closed


def test_counts_replies():
    import time
    fake = FakeSerial([READY_BANNER, "OK", "OK", "Invalid command"])
    link = ArmLink(serial_obj=fake, ready_timeout_s=1.0)
    link.open()
    try:
        deadline = time.monotonic() + 1.0
        while fake.incoming and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.05)
    finally:
        link.close()
    assert link.ok_count == 2
    assert link.error_count == 1
