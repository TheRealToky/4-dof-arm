"""Serial link to the arm firmware (4_servo_arm.ino).

Protocol: one ASCII line per update, four integers 0-1023 separated by
commas, e.g. "512,512,512,512\n". The firmware answers "OK" or
"Invalid command" to every line and prints a banner once after reset.
"""

from __future__ import annotations

import threading
import time

from .joints import COUNT_MAX, COUNT_MIN, SERVO_COUNT

READY_BANNER = "4-servo controller ready!"
DEFAULT_BAUD = 115200


def find_port() -> str:
    """Pick the serial port that looks most like an Arduino."""
    from serial.tools import list_ports

    ports = list(list_ports.comports())
    if not ports:
        raise RuntimeError("No serial ports found. Is the arm plugged in?")
    keywords = ("arduino", "ch340", "wch", "cp210", "ftdi", "usb serial", "usb-serial")
    for p in ports:
        text = f"{p.description} {p.manufacturer or ''}".lower()
        if any(k in text for k in keywords):
            return p.device
    if len(ports) == 1:
        return ports[0].device
    names = ", ".join(f"{p.device} ({p.description})" for p in ports)
    raise RuntimeError(f"Several serial ports found, pass one explicitly: {names}")


class ArmLink:
    """Connection to the arm. Use as a context manager.

    dry_run=True opens nothing and just records what would be sent, so the
    rest of the pipeline can be exercised without the hardware.
    """

    def __init__(self, port: str | None = None, baud: int = DEFAULT_BAUD,
                 dry_run: bool = False, max_rate_hz: float = 50.0,
                 ready_timeout_s: float = 5.0, serial_obj=None):
        self.port = port
        self.baud = baud
        self.dry_run = dry_run
        self.min_interval = 1.0 / max_rate_hz if max_rate_hz > 0 else 0.0
        self.ready_timeout_s = ready_timeout_s

        self._serial = serial_obj      # injectable for tests
        self._reader = None
        self._stop = threading.Event()
        self._last_send_time = 0.0

        self.last_sent: list[int] | None = None
        self.sent_count = 0
        self.ok_count = 0
        self.error_count = 0
        self.saw_banner = False

    # ---- lifecycle -------------------------------------------------------

    def open(self) -> "ArmLink":
        if self.dry_run:
            return self
        if self._serial is None:
            import serial

            if not self.port:
                self.port = find_port()
            self._serial = serial.Serial(self.port, self.baud, timeout=0.05)
        self._wait_for_banner()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        return self

    def close(self):
        self._stop.set()
        if self._reader is not None:
            self._reader.join(timeout=1.0)
            self._reader = None
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # ---- sending ---------------------------------------------------------

    @staticmethod
    def format_command(counts) -> str:
        return ",".join(map(str, counts)) + "\n"

    def send(self, counts, force: bool = False) -> bool:
        """Send four wire values. Returns True if a line actually went out.

        Skipped when nothing changed or when called faster than max_rate_hz
        (the caller's next call carries the newer value). force=True
        bypasses both checks.
        """
        counts = [int(c) for c in counts]
        if len(counts) != SERVO_COUNT:
            raise ValueError(f"expected {SERVO_COUNT} values, got {len(counts)}")
        if any(c < COUNT_MIN or c > COUNT_MAX for c in counts):
            raise ValueError(f"values must be {COUNT_MIN}-{COUNT_MAX}: {counts}")

        now = time.monotonic()
        if not force:
            if counts == self.last_sent:
                return False
            if now - self._last_send_time < self.min_interval:
                return False

        if not self.dry_run:
            self._serial.write(self.format_command(counts).encode("ascii"))
        self.last_sent = counts
        self._last_send_time = now
        self.sent_count += 1
        return True

    # ---- receiving -------------------------------------------------------

    def _wait_for_banner(self):
        """The board resets when the port opens; wait until it is listening."""
        deadline = time.monotonic() + self.ready_timeout_s
        while time.monotonic() < deadline:
            line = self._readline()
            if line == READY_BANNER:
                self.saw_banner = True
                return
        # Boards without auto-reset never print the banner; carry on.
        print(f"Warning: no '{READY_BANNER}' within {self.ready_timeout_s:.0f} s, continuing anyway.")

    def _readline(self) -> str:
        raw = self._serial.readline()
        return raw.decode("ascii", errors="replace").strip() if raw else ""

    def _read_loop(self):
        while not self._stop.is_set():
            try:
                line = self._readline()
            except Exception:
                break  # port closed or unplugged
            if line == "OK":
                self.ok_count += 1
            elif line == READY_BANNER:
                self.saw_banner = True
            elif line:
                self.error_count += 1
