"""Keyboard jog control for the 4-DOF arm.

Drives the raw servo values directly (no invert / offset / limits from
config.json), so it is also the tool for working out which servo is which
axis and what its safe range is.

    python keyboard_control.py              # port from config.json or auto-detect
    python keyboard_control.py --port COM3
    python keyboard_control.py --no-arm     # GUI only, nothing sent
"""

import argparse

import pygame

from arm.joints import (COUNT_HOME, COUNT_MAX, COUNT_MIN, SERVO_COUNT, clamp,
                        counts_to_deg, load_config)
from arm.link import ArmLink

STEP = 5                # counts per update
UPDATE_INTERVAL_MS = 15

SERVO_KEYS = [
    (pygame.K_q, pygame.K_a),  # Servo 1
    (pygame.K_w, pygame.K_s),  # Servo 2
    (pygame.K_e, pygame.K_d),  # Servo 3
    (pygame.K_r, pygame.K_f),  # Servo 4
]

BACKGROUND = (25, 30, 40)
WHITE = (255, 255, 255)
BLUE = (100, 200, 255)
GREY = (200, 200, 200)
RED = (255, 120, 120)


def draw_interface(screen, font, small_font, values, link, roles):
    screen.fill(BACKGROUND)
    screen.blit(font.render("4-DOF ROBOTIC ARM CONTROLLER", True, WHITE), (35, 25))

    for i, value in enumerate(values):
        angle = round(counts_to_deg(value))
        text = font.render(f"Servo {i + 1}: {value} | {angle} deg", True, BLUE)
        screen.blit(text, (50, 90 + i * 50))
        screen.blit(small_font.render(roles[i], True, GREY), (400, 95 + i * 50))

    instructions = [
        "Q/A: Servo 1", "W/S: Servo 2", "E/D: Servo 3", "R/F: Servo 4",
        "0: Reset all servos", "ESC: Exit",
    ]
    for i, instruction in enumerate(instructions):
        screen.blit(small_font.render(instruction, True, GREY), (50, 300 + i * 17))

    if link.dry_run:
        status, colour = "NO ARM (dry run)", RED
    else:
        status = f"{link.port}  sent {link.sent_count}  ok {link.ok_count}  err {link.error_count}"
        colour = RED if link.error_count else GREY
    screen.blit(small_font.render(status, True, colour), (250, 300))

    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", help="serial port, e.g. COM3")
    parser.add_argument("--no-arm", action="store_true", help="do not open the serial port")
    parser.add_argument("--config", help="path to config.json")
    args = parser.parse_args()

    config = load_config(args.config)
    serial_cfg = config.get("serial", {})
    roles = [s.get("role", "") for s in config.get("servos", [])] + [""] * SERVO_COUNT

    link = ArmLink(port=args.port or serial_cfg.get("port"),
                   baud=serial_cfg.get("baud", 115200),
                   dry_run=args.no_arm)

    with link:
        pygame.init()
        try:
            screen = pygame.display.set_mode((550, 420))
            pygame.display.set_caption("4-DOF Robotic Arm Controller")
            font = pygame.font.SysFont("Arial", 23)
            small_font = pygame.font.SysFont("Arial", 17)
            clock = pygame.time.Clock()

            values = [COUNT_HOME] * SERVO_COUNT
            link.send(values, force=True)

            last_update = 0
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_0:
                            values = [COUNT_HOME] * SERVO_COUNT
                            link.send(values, force=True)

                keys = pygame.key.get_pressed()
                now = pygame.time.get_ticks()
                if now - last_update >= UPDATE_INTERVAL_MS:
                    for i, (increase_key, decrease_key) in enumerate(SERVO_KEYS):
                        if keys[increase_key]:
                            values[i] += STEP
                        if keys[decrease_key]:
                            values[i] -= STEP
                        values[i] = clamp(values[i], COUNT_MIN, COUNT_MAX)
                    last_update = now

                # send() itself skips unchanged values and rate-limits.
                link.send(values)

                draw_interface(screen, font, small_font, values, link, roles)
                clock.tick(60)
        finally:
            pygame.quit()

    print("Controller disconnected.")


if __name__ == "__main__":
    main()
