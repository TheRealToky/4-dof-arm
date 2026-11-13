import serial
import time
import pygame

# ---------------- CONFIGURATION ----------------

PORT = "COM3"
BAUD_RATE = 115200

MIN_VALUE = 0
MAX_VALUE = 1023
START_VALUE = 512

STEP = 5
UPDATE_INTERVAL = 15

SERVO_KEYS = [
    (pygame.K_q, pygame.K_a),  # Servo 1
    (pygame.K_w, pygame.K_s),  # Servo 2
    (pygame.K_e, pygame.K_d),  # Servo 3
    (pygame.K_r, pygame.K_f)   # Servo 4
]

# ---------------- CONNECTION ----------------

arduino = serial.Serial(PORT, BAUD_RATE, timeout=0.02)
time.sleep(2)

pygame.init()

screen = pygame.display.set_mode((550, 400))
pygame.display.set_caption("4-DOF Robotic Arm Controller")

font = pygame.font.SysFont("Arial", 23)
small_font = pygame.font.SysFont("Arial", 17)

clock = pygame.time.Clock()

# Four virtual potentiometers
pot_values = [START_VALUE] * 4

last_update = 0
running = True

# ---------------- FUNCTIONS ----------------

def send_values():
    command = ",".join(map(str, pot_values)) + "\n"
    arduino.write(command.encode())


def draw_interface():
    screen.fill((25, 30, 40))

    title = font.render(
        "4-DOF ROBOTIC ARM CONTROLLER",
        True, (255, 255, 255)
    )

    screen.blit(title, (35, 25))

    for i, value in enumerate(pot_values):

        angle = round(180 - (value * 180 / 1023))

        text = font.render(
            f"Servo {i + 1}: {value} | {angle} deg",
            True, (100, 200, 255)
        )

        screen.blit(text, (50, 90 + i * 50))

    instructions = [
        "Q/A: Servo 1",
        "W/S: Servo 2",
        "E/D: Servo 3",
        "R/F: Servo 4",
        "0: Reset all servos",
        "ESC: Exit"
    ]

    for i, instruction in enumerate(instructions):
        text = small_font.render(
            instruction, True, (200, 200, 200)
        )

        screen.blit(text, (50, 300 + i * 17))

    pygame.display.flip()


# ---------------- MAIN LOOP ----------------

send_values()

try:
    while running:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False

                if event.key == pygame.K_0:
                    pot_values = [START_VALUE] * 4
                    send_values()

        keys = pygame.key.get_pressed()

        now = pygame.time.get_ticks()

        if now - last_update >= UPDATE_INTERVAL:

            old_values = pot_values.copy()

            for i, (increase_key, decrease_key) in enumerate(SERVO_KEYS):

                if keys[increase_key]:
                    pot_values[i] += STEP

                if keys[decrease_key]:
                    pot_values[i] -= STEP

                pot_values[i] = max(
                    MIN_VALUE,
                    min(MAX_VALUE, pot_values[i])
                )

            if pot_values != old_values:
                send_values()

            last_update = now

        draw_interface()

        clock.tick(60)

finally:
    arduino.close()
    pygame.quit()
    print("Controller disconnected.")
