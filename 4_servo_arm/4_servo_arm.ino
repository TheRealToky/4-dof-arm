
#include <Servo.h>

// Protocol: one ASCII line per update, four integers 0-1023 separated by
// commas, e.g. "512,512,512,512\n". Replies "OK" or "Invalid command".
// Value 0 -> 180 deg, 1023 -> 0 deg.

const byte SERVO_COUNT = 4;
const byte LINE_MAX = 32;   // "1023,1023,1023,1023" is 19 chars

Servo servos[SERVO_COUNT];

const byte servoPins[SERVO_COUNT] = {
  9, 10, 11, 6
};

int potValues[SERVO_COUNT] = {
  512, 512, 512, 512
};

char lineBuffer[LINE_MAX];
byte lineLength = 0;
bool lineOverflow = false;

void setup() {
  Serial.begin(115200);

  for (int i = 0; i < SERVO_COUNT; i++) {
    servos[i].attach(servoPins[i]);
    updateServo(i);
  }

  Serial.println("4-servo controller ready!");
}

void loop() {
  // Read everything that has arrived. If several complete lines are
  // waiting, only the newest valid one is applied, so the arm never
  // falls behind the host.
  bool haveNew = false;

  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n') {
      int values[SERVO_COUNT];

      if (!lineOverflow && parseLine(values)) {
        for (int i = 0; i < SERVO_COUNT; i++) {
          potValues[i] = values[i];
        }
        haveNew = true;
        Serial.println("OK");
      } else {
        Serial.println("Invalid command");
      }

      lineLength = 0;
      lineOverflow = false;
    } else if (c != '\r') {
      if (lineLength < LINE_MAX - 1) {
        lineBuffer[lineLength++] = c;
      } else {
        lineOverflow = true;
      }
    }
  }

  if (haveNew) {
    for (int i = 0; i < SERVO_COUNT; i++) {
      updateServo(i);
    }
  }
}

// Parses lineBuffer into exactly SERVO_COUNT integers in 0-1023.
// Digits and commas only; anything else rejects the whole line.
bool parseLine(int values[]) {
  byte index = 0;
  long current = 0;
  bool hasDigit = false;

  for (byte i = 0; i <= lineLength; i++) {
    char c = (i < lineLength) ? lineBuffer[i] : ',';   // virtual trailing comma

    if (c >= '0' && c <= '9') {
      current = current * 10 + (c - '0');
      hasDigit = true;
      if (current > 1023) return false;
    } else if (c == ',') {
      if (!hasDigit || index >= SERVO_COUNT) return false;
      values[index++] = (int)current;
      current = 0;
      hasDigit = false;
    } else if (c != ' ') {
      return false;
    }
  }

  return index == SERVO_COUNT;
}

void updateServo(int index) {
  // Same range as Servo.write(0-180), but in microseconds so the full
  // 0-1023 resolution is kept instead of being rounded to whole degrees.
  int pulse = map(potValues[index], 0, 1023, MAX_PULSE_WIDTH, MIN_PULSE_WIDTH);

  servos[index].writeMicroseconds(pulse);
}
