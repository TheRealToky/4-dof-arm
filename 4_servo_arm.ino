
#include <Servo.h>

const byte SERVO_COUNT = 4;

Servo servos[SERVO_COUNT];

const byte servoPins[SERVO_COUNT] = {
  9, 10, 11, 6
};

int potValues[SERVO_COUNT] = {
  512, 512, 512, 512
};

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(30);

  for (int i = 0; i < SERVO_COUNT; i++) {
    servos[i].attach(servoPins[i]);
    updateServo(i);
  }

  Serial.println("4-servo controller ready!");
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');

    int values[SERVO_COUNT];
    int start = 0;
    bool valid = true;

    for (int i = 0; i < SERVO_COUNT; i++) {
      int comma = command.indexOf(',', start);

      String value;

      if (i == SERVO_COUNT - 1) {
        value = command.substring(start);
      } else {
        if (comma == -1) {
          valid = false;
          break;
        }

        value = command.substring(start, comma);
      }

      value.trim();

      if (value.length() == 0) {
        valid = false;
        break;
      }

      // Reject non-numeric characters
      for (unsigned int j = 0; j < value.length(); j++) {
        if (value[j] < '0' || value[j] > '9') {
          valid = false;
          break;
        }
      }

      if (!valid) break;

      values[i] = value.toInt();

      if (values[i] < 0 || values[i] > 1023) {
        valid = false;
        break;
      }

      start = comma + 1;
    }

    if (valid) {
      for (int i = 0; i < SERVO_COUNT; i++) {
        potValues[i] = values[i];
        updateServo(i);
      }

      Serial.println("OK");
    } else {
      Serial.println("Invalid command");
    }
  }
}

void updateServo(int index) {
  int angle = map(potValues[index], 0, 1023, 180, 0);

  servos[index].write(angle);
}
