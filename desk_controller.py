
import RPi.GPIO as GPIO
import time
import logging

SERVO_PIN = 14
LEFT = 6.5
NEUTRAL = 7.5
RIGHT = 9


class DeskController:

    def __init__(self):
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(SERVO_PIN, GPIO.OUT)
        self.servo = GPIO.PWM(SERVO_PIN, 50)
        self.servo.start(0)

    def ascend(self):
       self.servo.ChangeDutyCycle(RIGHT)

    def descend(self):
        self.servo.ChangeDutyCycle(LEFT)

    def halt(self):
       self.servo.ChangeDutyCycle(NEUTRAL)

    def ascend_to_top(self):
        self.ascend()
        time.sleep(13)
        self.halt()

    def descend_to_bottom(self):
        self.descend()
        time.sleep(13)
        self.halt()

    def descend_to_half(self):
        self.descend()
        time.sleep(6)
        self.halt()

    def ascend_to_half(self):
        self.ascend()
        time.sleep(6)
        self.halt()

    def __enter__(self):
       return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()

    def close(self):
        self.servo.close()
        GPIO.cleanup()