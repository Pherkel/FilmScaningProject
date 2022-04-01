from dataclasses import dataclass


@dataclass
class Pinout:
    DIR = 21  # direction pin
    STEP = 20  # stepper pin
    SPR = 96  # steps per revolution
    SHUTTER = 0  # shutter pin
    MODE = (14, 15, 18)  # resolution settings
