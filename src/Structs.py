from dataclasses import dataclass


@dataclass
class Pinout:
    DIR: int = 21  # direction pin
    STEP: int = 20  # stepper pin
    SPR: int = 96  # steps per revolution
    SHUTTER: int = 0  # shutter pin
    MODE: int = (14, 15, 18)  # resolution settings


@dataclass
class Resolutions:
    Full = (0, 0, 0)
    Half = (1, 0, 0)
    Quarter = (0, 1, 0)
    Eighth = (1, 1, 0)
    Sixteenth = (1, 1, 1)
