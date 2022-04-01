from multiprocessing.connection import wait
import cv2
import time
import numpy as np
from Structs import Pinout, Resolutions
import RPi.GPIO as GPIO


class Scanner:
    """Class to Controll automatic film scanning"""

    def __init__(self, waitTime, MaxPics) -> None:
        self.waitingTime = waitTime
        self.MaxPicutureNumber = MaxPics

    # User input
    MaxPicutureNumber = 28

    # When no image border is located, 0 means it doesn't backup to take a picture, 1 for yes
    BackupPictureTake = 1
    waitingTime = 0.9  # 1 seconds after shuter is relesed, increase if pictures are blury
    colorNegative = 1    # 0 for Color positive, 1 for All Negative films
    framelength = 111  # 111 for landscape 35mm, increase if progression falls shor

    # Intrinsic values
    HowManyFramesPerImage = 1    # Not yet developed
    OverlapValue = 0.5  # Not yet developed
    # initial move is (x-1)/x th of the frame, 10 is ideal
    BorderExpectation = 10
    CorrectionSpeed = framelength*16  # resolution of a 1/16 micro stepper per frame
    CorrectionSpeedP = 832  # Portrait length, 1/16
    framelengthP = 52   # Portrait length , 1/1
    orientation = 0    # 0 for landscape 1 for portrait
    Cropfactor = 0.2  # max 1 min 0, ideal 0.2
    # border intensity is found needs to be 95% higher from the rest.
    threshold = 0.95

    latch = 0
    start = 0
    counter = 1  # picture Counter
    autoAuto = 0
    BadFrames = 0
    FrameMoves = 1

    def init_pins():
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(Pinout.DIR, GPIO.OUT)
        GPIO.setup(Pinout.STEP, GPIO.OUT)
        GPIO.output(Pinout.DIR, 1)

    def set_resolution(resolution: Resolutions):
        GPIO.setup(Pinout.MODE, GPIO.OUT)
        GPIO.output(Pinout.MODE, resolution)

    def label(res):
        label = np.zeros(len(res))
        label[0] = 1
        for i in range(len(res)-1):
            if res[i] == res[i+1]-1:
                label[i+1] = label[i]
            else:
                label[i+1] = label[i]+1
        return label

    def takeApicture(counter):
        print("picture number: ", counter)
        counter = counter + 1
