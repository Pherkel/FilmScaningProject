rom turtle import right
import cv2
from time import sleep
import numpy as np
import RPi.GPIO as GPIO
from src.FilmScanner.main import BorderExpectation
from scipy.stats import rankdata


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

    def init_pins() -> bool:
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(Pinout.DIR, GPIO.OUT)
            GPIO.setup(Pinout.STEP, GPIO.OUT)
            GPIO.output(Pinout.DIR, 1)
        except:
            return False
        return True

    def set_resolution(resolution: Resolutions) -> bool:
        GPIO.setup(Pinout.MODE, GPIO.OUT)
        GPIO.output(Pinout.MODE, resolution)
        return True

    def label(res) -> np.array:
        label = np.zeros(len(res))
        label[0] = 1
        for i in range(len(res)-1):
            if res[i] == res[i+1]-1:
                label[i+1] = label[i]
            else:
                label[i+1] = label[i]+1
        return label

    def takeApicture(self, counter: int) -> int:
        print("picture number: ", counter)
        counter = counter + 1
        GPIO.output(Pinout.SHUTTER, GPIO.HIGH)
        sleep(0.3)
        GPIO.output(Pinout.SHUTTER, GPIO.LOW)
        sleep(self.waitingTime)
        return counter

    def functionFAST(self, xx, yy, a, b, c, wait):
        Pinout.DIR()
        # todo

    def maxEdge(self, FrameProjection, position):
        if self.orientation == 0:
            FrameLimit = len(FrameProjection)
            EdgeValue = 5
        else:
            FrameLimit = len(FrameLimit)
            EdgeValue = 5

        if min(position) >= 0 + EdgeValue:
            a = float(FrameProjection[min(position)])
            b = float(FrameProjection[min(position) - 3])

        if max(position) <= FrameLimit-EdgeValue:
            a = float(FrameProjection[max(position)])
            b = float(FrameProjection[min(position) - 3])

        if min(position) <= EdgeValue or max(position) >= FrameLimit-EdgeValue:
            return -1, 1
        else:
            return left, right

    def captureData(self):
        video = cv2.VideoCapture(2)
        sleep(0.9)
        check, frame = video.read()
        height = len(frame[:, 0])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        FrameProjection = np.sum(frame, axis=self.orientation)

        CropVertical = round(height*self.Cropfactor)
        b = max(np.argwhere(FrameProjection > 0))-5
        a = np.argmax(FrameProjection > 0)+5
        frame = frame[0+CropVertical:height-CropVertical, a:int(b)]
        FrameProjection = FrameProjection[a:int(b)]
        if self.colorNegative == 0:
            frame = -frame
            FrameProjection = max(FrameProjection)-FrameProjection

        maskHor = (FrameProjection/max(FrameProjection)) > self.threshold

        position = np.where(maskHor == 1)[0]
        items = self.label(position)

        a = np.array([(x,
                       len(position[items == x]),
                       round(np.median(position[items == x])),
                       max(np.sum(frame, axis=self.orientation)[position[items == x]]))
                      for i, x in enumerate(np.unique(items))])

        PFramelength, LFramelength = np.shape(frame)
        return a, FrameProjection, position, items, PFramelength, LFramelength

    def CheckingEdgeDeciding(self, aa, FrameProjection, position, items, latch, BackupPictureTage):
        if self.orientation == 0:
            a = 11
            idealBoderSize = 22
        else:
            a = 22
            idealBoderSize = 44
        aa[:, 1] = abs(aa[:, 1]-idealBoderSize)
        Positonselected = 0
        weights = [[1], [4], [1]]
        if latch == 0:
            aa[:, 2] = abs(aa[:, 2]-round(LFramelenght/BorderExpectation))
        elif latch == 2:
            aa[:, 2] = abs(aa[:, 2]-(BorderExpectation-1.5) /
                           BorderExpectation*LFramelength)

        decide = sum([rankdata(-aa[:, 1]), rankdat(-aa[:, 2]),
                     rankdata(aa[:, 3])*np.array(weights)])
        Itemselected = aa[decide == max(decide), 0]
