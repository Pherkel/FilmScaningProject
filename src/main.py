import cv2
import time
import numpy as np

from Scanner import Scanner


# PIN Setup


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
BorderExpectation = 10   # initial move is (x-1)/x th of the frame, 10 is ideal
CorrectionSpeed = framelength*16  # resolution of a 1/16 micro stepper per frame
CorrectionSpeedP = 832  # Portrait length, 1/16
framelengthP = 52   # Portrait length , 1/1
orientation = 0    # 0 for landscape 1 for portrait
Cropfactor = 0.2  # max 1 min 0, ideal 0.2
# border intensity is found needs to be 95% higher from the rest.
threshold = 0.95


scanner = Scanner(0.9, 36)
