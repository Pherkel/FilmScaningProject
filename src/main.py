import numpy as np
import time
import math
import cv2 as cv
from matplotlib import pyplot as plt
from FrameDetector import FrameDetector

# read image

img = cv.imread("/config/FilmScaningProject/src/Photo-1.jpeg",
                cv.IMREAD_GRAYSCALE)

frame = FrameDetector(img)

if not frame.detect_edges():
    print("edge-detection failed!")

if not frame.detect_lines():
    print("line-detection failed!")

if not frame.segment_lines():
    print("line-segmentation failed!")

if not frame.calculate_intersections():
    print("calculating intersections failed!")

cdst = cv.cvtColor(frame.edges, cv.COLOR_GRAY2BGR)

# TODO: merge intersections within a specific distance of each other
# (do I really need to do this?)

# TODO: construct rectangle from given points:
# - probably 4 points?
# - aspect ratio should match 3:2
# - mostly 90° angles

# TODO: find center of detected rectangle

# TODO: estimate sizes with sprocket holes
# enables precise movement instructions to motor

if frame.lines is not None:
    for i in range(0, len(frame.lines)):
        rho = frame.lines[i][0][0]
        theta = frame.lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 10000*(-b)), int(y0 + 10000*(a)))
        pt2 = (int(x0 - 10000*(-b)), int(y0 - 10000*(a)))
        cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

for i in range(0, len(frame.intersections)):
    cv.circle(cdst, frame.intersections[i][0], radius=0,
              color=(0, 255, 0), thickness=10)

cv.imwrite("lines.png", cdst)
