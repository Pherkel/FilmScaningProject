import numpy as np
import cv2 as cv
import math
from FrameDetector import FrameDetector

img = cv.imread("/config/FilmScaningProject/src/Photo-1.jpeg",
                cv.IMREAD_GRAYSCALE)

frame = FrameDetector(img)

edges = FrameDetector.detect_edges(img)

lines = FrameDetector.detect_lines(edges)

segmented = FrameDetector._segment_by_angle_kmeans(lines)

intersections = FrameDetector.segmented_intersections(segmented)

cdst = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

# TODO: merge intersections within a specific distance of each other
# (do I really need to do this?)

# TODO: construct rectangle from given points:
# - probably 4 points?
# - aspect ratio should match 3:2
# - mostly 90° angles

# TODO: find center of detected rectangle

# TODO: estimate sizes with sprocket holes
# enables precise movement instructions to motor

for i in range(0, len(lines)):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 10000*(-b)), int(y0 + 10000*(a)))
    pt2 = (int(x0 - 10000*(-b)), int(y0 - 10000*(a)))
    cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

for i in range(0, len(intersections)):
    cv.circle(cdst, intersections[i][0], radius=0,
              color=(0, 255, 0), thickness=10)

cv.imwrite("lines.png", cdst)
