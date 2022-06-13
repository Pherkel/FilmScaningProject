import numpy as np
import cv2 as cv
from FilmScanner.FrameDetector import FrameDetector

img = cv.imread('..\..\misc\Photo-1.jpeg',
                cv.IMREAD_GRAYSCALE)

frame = FrameDetector(img)

frame.determine_lines()
frame.determine_intersections()
# TODO: make this function tons faster (k-d tree)
frame.determine_rectangle()

cdst = cv.cvtColor(frame.edges, cv.COLOR_GRAY2BGR)

# TODO: find center of detected rectangle

# TODO: merge intersections within a specific distance of each other

# TODO: estimate sizes with sprocket holes

"""
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
"""
for i in range(0, len(frame.intersections)):
    cv.circle(cdst, frame.intersections[i], radius=0,
              color=(0, 255, 0), thickness=10)

for rect in frame.rectangles:
    rect_coords = np.array(rect, np.int32)
    rect_coords = rect_coords.reshape((-1, 1, 2))
    cv.polylines(cdst, [rect_coords], True, (255, 0, 255), 2)

rect_coords = np.array(frame.rectangle, np.int32)
rect_coords = rect_coords.reshape((-1, 1, 2))
cv.polylines(cdst, [rect_coords], True, (255, 255, 255), 5)

cv.imwrite("..\..\lines.png", cdst)
