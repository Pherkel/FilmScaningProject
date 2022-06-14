import numpy as np
import cProfile
import pstats
import cv2
from FilmScanner.FrameDetector import FrameDetector

img = cv2.imread('..\..\misc\Photo-1.jpeg',
                 cv2.IMREAD_GRAYSCALE)
frame = FrameDetector(img)

frame.determine_lines()
frame.determine_intersections()
# TODO: make this function tons faster (k-d tree)

with cProfile.Profile() as pr:
    frame.determine_rectangle()

stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
# stats.print_stats()
stats.dump_stats(filename='rectangle_stats.prof')



cdst = cv2.cvtColor(frame.edges, cv2.COLOR_GRAY2BGR)

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
    cv2.circle(cdst, frame.intersections[i], radius=0,
               color=(0, 255, 0), thickness=10)

plot_rect = [frame.rectangle[0], frame.rectangle[1], frame.rectangle[3], frame.rectangle[2]]
rect_coords = np.array(plot_rect, np.int32)
rect_coords = rect_coords.reshape((-1, 1, 2))
cv2.polylines(cdst, [rect_coords], True, (255, 255, 255), 5)
print(frame.rectangle)

cv2.imwrite("..\..\misc\lines.png", cdst)
print("Finished")
