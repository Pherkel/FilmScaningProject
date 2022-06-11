import cv2 as cv
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
import math


class FrameDetector:
    __slots__ = ("img", "edges", "lines", "segmented", "intersections")

    def __init__(self, image):
        self.img = image

    def detect_edges(self) -> bool:
        try:
            #self.img = cv.GaussianBlur(self.img, (3, 3), 1)
            self.edges = cv.Canny(self.img, 100, 75, None, 3)
            return True
        except:
            return False

    def detect_lines(self) -> bool:
        try:
            self.lines = cv.HoughLines(self.edges, 1, np.pi / 180, 175, 0, 0)
            return True
        except:
            return False

    def segment_lines(self) -> bool:
        try:
            self._segment_by_angle_kmeans()
            return True
        except Exception as err:
            print(err)
            return False

    def calculate_intersections(self) -> bool:
        try:
            self.intersections = self._segmented_intersections()
            return True
        except Exception as err:
            print(err)
            return False

    def _segment_by_angle_kmeans(self, k=2, **kwargs) -> None:
        """Groups lines based on angle with k-means.

        Uses k-means on the coordinates of the angle on the unit circle 
        to segment `k` angles inside `lines`.
        """

        # Define criteria = (type, max_iter, epsilon)
        default_criteria_type = cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER
        criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
        flags = kwargs.get('flags', cv.KMEANS_RANDOM_CENTERS)
        attempts = kwargs.get('attempts', 10)

        # returns angles in [0, pi] in radians
        angles = np.array([line[0][1] for line in self.lines])
        # multiply the angles by two and find coordinates of that angle
        pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                        for angle in angles], dtype=np.float32)

        # run kmeans on the coords
        labels, centers = cv.kmeans(
            pts, k, None, criteria, attempts, flags)[1:]
        labels = labels.reshape(-1)  # transpose to row vec

        # segment lines based on their kmeans label
        segmented = defaultdict(list)
        for i, line in enumerate(self.lines):
            segmented[labels[i]].append(line)
        segmented = list(segmented.values())

        self.segmented = segmented

    @staticmethod
    def _intersection(line1, line2):
        """Finds the intersection of two lines given in Hesse normal form.

        Returns closest integer pixel locations.
        See https://stackoverflow.com/a/383527/5087436
        """
        rho1, theta1 = line1[0], line1[1]
        rho2, theta2 = line2[0], line2[1]
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [[x0, y0]]

    def _segmented_intersections(self):
        """Finds the intersections between groups of lines."""

        intersections = []
        for i, group in enumerate(self.lines[:-1]):
            for next_group in self.lines[i+1:]:
                for line1 in group:
                    for line2 in next_group:
                        intersections.append(
                            FrameDetector._intersection(line1, line2))

        return intersections
