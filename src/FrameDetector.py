import cv2 as cv
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
import math


class FrameDetector:
    __slots__ = ("img", "edges", "lines", "segmented", "intersections")

    def __init__(self, image):
        self.img = image

    @staticmethod
    def detect_edges(img) -> bool:
        try:
            img = cv.GaussianBlur(img, (3, 3), 1)
            img = cv.Canny(img, 100, 75, None, 3)
            return img
        except Exception as err:
            print(err)

    @staticmethod
    def detect_lines(edges) -> bool:
        try:
            lines = cv.HoughLines(edges, 1, np.pi / 180, 175, 0, 0)
            return lines
        except Exception as err:
            print(err)

    @staticmethod
    def segment_lines(lines):
        try:
            segmented = FrameDetector._segment_by_angle_kmeans(lines)
            return segmented

        except Exception as err:
            print(err)

    @staticmethod
    def _segment_by_angle_kmeans(lines, k=2, **kwargs):
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
        angles = np.array([line[0][1] for line in lines])
        # multiply the angles by two and find coordinates of that angle
        pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                        for angle in angles], dtype=np.float32)

        # run kmeans on the coords
        labels, centers = cv.kmeans(
            pts, k, None, criteria, attempts, flags)[1:]
        labels = labels.reshape(-1)  # transpose to row vec

        # segment lines based on their kmeans label
        segmented = defaultdict(list)
        for i, line in enumerate(lines):
            segmented[labels[i]].append(line)
        segmented = list(segmented.values())

        return segmented

    @staticmethod
    def intersection(line1, line2):
        """Finds the intersection of two lines given in Hesse normal form.

        Returns closest integer pixel locations.
        See https://stackoverflow.com/a/383527/5087436
        """
        rho1, theta1 = line1[0]
        rho2, theta2 = line2[0]
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [[x0, y0]]

    @staticmethod
    def segmented_intersections(lines):
        """Finds the intersections between groups of lines."""

        intersections = []
        for i, group in enumerate(lines[:-1]):
            for next_group in lines[i+1:]:
                for line1 in group:
                    for line2 in next_group:
                        intersections.append(
                            FrameDetector.intersection(line1, line2))

        return intersections
