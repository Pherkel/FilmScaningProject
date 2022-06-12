import cv2 as cv
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
import math


class FrameDetector:
    __slots__ = ("img", "edges", "lines", "intersections", "rectangle", "rectangles")

    def __init__(self, image):
        self.img = image
        self.rectangles = []

    def determine_lines(self):
        self.edges = FrameDetector.detect_edges(self.img)
        self.lines = FrameDetector.detect_lines(self.edges)

    def determine_intersections(self):
        segmented = FrameDetector._segment_by_angle_kmeans(self.lines)
        self.intersections = FrameDetector._segmented_intersections(segmented)

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
        pts = np.array([[np.cos(2 * angle), np.sin(2 * angle)]
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
    def _intersection(line1, line2):
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
        return [x0, y0]

    @staticmethod
    def _segmented_intersections(lines):
        """Finds the intersections between groups of lines."""

        intersections = []
        for i, group in enumerate(lines[:-1]):
            for next_group in lines[i + 1:]:
                for line1 in group:
                    for line2 in next_group:
                        intersections.append(
                            FrameDetector._intersection(line1, line2))

        return intersections

    @staticmethod
    def _vec_from_points(p1, p2):
        return [p1[0] - p2[0], p1[1] - p2[1]]

    @staticmethod
    def _vec_length(vec):
        return math.sqrt(math.pow(vec[0], 2) + math.pow(vec[1], 2))

    @staticmethod
    def _vec_sq_length(vec):
        return math.pow(vec[0], 2) + math.pow(vec[1], 2)

    @staticmethod
    def _vec_angle_fast(vec1, vec2):
        vec = FrameDetector._vec_from_points(vec1, vec2)

        if vec[1] >= 0:
            if vec[0] >= 0:
                return vec[1] / (vec[0] + vec[1])
            else:
                return 1 - vec[0] / (-vec[0] + vec[1])
        else:
            if vec[0] < 0:
                return 2 - vec[1] / (-vec[1] - vec[0])
            else:
                return 3 + vec[0] / (vec[0] - vec[1])

    @staticmethod
    def _angle_threshold(angle, lower=0.8, upper=1.2):
        pos = angle >= lower or angle <= upper
        neg = angle >= lower+2 or angle <= upper+2
        return not(pos or neg)

    @staticmethod
    def _form_rectangle(intersections):
        out = []

        for point1 in intersections:
            for point2 in intersections:
                for point3 in intersections:
                    for point4 in intersections:

                        if point1 is point2:
                            continue
                        if point1 is point3:
                            continue
                        if point1 is point4:
                            continue
                        if point2 is point3:
                            continue
                        if point2 is point4:
                            continue
                        if point3 is point4:
                            continue

                        rect = [point1, point2, point3, point4]

                        rect.sort(key=lambda x: (x[0], x[1]))
                        if rect in out: continue
                        if FrameDetector._rate_rectangle(rect) > 0.2: continue

                        p1_p2 = FrameDetector._vec_from_points(point1, point2)
                        p2_p3 = FrameDetector._vec_from_points(point2, point3)
                        p3_p4 = FrameDetector._vec_from_points(point3, point4)
                        p4_p1 = FrameDetector._vec_from_points(point4, point1)

                        angle1 = FrameDetector._vec_angle_fast(p1_p2, p2_p3)
                        angle2 = FrameDetector._vec_angle_fast(p2_p3, p3_p4)
                        angle3 = FrameDetector._vec_angle_fast(p3_p4, p4_p1)
                        angle4 = FrameDetector._vec_angle_fast(p4_p1, p1_p2)

                        if FrameDetector._angle_threshold(angle1) or FrameDetector._angle_threshold(
                                angle2) or FrameDetector._angle_threshold(angle3) or FrameDetector._angle_threshold(
                            angle4):
                                continue

                        out.append(rect)

        return out

    @staticmethod
    def _rate_rectangle(rectangle) -> int:

        top = FrameDetector._vec_from_points(rectangle[0], rectangle[1])
        right = FrameDetector._vec_from_points(rectangle[0], rectangle[2])
        bottom = FrameDetector._vec_from_points(rectangle[2], rectangle[3])
        left = FrameDetector._vec_from_points(rectangle[1], rectangle[3])

        height = 0.5 * (FrameDetector._vec_sq_length(right) + FrameDetector._vec_sq_length(left))
        length = 0.5 * (FrameDetector._vec_sq_length(top) + FrameDetector._vec_sq_length(bottom))

        aspect_r = length / height

        return abs(math.pow((2/3), 2) - aspect_r)

    def determine_rectangle(self):
        # outline
        # 1. pick point from intersections -> if too slow ransac
        # 2. determine 3 other random points to form a rectangle
        # 3. check if the selected rect has already been looked at
        # 4. rank the rectangle by how many edgepoints are on it's perimeter
        # 5. save the combination of points for this rectangle
        # 6. do for all points

        best_rating = 1000
        best_rect = (0, 0, 0, 0)

        rects = FrameDetector._form_rectangle(self.intersections)

        for rect in rects:

            rating = FrameDetector._rate_rectangle(rect)

            if rating < best_rating:
                best_rating = rating
                best_rect = rect
                print(rating)
                self.rectangles.append(rect)

        self.rectangle = best_rect
