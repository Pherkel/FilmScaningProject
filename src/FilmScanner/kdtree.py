import collections
import math
from collections import namedtuple
from operator import itemgetter
from pprint import pformat


class Node(namedtuple("Node", "value left right")):
    def __repr__(self):
        return pformat(tuple(self))


class kdtree:
    tree = Node

    def __init__(self, points, depth):
        self.tree = kdtree.build_kdtree(points, depth)

    @staticmethod
    def build_kdtree(points, depth: int = 0):
        if not points:
            return None

        k = len(points[0])  # assumes all points have the same dimension
        # Select axis based on depth so that axis cycles through all valid values
        axis = depth % k

        # Sort point list by axis and choose median as pivot element
        points.sort(key=itemgetter(axis))
        median = len(points) // 2

        # Create node and construct subtrees
        return Node(
            value=points[median],
            left=kdtree.build_kdtree(points[:median], depth + 1),
            right=kdtree.build_kdtree(points[median + 1:], depth + 1),
        )

    NNRecord = collections.namedtuple("NNRecord", ["point", "distance"])

    @staticmethod
    def find_nn(tree, point):
        NNRecord = collections.namedtuple("NNRecord", ["point", "distance"])
        k = len(point)
        best = None

        def search(tree, depth):
            nonlocal best

            if tree is None:
                return

            distance = kdtree._sed(tree.value, point)
            if best is None or distance < best.distance:
                best = NNRecord(point=tree.value, distance=distance)

            axis = depth % k
            diff = point[axis] - tree.value[axis]
            if diff <= 0:
                close, away = tree.left, tree.right
            else:
                close, away = tree.right, tree.left

            search(tree=close, depth=depth + 1)
            if diff ** 2 < best.distance:
                search(tree=away, depth=depth + 1)

        search(tree=tree, depth=0)
        return best.point

    @staticmethod
    def _sed(vec1, vec2):

        vec = [vec1[0] - vec2[0], vec1[1] - vec2[1]]
        return math.pow(vec[0], 2) + math.pow(vec[1], 2)
