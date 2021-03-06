from FilmScanner.FrameDetector import FrameDetector
import pytest
import math


@pytest.mark.parametrize("rect, expected", [
    ([[0, 0], [0, 24], [36, 0], [36, 24]], math.pow(1.5, 2)),
    ([[0, 0], [0, 1], [1, 0], [1, 1]], math.pow(1.0, 2)),  # 1x1 cube

])
def test_aspect_ratio(rect, expected):
    # pytest.set_trace()
    assert FrameDetector._aspect_ratio(rect) == expected
