from FilmScanner.FrameDetector import FrameDetector
import pytest

@pytest.mark.parametrize("vec1, vec2, expected", [
    ([1, 0], [0, 1], 1),
])
def test_angles(vec1, vec2, expected):
    assert FrameDetector._vec_angle_fast(vec1, vec2) == expected

@pytest.mark.parametrize("rect, expected", [
    ([[0, 0], [36, 0], [36, 24], [0, 24]], 0)
])
def test_rect_angles(rect, expected):
    #pytest.set_trace()
    assert FrameDetector._angles(rect) == expected
