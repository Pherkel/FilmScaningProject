from FilmScanner.FrameDetector import FrameDetector
import pytest

@pytest.mark.parametrize("rect, expected", [
    ([[0, 0], [36, 0], [36, 24], [0, 24]], 0)
])
def test_angles(rect, expected):
    pytest.set_trace()
    assert FrameDetector._angles(rect) == expected
