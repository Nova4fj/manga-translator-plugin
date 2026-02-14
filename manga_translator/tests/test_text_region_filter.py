"""Tests for the TextRegionFilter module."""

import cv2
import numpy as np
import pytest

from manga_translator.components.text_region_filter import (
    TextRegionFilter,
    TextRegionScore,
)


@pytest.fixture
def filter():
    """Default TextRegionFilter instance."""
    return TextRegionFilter()


# ---------------------------------------------------------------
# 1. test_empty_region
# ---------------------------------------------------------------

def test_empty_region(filter):
    """None and empty arrays should return region_type 'empty'."""
    score_none = filter.analyze_region(None)
    assert score_none.region_type == "empty"
    assert score_none.has_text is False

    score_empty = filter.analyze_region(np.array([], dtype=np.uint8))
    assert score_empty.region_type == "empty"
    assert score_empty.has_text is False


# ---------------------------------------------------------------
# 2. test_solid_white
# ---------------------------------------------------------------

def test_solid_white(filter):
    """A uniform white image should be classified as 'solid'."""
    white = np.ones((100, 100), dtype=np.uint8) * 255
    score = filter.analyze_region(white)
    assert score.region_type == "solid"
    assert score.has_text is False
    assert score.confidence < 0.3


# ---------------------------------------------------------------
# 3. test_solid_black
# ---------------------------------------------------------------

def test_solid_black(filter):
    """A uniform black image should be classified as 'solid'."""
    black = np.zeros((100, 100), dtype=np.uint8)
    score = filter.analyze_region(black)
    assert score.region_type == "solid"
    assert score.has_text is False


# ---------------------------------------------------------------
# 4. test_gradient
# ---------------------------------------------------------------

def test_gradient(filter):
    """A smooth linear gradient should be classified as 'gradient'."""
    gradient = np.zeros((100, 100), dtype=np.uint8)
    for i in range(100):
        gradient[i, :] = int(i * 255 / 99)
    score = filter.analyze_region(gradient)
    assert score.region_type == "gradient"
    assert score.has_text is False


# ---------------------------------------------------------------
# 5. test_text_on_white
# ---------------------------------------------------------------

def test_text_on_white(filter):
    """Black text drawn on a white background should be detected as 'text'."""
    img = np.ones((200, 300), dtype=np.uint8) * 255

    # Draw multiple lines of text to create a realistic text region
    cv2.putText(img, "Hello World!", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, 0, 2, cv2.LINE_AA)
    cv2.putText(img, "Second line", (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, 0, 2, cv2.LINE_AA)
    cv2.putText(img, "Third line!!", (10, 140), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, 0, 2, cv2.LINE_AA)
    cv2.putText(img, "More text...", (10, 190), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, 0, 2, cv2.LINE_AA)

    score = filter.analyze_region(img)
    assert score.region_type == "text"
    assert score.has_text is True
    assert score.confidence >= 0.3


# ---------------------------------------------------------------
# 6. test_random_noise
# ---------------------------------------------------------------

def test_random_noise(filter):
    """Random noise should be classified as 'noise'."""
    rng = np.random.RandomState(42)
    noise = rng.randint(0, 256, (100, 100), dtype=np.uint8)
    score = filter.analyze_region(noise)
    assert score.region_type == "noise"
    assert score.has_text is False


# ---------------------------------------------------------------
# 7. test_small_region
# ---------------------------------------------------------------

def test_small_region(filter):
    """A very small region (< 100 pixels) should return 'empty'."""
    tiny = np.ones((5, 5), dtype=np.uint8) * 128
    score = filter.analyze_region(tiny)
    assert score.region_type == "empty"
    assert score.has_text is False


# ---------------------------------------------------------------
# 8. test_filter_regions_mixed
# ---------------------------------------------------------------

def test_filter_regions_mixed(filter):
    """filter_regions should return only the text regions with their indices."""
    # Region 0: solid white (not text)
    white = np.ones((100, 100), dtype=np.uint8) * 255

    # Region 1: text on white (text)
    text_img = np.ones((200, 300), dtype=np.uint8) * 255
    cv2.putText(text_img, "Hello World!", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, 0, 2, cv2.LINE_AA)
    cv2.putText(text_img, "Second line", (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, 0, 2, cv2.LINE_AA)
    cv2.putText(text_img, "Third line!!", (10, 140), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, 0, 2, cv2.LINE_AA)
    cv2.putText(text_img, "More text...", (10, 190), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, 0, 2, cv2.LINE_AA)

    # Region 2: solid black (not text)
    black = np.zeros((100, 100), dtype=np.uint8)

    regions = [white, text_img, black]
    results = filter.filter_regions(regions)

    # Only the text region (index 1) should pass
    indices = [idx for idx, _ in results]
    assert 1 in indices
    assert 0 not in indices
    assert 2 not in indices


# ---------------------------------------------------------------
# 9. test_text_line_score_uniform
# ---------------------------------------------------------------

def test_text_line_score_uniform(filter):
    """A uniform image should have a low text line score."""
    uniform = np.ones((100, 100), dtype=np.uint8) * 200
    score = filter._compute_text_line_score(uniform)
    assert score == 0.0


# ---------------------------------------------------------------
# 10. test_text_line_score_with_lines
# ---------------------------------------------------------------

def test_text_line_score_with_lines(filter):
    """An image with alternating horizontal stripes should have a higher
    text line score than a uniform image."""
    striped = np.zeros((100, 100), dtype=np.uint8)
    for i in range(100):
        if (i // 5) % 2 == 0:
            striped[i, :] = 255
        else:
            striped[i, :] = 0

    score = filter._compute_text_line_score(striped)
    assert score > 0.0


# ---------------------------------------------------------------
# 11. test_custom_thresholds
# ---------------------------------------------------------------

def test_custom_thresholds(filter):
    """Custom min_confidence should change whether has_text is True."""
    # Create a borderline text image
    img = np.ones((200, 300), dtype=np.uint8) * 255
    cv2.putText(img, "Hello World!", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, 0, 2, cv2.LINE_AA)
    cv2.putText(img, "Second line", (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, 0, 2, cv2.LINE_AA)
    cv2.putText(img, "Third line!!", (10, 140), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, 0, 2, cv2.LINE_AA)

    # Very strict filter — high min_confidence
    strict_filter = TextRegionFilter(min_confidence=0.95)
    score_strict = strict_filter.analyze_region(img)

    # Lenient filter — low min_confidence
    lenient_filter = TextRegionFilter(min_confidence=0.1)
    score_lenient = lenient_filter.analyze_region(img)

    # The strict filter should reject what the lenient one accepts
    # (or at minimum, the lenient filter should accept it)
    assert score_lenient.region_type == "text"
    assert score_lenient.has_text is True
    # Strict filter: same region_type but has_text should be False
    # because confidence < 0.95
    assert score_strict.region_type == "text"
    assert score_strict.has_text is False


# ---------------------------------------------------------------
# 12. test_grayscale_input
# ---------------------------------------------------------------

def test_grayscale_input(filter):
    """The filter should work with 2D (grayscale) arrays directly."""
    gray = np.ones((100, 100), dtype=np.uint8) * 255
    cv2.putText(gray, "Test text", (5, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, 0, 2, cv2.LINE_AA)
    cv2.putText(gray, "More text", (5, 90), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, 0, 2, cv2.LINE_AA)

    assert len(gray.shape) == 2  # Confirm it's 2D
    score = filter.analyze_region(gray)

    # Should produce a valid result (not crash)
    assert isinstance(score, TextRegionScore)
    assert score.region_type in ("text", "empty", "solid", "gradient", "noise")
