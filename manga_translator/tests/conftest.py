"""Shared test fixtures for manga translator tests."""

import numpy as np
import pytest


@pytest.fixture
def sample_manga_page():
    """Create a synthetic manga page with white bubbles on gray background."""
    page = np.full((800, 600, 3), 180, dtype=np.uint8)  # gray background

    # Draw 3 white speech bubbles (ellipses)
    import cv2

    # Bubble 1 — top right (manga reading order)
    cv2.ellipse(page, (450, 150), (100, 60), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(page, (450, 150), (100, 60), 0, 0, 360, (0, 0, 0), 2)
    # Add fake text (dark scribbles inside)
    cv2.putText(page, "TEST", (400, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)

    # Bubble 2 — middle left
    cv2.ellipse(page, (150, 400), (120, 70), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(page, (150, 400), (120, 70), 0, 0, 360, (0, 0, 0), 2)
    cv2.putText(page, "HELLO", (90, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)

    # Bubble 3 — bottom right
    cv2.ellipse(page, (400, 650), (80, 50), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(page, (400, 650), (80, 50), 0, 0, 360, (0, 0, 0), 2)
    cv2.putText(page, "OK", (375, 655), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)

    return page


@pytest.fixture
def white_bubble_region():
    """A small white image region simulating inside a bubble with text."""
    region = np.full((120, 200, 3), 255, dtype=np.uint8)
    import cv2

    cv2.putText(region, "Hello", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    return region


@pytest.fixture
def default_settings():
    """Default plugin settings."""
    from manga_translator.config.settings import PluginSettings

    return PluginSettings()
