"""Tests for region exclusion mask functionality."""

import numpy as np
from types import SimpleNamespace

from manga_translator.region_mask import (
    parse_exclusion_regions,
    create_exclusion_mask,
    filter_bubbles_by_mask,
)


class TestParseExclusionRegions:
    def test_valid_single_region(self):
        result = parse_exclusion_regions("10,20,100,50")
        assert result == [(10, 20, 100, 50)]

    def test_valid_multiple_regions(self):
        result = parse_exclusion_regions("0,0,100,50;200,300,100,50")
        assert result == [(0, 0, 100, 50), (200, 300, 100, 50)]

    def test_empty_string(self):
        assert parse_exclusion_regions("") == []

    def test_none_like_empty(self):
        # None is falsy, same branch as empty string
        assert parse_exclusion_regions(None) == []

    def test_whitespace_handling(self):
        result = parse_exclusion_regions(" 10 , 20 , 100 , 50 ; 5,5,10,10 ")
        assert result == [(10, 20, 100, 50), (5, 5, 10, 10)]

    def test_malformed_part_skipped(self):
        # A part with only 3 coords is silently skipped
        result = parse_exclusion_regions("10,20,100;0,0,50,50")
        assert result == [(0, 0, 50, 50)]

    def test_extra_parts_skipped(self):
        # A part with 5 coords is also skipped (only exactly 4 accepted)
        result = parse_exclusion_regions("1,2,3,4,5;10,20,30,40")
        assert result == [(10, 20, 30, 40)]


class TestCreateExclusionMask:
    def test_no_regions(self):
        mask = create_exclusion_mask(100, 200, [])
        assert mask.shape == (100, 200)
        assert np.all(mask == 255)

    def test_single_region(self):
        mask = create_exclusion_mask(100, 200, [(10, 20, 50, 30)])
        # Excluded region should be 0
        assert mask[20, 10] == 0
        assert mask[49, 59] == 0  # y=20+30-1=49, x=10+50-1=59
        # Outside region should be 255
        assert mask[0, 0] == 255
        assert mask[99, 199] == 255

    def test_multiple_regions(self):
        mask = create_exclusion_mask(200, 200, [(0, 0, 50, 50), (100, 100, 50, 50)])
        assert mask[25, 25] == 0      # inside first region
        assert mask[125, 125] == 0    # inside second region
        assert mask[75, 75] == 255    # between regions

    def test_mask_dtype(self):
        mask = create_exclusion_mask(10, 10, [(0, 0, 5, 5)])
        assert mask.dtype == np.uint8


class TestFilterBubblesByMask:
    @staticmethod
    def _make_bubble(bbox):
        """Create a minimal bubble-like object with a bbox attribute."""
        return SimpleNamespace(bbox=bbox)

    def test_all_included(self):
        mask = np.ones((500, 500), dtype=np.uint8) * 255
        bubbles = [
            self._make_bubble((100, 100, 50, 50)),
            self._make_bubble((200, 200, 60, 40)),
        ]
        result = filter_bubbles_by_mask(bubbles, mask)
        assert len(result) == 2

    def test_all_excluded(self):
        mask = np.zeros((500, 500), dtype=np.uint8)
        bubbles = [
            self._make_bubble((100, 100, 50, 50)),
            self._make_bubble((200, 200, 60, 40)),
        ]
        result = filter_bubbles_by_mask(bubbles, mask)
        assert len(result) == 0

    def test_partial_exclusion(self):
        mask = np.ones((500, 500), dtype=np.uint8) * 255
        # Exclude a region that covers the center of the first bubble
        # Bubble at (100,100,50,50) has center (125,125)
        mask[100:150, 100:150] = 0
        bubbles = [
            self._make_bubble((100, 100, 50, 50)),  # center (125,125) -> excluded
            self._make_bubble((300, 300, 60, 40)),  # center (330,320) -> included
        ]
        result = filter_bubbles_by_mask(bubbles, mask)
        assert len(result) == 1
        assert result[0].bbox == (300, 300, 60, 40)

    def test_bubble_center_on_boundary(self):
        # Bubble center exactly on the edge of the excluded region
        mask = np.ones((500, 500), dtype=np.uint8) * 255
        mask[0:50, 0:50] = 0
        # Bubble at (0,0,100,100) has center (50,50) which is at mask[50,50]=255
        bubbles = [self._make_bubble((0, 0, 100, 100))]
        result = filter_bubbles_by_mask(bubbles, mask)
        assert len(result) == 1

    def test_empty_bubbles_list(self):
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        result = filter_bubbles_by_mask([], mask)
        assert result == []
