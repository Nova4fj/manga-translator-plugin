"""Tests for panel-aware reading order optimizer."""

import pytest

from manga_translator.components.reading_order import ReadingOrderOptimizer, Panel


class _FakeBubble:
    def __init__(self, bbox):
        self.bbox = bbox
        x, y, w, h = bbox
        self.center = (x + w // 2, y + h // 2)


class TestReadingOrderRTL:
    def test_single_bubble(self):
        opt = ReadingOrderOptimizer(reading_direction="rtl")
        bubbles = [_FakeBubble((100, 100, 50, 50))]
        result = opt.sort_bubbles(bubbles)
        assert len(result) == 1

    def test_two_bubbles_same_row_rtl(self):
        opt = ReadingOrderOptimizer(reading_direction="rtl")
        b_left = _FakeBubble((50, 100, 60, 40))
        b_right = _FakeBubble((200, 100, 60, 40))
        result = opt.sort_bubbles([b_left, b_right])
        assert result[0] is b_right
        assert result[1] is b_left

    def test_two_rows_top_to_bottom(self):
        opt = ReadingOrderOptimizer(reading_direction="rtl")
        b_top = _FakeBubble((200, 50, 60, 40))
        b_bottom = _FakeBubble((200, 300, 60, 40))
        result = opt.sort_bubbles([b_bottom, b_top])
        assert result[0] is b_top
        assert result[1] is b_bottom

    def test_four_bubbles_manga_order(self):
        """Classic 2x2 manga layout: TR, TL, BR, BL."""
        opt = ReadingOrderOptimizer(reading_direction="rtl")
        tl = _FakeBubble((50, 50, 60, 40))
        tr = _FakeBubble((300, 50, 60, 40))
        bl = _FakeBubble((50, 300, 60, 40))
        br = _FakeBubble((300, 300, 60, 40))
        result = opt.sort_bubbles([tl, tr, bl, br])
        assert result == [tr, tl, br, bl]

    def test_empty_list(self):
        opt = ReadingOrderOptimizer(reading_direction="rtl")
        assert opt.sort_bubbles([]) == []


class TestReadingOrderLTR:
    def test_two_bubbles_same_row_ltr(self):
        opt = ReadingOrderOptimizer(reading_direction="ltr")
        b_left = _FakeBubble((50, 100, 60, 40))
        b_right = _FakeBubble((200, 100, 60, 40))
        result = opt.sort_bubbles([b_left, b_right])
        assert result[0] is b_left
        assert result[1] is b_right

    def test_four_bubbles_manhwa_order(self):
        """2x2 manhwa layout: TL, TR, BL, BR."""
        opt = ReadingOrderOptimizer(reading_direction="ltr")
        tl = _FakeBubble((50, 50, 60, 40))
        tr = _FakeBubble((300, 50, 60, 40))
        bl = _FakeBubble((50, 300, 60, 40))
        br = _FakeBubble((300, 300, 60, 40))
        result = opt.sort_bubbles([tl, tr, bl, br])
        assert result == [tl, tr, bl, br]


class TestDetectPanels:
    def test_single_panel(self):
        opt = ReadingOrderOptimizer(reading_direction="rtl")
        bubbles = [
            _FakeBubble((100, 100, 50, 50)),
            _FakeBubble((160, 100, 50, 50)),
        ]
        panels = opt.detect_panels((800, 600), bubbles)
        assert len(panels) >= 1
        all_indices = []
        for p in panels:
            all_indices.extend(p.bubble_indices)
        assert set(all_indices) == {0, 1}

    def test_two_rows_two_panels(self):
        opt = ReadingOrderOptimizer(reading_direction="rtl")
        bubbles = [
            _FakeBubble((100, 50, 50, 40)),
            _FakeBubble((100, 300, 50, 40)),
        ]
        panels = opt.detect_panels((600, 400), bubbles)
        assert len(panels) == 2

    def test_panel_has_bbox(self):
        opt = ReadingOrderOptimizer(reading_direction="rtl")
        bubbles = [_FakeBubble((100, 100, 50, 50))]
        panels = opt.detect_panels((600, 400), bubbles)
        assert len(panels) == 1
        assert len(panels[0].bbox) == 4


class TestInvalidInput:
    def test_invalid_direction(self):
        with pytest.raises(ValueError, match="reading_direction"):
            ReadingOrderOptimizer(reading_direction="btt")
