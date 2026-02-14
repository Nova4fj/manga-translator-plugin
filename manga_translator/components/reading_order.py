"""Panel-aware reading order for manga bubbles.

Clusters bubbles into panels and sorts them in manga reading order
(right-to-left, top-to-bottom) or manhwa order (left-to-right, top-to-bottom).
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Panel:
    """A detected panel containing speech bubbles."""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    bubble_indices: List[int] = field(default_factory=list)


class ReadingOrderOptimizer:
    """Sorts bubbles into culturally-correct reading order.

    Manga (Japanese): right-to-left, top-to-bottom.
    Manhwa (Korean) / Manhua (Chinese): left-to-right, top-to-bottom.
    """

    def __init__(
        self,
        reading_direction: str = "rtl",
        panel_gap_ratio: float = 0.15,
    ):
        if reading_direction not in ("rtl", "ltr"):
            raise ValueError(f"reading_direction must be 'rtl' or 'ltr', got {reading_direction!r}")
        self.reading_direction = reading_direction
        self.panel_gap_ratio = panel_gap_ratio

    def detect_panels(
        self,
        page_shape: Tuple[int, int],
        bubbles: List,
    ) -> List[Panel]:
        """Cluster bubbles into panels based on spatial proximity."""
        if not bubbles:
            return []

        h, w = page_shape
        gap_threshold = w * self.panel_gap_ratio

        indexed = [(i, b) for i, b in enumerate(bubbles)]
        indexed.sort(key=lambda ib: ib[1].center[1])

        # Cluster into rows by Y proximity
        rows: List[List[Tuple[int, object]]] = []
        current_row: List[Tuple[int, object]] = [indexed[0]]

        for i in range(1, len(indexed)):
            idx, bubble = indexed[i]
            prev_idx, prev_bubble = current_row[-1]
            _, prev_y, _, prev_h = prev_bubble.bbox
            _, cur_y, _, cur_h = bubble.bbox

            vertical_gap = cur_y - (prev_y + prev_h)
            row_height = max(prev_h, cur_h)
            if vertical_gap < row_height * 0.5:
                current_row.append((idx, bubble))
            else:
                rows.append(current_row)
                current_row = [(idx, bubble)]
        rows.append(current_row)

        # Within each row, sort and cluster into panels by X proximity
        panels: List[Panel] = []
        for row in rows:
            if self.reading_direction == "rtl":
                row.sort(key=lambda ib: -ib[1].center[0])
            else:
                row.sort(key=lambda ib: ib[1].center[0])

            current_panel_bubbles = [row[0]]
            for j in range(1, len(row)):
                idx, bubble = row[j]
                prev_idx, prev_bubble = current_panel_bubbles[-1]
                px, _, pw, _ = prev_bubble.bbox
                cx, _, cw, _ = bubble.bbox

                if self.reading_direction == "rtl":
                    h_gap = px - (cx + cw)
                else:
                    h_gap = cx - (px + pw)

                if abs(h_gap) < gap_threshold:
                    current_panel_bubbles.append((idx, bubble))
                else:
                    panels.append(self._make_panel(current_panel_bubbles))
                    current_panel_bubbles = [(idx, bubble)]
            panels.append(self._make_panel(current_panel_bubbles))

        return panels

    def sort_bubbles(self, bubbles: List) -> List:
        """Sort bubbles into reading order."""
        if len(bubbles) <= 1:
            return list(bubbles)

        max_x = max(b.bbox[0] + b.bbox[2] for b in bubbles)
        max_y = max(b.bbox[1] + b.bbox[3] for b in bubbles)
        page_shape = (max_y + 100, max_x + 100)

        panels = self.detect_panels(page_shape, bubbles)

        ordered_indices = []
        for panel in panels:
            ordered_indices.extend(panel.bubble_indices)

        all_indices = set(range(len(bubbles)))
        missing = all_indices - set(ordered_indices)
        ordered_indices.extend(sorted(missing))

        return [bubbles[i] for i in ordered_indices]

    def _make_panel(self, indexed_bubbles: List[Tuple[int, object]]) -> Panel:
        """Create a Panel from a list of (index, bubble) pairs."""
        indices = [i for i, _ in indexed_bubbles]
        bboxes = [b.bbox for _, b in indexed_bubbles]

        x_min = min(bx for bx, _, _, _ in bboxes)
        y_min = min(by for _, by, _, _ in bboxes)
        x_max = max(bx + bw for bx, _, bw, _ in bboxes)
        y_max = max(by + bh for _, by, _, bh in bboxes)

        return Panel(
            bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
            bubble_indices=indices,
        )
