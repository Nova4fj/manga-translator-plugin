"""Region exclusion mask for skipping page areas during detection."""

import numpy as np
from typing import List, Tuple


def parse_exclusion_regions(regions_str: str) -> List[Tuple[int, int, int, int]]:
    """Parse semicolon-separated exclusion regions.

    Format: 'x,y,w,h;x,y,w,h;...'
    Each region is defined as (x, y, width, height).

    Args:
        regions_str: Semicolon-separated region strings, e.g. '0,0,100,50;200,300,100,50'.

    Returns:
        List of (x, y, w, h) tuples.
    """
    if not regions_str:
        return []
    regions = []
    for part in regions_str.split(";"):
        coords = part.strip().split(",")
        if len(coords) == 4:
            regions.append(tuple(int(c.strip()) for c in coords))
    return regions


def create_exclusion_mask(
    height: int, width: int, regions: List[Tuple[int, int, int, int]]
) -> np.ndarray:
    """Create a binary mask where 0 = excluded, 255 = included.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        regions: List of (x, y, w, h) rectangles to exclude.

    Returns:
        Single-channel uint8 mask (255 for included, 0 for excluded).
    """
    mask = np.ones((height, width), dtype=np.uint8) * 255
    for x, y, w, h in regions:
        mask[y : y + h, x : x + w] = 0
    return mask


def filter_bubbles_by_mask(bubbles, mask: np.ndarray):
    """Remove bubbles whose center falls in an excluded region.

    Args:
        bubbles: List of BubbleRegion objects (must have .bbox attribute).
        mask: Exclusion mask from create_exclusion_mask (0 = excluded).

    Returns:
        Filtered list of BubbleRegion objects.
    """
    filtered = []
    for b in bubbles:
        bx, by, bw, bh = b.bbox
        cx, cy = bx + bw // 2, by + bh // 2
        if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]:
            if mask[cy, cx] > 0:
                filtered.append(b)
    return filtered
