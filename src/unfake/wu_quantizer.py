"""
Based on Xiaolin Wu's "Color Quantization by Dynamic Programming and Principal Analysis" (1992)
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("unfake.py")


@dataclass
class Box:
    """Represents a box in RGB color space"""

    r_min: int
    r_max: int
    g_min: int
    g_max: int
    b_min: int
    b_max: int
    volume: int = 0


class WuQuantizer:
    """
    Wu's color quantization algorithm implementation.
    This algorithm recursively subdivides the RGB color cube into smaller boxes,
    maximizing the variance of colors within each box.
    """

    pixels: list[tuple[int, int, int]]
    boxes: list[Box]

    def __init__(self, max_colors: int = 256, significant_bits: int = 5) -> None:
        """
        Initialize Wu quantizer

        Args:
            max_colors: Maximum number of colors in the palette
            significant_bits: Number of significant bits per channel (5 = 32 levels per channel)
        """
        # Lookup table for squared values
        self.sqr_table = np.array([i * i for i in range(256)], dtype=np.int32)
        self._reset(max_colors, significant_bits)

    def _reset(self, max_colors: int, significant_bits: int) -> None:
        """Reset the quantizer"""
        self.max_colors = max_colors
        self.significant_bits = significant_bits
        self.side_size = 1 << significant_bits  # 2^bits
        self.max_side_index = self.side_size - 1

        self.weights = np.zeros((self.side_size, self.side_size, self.side_size), dtype=np.int64)
        self.moments_r = np.zeros((self.side_size, self.side_size, self.side_size), dtype=np.int64)
        self.moments_g = np.zeros((self.side_size, self.side_size, self.side_size), dtype=np.int64)
        self.moments_b = np.zeros((self.side_size, self.side_size, self.side_size), dtype=np.int64)
        self.moments = np.zeros((self.side_size, self.side_size, self.side_size), dtype=np.float64)
        self.pixels = []
        self.boxes = []

    def _hist_index(self, value: int) -> int:
        """Convert 8-bit value to histogram index"""
        return value >> (8 - self.significant_bits)

    def add_pixels(self, pixels: np.ndarray) -> None:
        """
        Add pixels to the histogram

        Args:
            pixels: Array of RGB(A) pixels
        """
        for pixel in pixels:
            if len(pixel) >= 4 and pixel[3] < 128:  # Skip transparent pixels
                continue

            r, g, b = pixel[0], pixel[1], pixel[2]
            ir = self._hist_index(r)
            ig = self._hist_index(g)
            ib = self._hist_index(b)

            self.weights[ir, ig, ib] += 1
            self.moments_r[ir, ig, ib] += r
            self.moments_g[ir, ig, ib] += g
            self.moments_b[ir, ig, ib] += b
            self.moments[ir, ig, ib] += self.sqr_table[r] + self.sqr_table[g] + self.sqr_table[b]

            self.pixels.append((r, g, b))

    def _compute_cumulative_moments(self) -> None:
        """Convert histogram to cumulative moments"""
        dim = self.side_size

        # Build 3D cumulative sums
        for r in range(dim):
            area_r = np.zeros(dim)
            area_g = np.zeros(dim)
            area_b = np.zeros(dim)
            area_w = np.zeros(dim)
            area_m = np.zeros(dim)

            for g in range(dim):
                line_r = 0
                line_g = 0
                line_b = 0
                line_w = 0
                line_m = 0.0

                for b in range(dim):
                    line_w += self.weights[r, g, b]
                    line_r += self.moments_r[r, g, b]
                    line_g += self.moments_g[r, g, b]
                    line_b += self.moments_b[r, g, b]
                    line_m += self.moments[r, g, b]

                    area_w[b] += line_w
                    area_r[b] += line_r
                    area_g[b] += line_g
                    area_b[b] += line_b
                    area_m[b] += line_m

                    if r > 0:
                        self.weights[r, g, b] = self.weights[r - 1, g, b] + area_w[b]
                        self.moments_r[r, g, b] = self.moments_r[r - 1, g, b] + area_r[b]
                        self.moments_g[r, g, b] = self.moments_g[r - 1, g, b] + area_g[b]
                        self.moments_b[r, g, b] = self.moments_b[r - 1, g, b] + area_b[b]
                        self.moments[r, g, b] = self.moments[r - 1, g, b] + area_m[b]
                    else:
                        self.weights[r, g, b] = area_w[b]
                        self.moments_r[r, g, b] = area_r[b]
                        self.moments_g[r, g, b] = area_g[b]
                        self.moments_b[r, g, b] = area_b[b]
                        self.moments[r, g, b] = area_m[b]

    def _volume(self, box: Box, moment: np.ndarray) -> float:
        """Compute sum over a box of any given moment"""
        return (  # type: ignore[no-any-return]
            moment[box.r_max, box.g_max, box.b_max]
            - moment[box.r_max, box.g_max, box.b_min]
            - moment[box.r_max, box.g_min, box.b_max]
            + moment[box.r_max, box.g_min, box.b_min]
            - moment[box.r_min, box.g_max, box.b_max]
            + moment[box.r_min, box.g_max, box.b_min]
            + moment[box.r_min, box.g_min, box.b_max]
            - moment[box.r_min, box.g_min, box.b_min]
        )

    def _bottom(self, box: Box, direction: int, moment: np.ndarray) -> float:
        """Compute part of volume sum for bottom of box"""
        if direction == 0:  # Red
            return -self._volume(
                Box(box.r_min, box.r_min, box.g_min, box.g_max, box.b_min, box.b_max), moment
            )
        elif direction == 1:  # Green
            return -self._volume(
                Box(box.r_min, box.r_max, box.g_min, box.g_min, box.b_min, box.b_max), moment
            )
        else:  # Blue
            return -self._volume(
                Box(box.r_min, box.r_max, box.g_min, box.g_max, box.b_min, box.b_min), moment
            )

    def _top(self, box: Box, direction: int, position: int, moment: np.ndarray) -> float:
        """Compute part of volume sum for top of box"""
        if direction == 0:  # Red
            return self._volume(
                Box(position, box.r_max, box.g_min, box.g_max, box.b_min, box.b_max), moment
            )
        elif direction == 1:  # Green
            return self._volume(
                Box(box.r_min, box.r_max, position, box.g_max, box.b_min, box.b_max), moment
            )
        else:  # Blue
            return self._volume(
                Box(box.r_min, box.r_max, box.g_min, box.g_max, position, box.b_max), moment
            )

    def _variance(self, box: Box) -> float:
        """Compute the weighted variance of a box"""
        dr = self._volume(box, self.moments_r)
        dg = self._volume(box, self.moments_g)
        db = self._volume(box, self.moments_b)
        dm = self._volume(box, self.moments)
        dw = self._volume(box, self.weights)

        if dw == 0:
            return 0.0

        return dm - (dr * dr + dg * dg + db * db) / dw

    def _maximize(
        self,
        box: Box,
        direction: int,
        first: int,
        last: int,
        whole_r: float,
        whole_g: float,
        whole_b: float,
        whole_w: float,
    ) -> tuple[float, int]:
        """Find the optimal split position that maximizes variance"""
        bottom_r = self._bottom(box, direction, self.moments_r)
        bottom_g = self._bottom(box, direction, self.moments_g)
        bottom_b = self._bottom(box, direction, self.moments_b)
        bottom_w = self._bottom(box, direction, self.weights)

        max_variance = 0.0
        cut_position = -1

        for i in range(first, last):
            half_r = bottom_r + self._top(box, direction, i, self.moments_r)
            half_g = bottom_g + self._top(box, direction, i, self.moments_g)
            half_b = bottom_b + self._top(box, direction, i, self.moments_b)
            half_w = bottom_w + self._top(box, direction, i, self.weights)

            if half_w == 0:
                continue

            temp = (half_r * half_r + half_g * half_g + half_b * half_b) / half_w

            half_r = whole_r - half_r
            half_g = whole_g - half_g
            half_b = whole_b - half_b
            half_w = whole_w - half_w

            if half_w == 0:
                continue

            temp += (half_r * half_r + half_g * half_g + half_b * half_b) / half_w

            if temp > max_variance:
                max_variance = temp
                cut_position = i

        return max_variance, cut_position

    def _cut(self, box1: Box, box2: Box) -> bool:
        """Cut box1 into box1 and box2"""
        whole_r = self._volume(box1, self.moments_r)
        whole_g = self._volume(box1, self.moments_g)
        whole_b = self._volume(box1, self.moments_b)
        whole_w = self._volume(box1, self.weights)

        max_r, cut_r = self._maximize(
            box1, 0, box1.r_min + 1, box1.r_max, whole_r, whole_g, whole_b, whole_w
        )
        max_g, cut_g = self._maximize(
            box1, 1, box1.g_min + 1, box1.g_max, whole_r, whole_g, whole_b, whole_w
        )
        max_b, cut_b = self._maximize(
            box1, 2, box1.b_min + 1, box1.b_max, whole_r, whole_g, whole_b, whole_w
        )

        # Choose direction with maximum variance
        if max_r >= max_g and max_r >= max_b:
            direction = 0
            if cut_r < 0:
                return False
        elif max_g >= max_r and max_g >= max_b:
            direction = 1
        else:
            direction = 2

        # Copy box1 to box2
        box2.r_min = box1.r_min
        box2.g_min = box1.g_min
        box2.b_min = box1.b_min
        box2.r_max = box1.r_max
        box2.g_max = box1.g_max
        box2.b_max = box1.b_max

        # Do the cut
        if direction == 0:  # Red
            box2.r_min = box1.r_max = cut_r
        elif direction == 1:  # Green
            box2.g_min = box1.g_max = cut_g
        else:  # Blue
            box2.b_min = box1.b_max = cut_b

        # Update volumes
        box1.volume = (
            (box1.r_max - box1.r_min) * (box1.g_max - box1.g_min) * (box1.b_max - box1.b_min)
        )
        box2.volume = (
            (box2.r_max - box2.r_min) * (box2.g_max - box2.g_min) * (box2.b_max - box2.b_min)
        )

        return True

    def quantize(self, pixels: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
        """
        Perform color quantization on the given pixels

        Args:
            pixels: Input image as numpy array

        Returns:
            Quantized image and color palette
        """
        # Reset state
        self._reset(self.max_colors, self.significant_bits)

        # Build histogram
        h, w = pixels.shape[:2]
        self.add_pixels(pixels.reshape(-1, pixels.shape[2]))

        # Compute cumulative moments
        self._compute_cumulative_moments()

        # Initialize first box
        self.boxes = []
        for i in range(self.max_colors):
            self.boxes.append(Box(0, 0, 0, 0, 0, 0))

        self.boxes[0].r_max = self.max_side_index
        self.boxes[0].g_max = self.max_side_index
        self.boxes[0].b_max = self.max_side_index

        # Perform cuts
        n_boxes = 1
        next_box = 0

        for i in range(1, self.max_colors):
            if self._cut(self.boxes[next_box], self.boxes[i]):
                volume_variance = (
                    self._variance(self.boxes[next_box]) if self.boxes[next_box].volume > 1 else 0
                )
                self.boxes[next_box].volume = int(volume_variance)

                volume_variance = self._variance(self.boxes[i]) if self.boxes[i].volume > 1 else 0
                self.boxes[i].volume = int(volume_variance)
                n_boxes += 1
            else:
                self.boxes[next_box].volume = 0

            # Find box with largest variance for next cut
            next_box = 0
            temp = self.boxes[0].volume

            for j in range(1, i + 1):
                if self.boxes[j].volume > temp:
                    temp = self.boxes[j].volume
                    next_box = j

            if temp <= 0:
                n_boxes = i + 1
                break

        # Generate palette from boxes
        palette = []
        for i in range(n_boxes):
            weight = self._volume(self.boxes[i], self.weights)
            if weight > 0:
                r = int(self._volume(self.boxes[i], self.moments_r) / weight)
                g = int(self._volume(self.boxes[i], self.moments_g) / weight)
                b = int(self._volume(self.boxes[i], self.moments_b) / weight)
                palette.append((r, g, b))

        # Map pixels to palette colors
        quantized = np.zeros_like(pixels)

        for y in range(h):
            for x in range(w):
                if pixels.shape[2] >= 4 and pixels[y, x, 3] < 128:
                    quantized[y, x] = [0, 0, 0, 0] if pixels.shape[2] == 4 else [0, 0, 0]
                else:
                    # Find nearest palette color
                    pixel = pixels[y, x, :3]
                    min_dist = float("inf")
                    best_idx = 0

                    for idx, color in enumerate(palette):
                        dist: float = np.sum((pixel - np.array(color)) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            best_idx = idx

                    quantized[y, x, :3] = palette[best_idx]
                    if pixels.shape[2] == 4:
                        quantized[y, x, 3] = 255

        return quantized.astype(np.uint8), palette
