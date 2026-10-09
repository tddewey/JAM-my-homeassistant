"""Template-matching recognizer for the NBA Jam overlay (fixed font, fixed layout).

Glyphs are cut out of the scanline-bridged text mask (BaseOCRDetector.text_mask),
normalised to a fixed size and compared by Pearson correlation against templates
learned from labeled frames. Digits are matched one glyph at a time; the quarter
text is matched as a whole phrase.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import DetectionConfig
from detectors.base_ocr_detector import BaseOCRDetector

GLYPH_SIZE = (28, 44)       # (width, height) every digit glyph is resized to
PHRASE_SIZE = (240, 36)     # (width, height) the quarter text is resized to
MIN_GLYPH_H = 35          # digits are ~40x55 px at 1080p


def _normalise(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Crop to the mask's bounding box, resize to `size`, return float32 0..1."""
    ys, xs = np.nonzero(mask)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two equal-shaped arrays (-1..1)."""
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.0


# Score digits are right-aligned in fixed slots ~52 px apart (1080p). Slot bounds are
# offsets from the right edge of the score region: (left, right), ones digit first.
# Measured on the CRT-shader output; hundreds slot is extrapolated (no 3-digit scores seen).
SLOT_OFFSETS = [(-70, -15), (-123, -71), (-175, -124)]
MIN_GLYPH_AREA = 500


def segment_glyphs(mask: np.ndarray) -> List[np.ndarray]:
    """Split a score mask into per-digit masks, ordered left to right.

    Digits touch each other too often for connected components to separate
    them, so each fixed slot is inspected on its own: the largest component
    in the slot is the glyph, and an empty slot ends the number.
    """
    h, w = mask.shape
    glyphs = []
    for left, right in SLOT_OFFSETS:
        x0, x1 = max(0, w + left), w + right
        if x1 <= x0:
            break
        slot = mask[:, x0:x1]
        count, labels, stats, _ = cv2.connectedComponentsWithStats(slot, connectivity=8)
        best, best_area = None, 0
        for i in range(1, count):
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            if bh >= MIN_GLYPH_H and area > best_area:
                best, best_area = i, area
        if best is None or best_area < MIN_GLYPH_AREA:
            break
        glyphs.append(((labels == best).astype(np.uint8)) * 255)
    glyphs.reverse()
    return glyphs


class TemplateLibrary:
    """Labeled glyph/phrase templates with nearest-neighbour matching."""

    def __init__(self):
        self.digits: Dict[str, List[np.ndarray]] = {str(d): [] for d in range(10)}
        self.phrases: Dict[str, List[np.ndarray]] = {}

    # ---- building ----
    def add_digit(self, digit: str, glyph_mask: np.ndarray) -> None:
        self.digits[digit].append(_normalise(glyph_mask, GLYPH_SIZE))

    def add_phrase(self, phrase: str, mask: np.ndarray) -> None:
        if mask.any():
            self.phrases.setdefault(phrase, []).append(_normalise(mask, PHRASE_SIZE))

    # ---- matching ----
    def classify_digit(self, glyph_mask: np.ndarray) -> Tuple[Optional[str], float]:
        """Best digit for a glyph and its correlation (None if no templates)."""
        g = _normalise(glyph_mask, GLYPH_SIZE)
        best, best_score = None, -2.0
        for digit, templates in self.digits.items():
            for t in templates:
                s = _corr(g, t)
                if s > best_score:
                    best, best_score = digit, s
        return best, best_score

    def classify_phrase(self, mask: np.ndarray) -> Tuple[Optional[str], float]:
        if not mask.any() or not self.phrases:
            return None, -2.0
        g = _normalise(mask, PHRASE_SIZE)
        best, best_score = None, -2.0
        for phrase, templates in self.phrases.items():
            for t in templates:
                s = _corr(g, t)
                if s > best_score:
                    best, best_score = phrase, s
        return best, best_score

    # ---- persistence ----
    def save(self, path: str) -> None:
        data = {f"digit_{d}": np.stack(v) for d, v in self.digits.items() if v}
        data.update({f"phrase_{p}": np.stack(v) for p, v in self.phrases.items()})
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path: str) -> "TemplateLibrary":
        lib = cls()
        with np.load(path) as npz:
            for key in npz.files:
                kind, name = key.split('_', 1)
                arr = list(npz[key])
                if kind == 'digit':
                    lib.digits[name] = arr
                else:
                    lib.phrases[name] = arr
        return lib


class TemplateRecognizer:
    """Reads both scores and the quarter text from a BGR frame using a TemplateLibrary."""

    MIN_DIGIT_SCORE = 0.5     # correlation below this means "not a digit"
    # Third (hundreds) glyph must match far better: stray background blobs sometimes
    # land in that slot, and 3-digit scores are rare
    MIN_HUNDREDS_SCORE = 0.85
    MIN_PHRASE_SCORE = 0.5

    def __init__(self, det_cfg: DetectionConfig, library: TemplateLibrary):
        self.cfg = det_cfg
        self.lib = library
        self.base = BaseOCRDetector(det_cfg)

    @staticmethod
    def _crop(frame, r):
        return frame[r.y:r.y + r.height, r.x:r.x + r.width]

    def score_glyphs(self, frame: np.ndarray, player: str) -> List[np.ndarray]:
        mask = self.base.text_mask(self._crop(frame, self.cfg.score_regions[player]))
        return segment_glyphs(mask) if mask is not None else []

    def read_score(self, frame: np.ndarray, player: str) -> Optional[int]:
        digits = []
        glyphs = self.score_glyphs(frame, player)
        for i, g in enumerate(glyphs):
            d, s = self.lib.classify_digit(g)
            is_hundreds = len(glyphs) == 3 and i == 0
            if d is None or s < (self.MIN_HUNDREDS_SCORE if is_hundreds else self.MIN_DIGIT_SCORE):
                if is_hundreds:
                    continue          # treat as background noise, keep the two real digits
                return None
            digits.append(d)
        return int(''.join(digits)) if digits else None

    def quarter_mask(self, frame: np.ndarray) -> Optional[np.ndarray]:
        return self.base.text_mask(self._crop(frame, self.cfg.quarter_text_region), min_area=40)

    def read_quarter(self, frame: np.ndarray) -> Optional[str]:
        mask = self.quarter_mask(frame)
        if mask is None:
            return None
        phrase, score = self.lib.classify_phrase(mask)
        return phrase if score >= self.MIN_PHRASE_SCORE else None

    def read(self, frame: np.ndarray):
        """(player1, player2, quarter); all None when the overlay is not on screen."""
        if not self.base.has_purple_band(frame):
            return None, None, None
        return self.read_score(frame, 'player1'), self.read_score(frame, 'player2'), self.read_quarter(frame)
