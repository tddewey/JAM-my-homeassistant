"""Full-screen scene detection for the NBA Jam cabinet (team select, match-up, game over...).

Each non-gameplay screen has a fixed title (or centre graphic) that does not move,
so a Pearson correlation of that crop against a stored reference is enough. All work
is done on a 480x270 grayscale copy of the frame, which keeps it to a few milliseconds.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

SCENE_SIZE = (480, 270)
# Regions on the 480x270 frame: (rows, cols)
TITLE = (slice(4, 34), slice(150, 330))
CENTER = (slice(90, 200), slice(150, 330))

# prototype name -> (scene label, region). The title blinks between "PRESS SHOOT TO
# SELECT" and "PRESS TURBO TO SWAP", so team select has two prototypes.
SCENE_SPECS: Dict[str, Tuple[str, Tuple[slice, slice]]] = {
    'team_select_a': ('team_select', TITLE),
    'team_select_b': ('team_select', TITLE),
    'matchup': ('matchup', TITLE),
    'final_stats': ('final_stats', TITLE),
    'final_score': ('final_score', TITLE),
    'game_over': ('game_over', CENTER),
    # Look-alikes that must NOT be mistaken for the scenes above
    'substitutions': ('substitutions', TITLE),
    'half_stats': ('half_stats', TITLE),
}

MIN_CORRELATION = 0.8
BLACK_MEAN = 5.0          # whole-frame mean gray below this is the black transition
LIVE_COURT_MEAN = 30.0    # lower-frame mean gray above this means gameplay is visible
WIN_FRACTION = 0.18       # share of magenta pixels in the upper-middle for "X WINS!"


def _gray_small(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, SCENE_SIZE, interpolation=cv2.INTER_AREA)
    return small, cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def crop_prototype(frame: np.ndarray, name: str) -> np.ndarray:
    """Cut the reference crop for prototype `name` from a full BGR frame."""
    _, gray = _gray_small(frame)
    rows, cols = SCENE_SPECS[name][1]
    return gray[rows, cols].copy()


class SceneDetector:
    """Labels a frame with one scene name, or None if it matches no known screen."""

    def __init__(self, prototypes: Dict[str, np.ndarray]):
        missing = set(SCENE_SPECS) - set(prototypes)
        if missing:
            raise ValueError(f"scene prototypes missing: {sorted(missing)}")
        self.prototypes = prototypes

    @classmethod
    def load(cls, path: str) -> "SceneDetector":
        with np.load(path) as npz:
            return cls({k: npz[k] for k in npz.files})

    @staticmethod
    def save(prototypes: Dict[str, np.ndarray], path: str) -> None:
        np.savez_compressed(path, **prototypes)

    def classify(self, frame: np.ndarray) -> Optional[str]:
        """Return the scene label for a full-size BGR frame.

        Labels: team_select, matchup, final_stats, final_score, game_over,
        substitutions, half_stats (title screens), win (X WINS! graphic),
        black (transition), live (gameplay visible), or None.
        """
        small, gray = _gray_small(frame)

        best_label, best = None, MIN_CORRELATION
        for name, (label, (rows, cols)) in SCENE_SPECS.items():
            c = _corr(gray[rows, cols], self.prototypes[name])
            if c > best:
                best_label, best = label, c
        if best_label:
            return best_label

        if gray.mean() < BLACK_MEAN:
            return 'black'

        hsv = cv2.cvtColor(small[30:130, 120:360], cv2.COLOR_BGR2HSV)
        if cv2.inRange(hsv, (135, 140, 120), (165, 255, 255)).mean() / 255 > WIN_FRACTION:
            return 'win'

        if gray[150:260].mean() > LIVE_COURT_MEAN:
            return 'live'
        return None
