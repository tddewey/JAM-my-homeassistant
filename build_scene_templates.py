#!/usr/bin/env python3
"""Build templates/scenes.npz from known-good frames of the recorded games.

Frames come from the 480x270 cache (dataset/cache, written by cache_frames.py),
chosen by (game, elapsed second). To rebuild for a different cabinet, update
PICKS from your own recordings.
"""

import numpy as np

from detectors.scene_detector import SCENE_SPECS, SceneDetector, crop_prototype

# prototype -> (game, elapsed seconds) of a clean frame showing that screen
PICKS = {
    'team_select_a': ('game1', 63),      # "PRESS SHOOT TO SELECT"
    'team_select_b': ('game1', 60),      # "PRESS TURBO TO SWAP"
    'matchup': ('game1', 81),
    'final_stats': ('game1', 611),
    'final_score': ('game2', 563.7),
    'game_over': ('game2', 560.1),
    'substitutions': ('game1', 335),
    'half_stats': ('game1', 317),
    'get_ready': ('game3', 184.1),       # "GET READY FOR 2ND QUARTER"
}


def main():
    import cv2
    cache = {}
    protos = {}
    for name, (game, t) in PICKS.items():
        if game not in cache:
            cache[game] = (np.load(f"dataset/cache/{game}_480.npy", mmap_mode='r'),
                           np.array([float(x) for x in open(f"dataset/cache/{game}_times.txt").read().split()]))
        frames, times = cache[game]
        frame = np.asarray(frames[int(np.argmin(abs(times - t)))])
        # crop_prototype works on any size frame (it resizes), so the 480x270 cache is fine
        protos[name] = crop_prototype(frame, name)
    assert set(protos) == set(SCENE_SPECS)
    SceneDetector.save(protos, 'templates/scenes.npz')
    print("wrote templates/scenes.npz:", sorted(protos))


if __name__ == '__main__':
    main()
