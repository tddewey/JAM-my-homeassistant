"""Cache a recorded game as 480x270 BGR frames (dataset/cache/GAME_480.npy + GAME_times.txt).

Usage: python cache_frames.py game1 game2 game3
Frames are read from dataset/GAME/*.png or dataset/GAME/*/*.png.
"""
import glob
import sys

import cv2
import numpy as np
from multiprocessing import Pool


def load(f):
    return cv2.resize(cv2.imread(f), (480, 270), interpolation=cv2.INTER_AREA)


if __name__ == '__main__':
    for g in sys.argv[1:] or ('game1', 'game2'):
        fs = sorted(glob.glob(f"dataset/{g}/*.png") + glob.glob(f"dataset/{g}/*/*.png"),
                    key=lambda f: float(f.split('/')[-1][:-4]))
        with Pool(6) as p:
            arr = np.stack(p.map(load, fs, chunksize=8))
        np.save(f"dataset/cache/{g}_480.npy", arr)
        with open(f"dataset/cache/{g}_times.txt", "w") as o:
            o.write("\n".join(f.split('/')[-1][:-4] for f in fs))
        print(g, arr.shape, flush=True)
