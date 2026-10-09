#!/usr/bin/env python3
"""Pre-fill a labels CSV for evaluate.py from the mask recognizer, for hand correction.

Usage: python prefill_labels.py FRAME_DIR --out labels.csv [--root DIR]
The 'frame' column is relative to --root (default: the CSV's directory).
"""

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import cv2

from config import Config
from evaluate import MaskRecognizer

_rec = None


def _init(config_path):
    global _rec
    _rec = MaskRecognizer(Config.load(config_path).detection)


def _label(path):
    frame = cv2.imread(path)
    if not _rec.base.has_purple_band(frame):
        return path, 0, None, None, None
    p1, p2, q = _rec.read(frame)
    return path, 1, p1, p2, q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('frame_dir')
    parser.add_argument('--out', required=True)
    parser.add_argument('--root', default=None)
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()

    out = Path(args.out)
    root = Path(args.root) if args.root else out.parent
    paths = sorted(str(p) for p in Path(args.frame_dir).glob('*.png'))
    with Pool(initializer=_init, initargs=(args.config,)) as pool:
        results = pool.map(_label, paths, chunksize=8)

    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame', 'overlay', 'p1', 'p2', 'quarter'])
        for path, overlay, p1, p2, q in results:
            rel = Path(path).resolve().relative_to(root.resolve())
            w.writerow([rel, overlay, '' if p1 is None else p1, '' if p2 is None else p2, q or ''])
    print(f"{sum(r[1] for r in results)} overlay frames of {len(results)} -> {out}")


if __name__ == '__main__':
    main()
