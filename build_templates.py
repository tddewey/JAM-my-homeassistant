#!/usr/bin/env python3
"""Build a glyph/phrase template library from labeled frames.

Usage: python build_templates.py --labels dataset/game1/labels.csv --root dataset \
           --out templates.npz [--min-time T] [--max-time T]

Frames whose segmented glyph count does not match the label's digit count are
skipped (reported), so occasional segmentation failures cannot poison the library.
Frame filenames are elapsed seconds (NNNNN.NN.png), which --min-time/--max-time use.
"""

import argparse
import csv
from pathlib import Path

import cv2

from config import Config
from detectors.template_matcher import TemplateLibrary, TemplateRecognizer


def frame_time(frame_path: str) -> float:
    return float(Path(frame_path).stem)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', required=True)
    parser.add_argument('--root', default=None)
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--min-time', type=float, default=None)
    parser.add_argument('--max-time', type=float, default=None)
    parser.add_argument('--split', choices=['even', 'odd'], default=None,
                        help='Only use frames whose 20-second time bucket is even/odd (for held-out testing)')
    parser.add_argument('--per-class', type=int, default=0,
                        help='Max templates kept per digit/phrase (0 = all). Evenly subsampled.')
    args = parser.parse_args()

    labels = Path(args.labels)
    root = Path(args.root) if args.root else labels.parent
    cfg = Config.load(args.config).detection
    rec = TemplateRecognizer(cfg, TemplateLibrary())

    rows = [r for r in csv.DictReader(open(labels)) if r['overlay'].strip() == '1']
    if args.min_time is not None:
        rows = [r for r in rows if frame_time(r['frame']) >= args.min_time]
    if args.max_time is not None:
        rows = [r for r in rows if frame_time(r['frame']) < args.max_time]

    if args.split:
        want = 0 if args.split == 'even' else 1
        rows = [r for r in rows if int(frame_time(r['frame']) // 20) % 2 == want]

    pending = {}   # class -> list of masks
    skipped = 0
    for row in rows:
        frame = cv2.imread(str(root / row['frame']))
        for player, key in (('player1', 'p1'), ('player2', 'p2')):
            text = row[key].strip()
            glyphs = rec.score_glyphs(frame, player)
            if len(glyphs) != len(text):
                skipped += 1
                continue
            for ch, g in zip(text, glyphs):
                pending.setdefault(('digit', ch), []).append(g)
        q = rec.quarter_mask(frame)
        if q is not None and row['quarter'].strip():
            pending.setdefault(('phrase', row['quarter'].strip()), []).append(q)

    lib = TemplateLibrary()
    for (kind, name), masks in sorted(pending.items()):
        if args.per_class and len(masks) > args.per_class:
            step = len(masks) / args.per_class
            masks = [masks[int(i * step)] for i in range(args.per_class)]
        for m in masks:
            (lib.add_digit if kind == 'digit' else lib.add_phrase)(name, m)
    lib.save(args.out)

    print(f"{len(rows)} frames used, {skipped} score reads skipped (glyph count mismatch)")
    print("digits:", {d: len(v) for d, v in lib.digits.items()})
    print("phrases:", {p: len(v) for p, v in lib.phrases.items()})
    print(f"-> {args.out}")


if __name__ == '__main__':
    main()
