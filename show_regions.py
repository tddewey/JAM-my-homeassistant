#!/usr/bin/env python3
"""Draw the configured detection regions on one or more frames for checking.

Usage: python show_regions.py frame.png [frame2.png ...] [--out DIR]
Writes <name>_regions.png next to each frame (or into --out) and prints the
purple-band ratio for each so the gate can be checked at a glance.
"""

import argparse
from pathlib import Path

import cv2

from config import Config
from detectors.base_ocr_detector import BaseOCRDetector

COLORS = {
    'player1': (0, 255, 0),
    'player2': (0, 200, 255),
    'quarter': (255, 128, 0),
    'purple band': (255, 0, 255),
    'team heading': (0, 0, 255),
}


def main():
    parser = argparse.ArgumentParser(description="Overlay detection regions on frames")
    parser.add_argument('frames', nargs='+')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    det = Config.load(args.config).detection
    helper = BaseOCRDetector(det)
    regions = {f'player{i}': r for i, r in ((1, det.score_regions.get('player1')), (2, det.score_regions.get('player2')))}
    regions.update({
        'quarter': det.quarter_text_region,
        'purple band': det.purple_band_region,
        'team heading': det.team_selection_heading_region,
    })

    for path in args.frames:
        frame = cv2.imread(path)
        if frame is None:
            print(f"{path}: could not read")
            continue
        ratio = helper.purple_ratio(frame)
        out = frame.copy()
        for name, r in regions.items():
            if r is None:
                continue
            color = COLORS[name]
            cv2.rectangle(out, (r.x, r.y), (r.x + r.width, r.y + r.height), color, 2)
            cv2.putText(out, name, (r.x, max(15, r.y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        dest_dir = Path(args.out) if args.out else Path(path).parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{Path(path).stem}_regions.png"
        cv2.imwrite(str(dest), out)
        print(f"{path}: purple={'n/a' if ratio is None else f'{ratio:.2f}'} gate={helper.has_purple_band(frame)} -> {dest}")


if __name__ == '__main__':
    main()
