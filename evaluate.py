#!/usr/bin/env python3
"""Evaluate recognizers against labeled frames.

Labels CSV columns: frame,overlay,p1,p2,quarter
  frame    path relative to --root
  overlay  1 if the purple score/quarter overlay is on screen, else 0
  p1, p2   scores shown (blank when overlay is 0)
  quarter  overlay text, e.g. "1ST QUARTER" (blank when overlay is 0)

Recognizers:
  legacy   the existing ScoreDetector / GameQuarterDetector, unchanged
  mask     scanline-bridged glyph mask + Tesseract (see BaseOCRDetector.text_mask)
  template glyph/phrase template matching (needs --templates from build_templates.py)

Usage: python evaluate.py --labels dataset/labels_test.csv [--recognizer legacy mask]
"""

import argparse
import csv
import difflib
import time
from pathlib import Path

import cv2
import pytesseract

from config import Config
from detectors.base_ocr_detector import BaseOCRDetector
from detectors.game_quarter_detector import GameQuarterDetector
from detectors.score_detector import ScoreDetector
from detectors.template_matcher import TemplateLibrary, TemplateRecognizer

PHRASES = ["1ST QUARTER", "2ND QUARTER", "3RD QUARTER", "4TH QUARTER", "HALFTIME", "FINAL"]
DIGIT_CFG = '--psm 8 -c tessedit_char_whitelist=0123456789'
TEXT_CFG = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


def closest_phrase(text: str, cutoff: float = 0.6):
    """Fuzzy-match OCR output to a known overlay phrase (spaces are ignored)."""
    key = text.replace(' ', '')
    if not key:
        return None
    scored = [(difflib.SequenceMatcher(None, key, p.replace(' ', '')).ratio(), p) for p in PHRASES]
    ratio, phrase = max(scored)
    return phrase if ratio >= cutoff else None


class LegacyRecognizer:
    name = 'legacy'

    def __init__(self, det_cfg):
        self.det_cfg = det_cfg
        self.scores = ScoreDetector(det_cfg)

    def read(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scores = self.scores.detect_scores(gray, frame)
        info = GameQuarterDetector(self.det_cfg).detect_quarter_text(frame)
        quarter = None
        if info:
            s, q = info.get('state'), info.get('quarter')
            ords = {1: '1ST', 2: '2ND', 3: '3RD', 4: '4TH'}
            if s == 'quarter' and q:
                quarter = f"{ords[q]} QUARTER"
            elif s == 'halftime':
                quarter = 'HALFTIME'
            elif s == 'final':
                quarter = 'FINAL'
        return scores['player1'], scores['player2'], quarter


class MaskRecognizer:
    name = 'mask'

    def __init__(self, det_cfg):
        self.cfg = det_cfg
        self.base = BaseOCRDetector(det_cfg)

    def _crop(self, frame, r):
        return frame[r.y:r.y + r.height, r.x:r.x + r.width]

    def _ocr(self, frame, region, config):
        mask = self.base.text_mask(self._crop(frame, region))
        if mask is None or not mask.any():
            return ''
        return pytesseract.image_to_string(self.base.mask_to_ocr_image(mask), config=config).strip()

    def read(self, frame):
        if not self.base.has_purple_band(frame):
            return None, None, None
        out = []
        for player in ('player1', 'player2'):
            text = self._ocr(frame, self.cfg.score_regions[player], DIGIT_CFG)
            out.append(int(text) if text.isdigit() else None)
        quarter = closest_phrase(self._ocr(frame, self.cfg.quarter_text_region, TEXT_CFG))
        return out[0], out[1], quarter


def parse_int(v):
    v = (v or '').strip()
    return int(v) if v else None


def main():
    parser = argparse.ArgumentParser(description="Evaluate recognizers on labeled frames")
    parser.add_argument('--labels', required=True)
    parser.add_argument('--root', default=None, help='Directory frame paths are relative to (default: labels file dir)')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--recognizer', nargs='+', default=['legacy', 'mask'], choices=['legacy', 'mask', 'template'])
    parser.add_argument('--templates', default=None, help='Template library .npz for the template recognizer')
    parser.add_argument('--min-time', type=float, default=None, help='Only evaluate frames at/after this elapsed second')
    parser.add_argument('--max-time', type=float, default=None, help='Only evaluate frames before this elapsed second')
    parser.add_argument('--errors', action='store_true', help='List every wrong prediction')
    parser.add_argument('--split', choices=['even', 'odd'], default=None,
                        help='Only evaluate frames whose 20-second time bucket is even/odd')
    args = parser.parse_args()

    labels_path = Path(args.labels)
    root = Path(args.root) if args.root else labels_path.parent
    det_cfg = Config.load(args.config).detection
    gate = BaseOCRDetector(det_cfg)
    classes = {'legacy': LegacyRecognizer, 'mask': MaskRecognizer}
    if 'template' in args.recognizer:
        if not args.templates:
            raise SystemExit("--templates is required for the template recognizer")
        library = TemplateLibrary.load(args.templates)
        classes['template'] = lambda cfg: TemplateRecognizer(cfg, library)

    with open(labels_path, newline='') as f:
        rows = list(csv.DictReader(f))
    t = lambda r: float(Path(r['frame']).stem)
    if args.min_time is not None:
        rows = [r for r in rows if t(r) >= args.min_time]
    if args.max_time is not None:
        rows = [r for r in rows if t(r) < args.max_time]
    if args.split:
        want = 0 if args.split == 'even' else 1
        rows = [r for r in rows if int(t(r) // 20) % 2 == want]
    print(f"{len(rows)} labeled frames from {labels_path}")

    def load(row):
        img = cv2.imread(str(root / row['frame']))
        if img is None:
            raise SystemExit(f"cannot read {root / row['frame']}")
        return img

    # Frames are large (1080p PNG), so stream them instead of holding the set in memory
    gate_ok = 0
    gate_errors = []
    for row in rows:
        want = row['overlay'].strip() == '1'
        got = gate.has_purple_band(load(row))
        gate_ok += got == want
        if got != want:
            gate_errors.append(f"  {row['frame']}: gate={got}, want={want}")
    print(f"Purple gate: {gate_ok}/{len(rows)} correct")
    if args.errors:
        print('\n'.join(gate_errors))
    print()

    overlay_rows = [r for r in rows if r['overlay'].strip() == '1']
    for name in args.recognizer:
        rec = classes[name](det_cfg)
        rec.name = name
        hits = {'p1': 0, 'p2': 0, 'quarter': 0}
        errors = []
        start = time.time()
        for row in overlay_rows:
            p1, p2, q = rec.read(load(row))
            want = {'p1': parse_int(row['p1']), 'p2': parse_int(row['p2']), 'quarter': row['quarter'].strip() or None}
            got = {'p1': p1, 'p2': p2, 'quarter': q}
            for k in hits:
                if got[k] == want[k]:
                    hits[k] += 1
                else:
                    errors.append(f"  {row['frame']} {k}: got {got[k]!r}, want {want[k]!r}")
        n = len(overlay_rows)
        ms = (time.time() - start) / max(n, 1) * 1000
        print(f"[{name}] over {n} overlay frames ({ms:.0f} ms/frame)")
        for k, v in hits.items():
            print(f"  {k:8s} {v}/{n} ({100 * v / max(n, 1):.0f}%)")
        if args.errors:
            print('\n'.join(errors))
        print()


if __name__ == '__main__':
    main()
