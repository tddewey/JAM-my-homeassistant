#!/usr/bin/env python3
"""Replay a recorded game through the GameTracker and print the events it emits.

  python replay.py extract FRAME_DIR --templates T.npz --out features.json
  python replay.py run features.json [--interval 0.5]

`extract` is slow (decodes every PNG); `run` is instant, so tracker changes can be
iterated on quickly. Use templates trained on a different game for an honest test.
"""

import argparse
import glob
import json
from multiprocessing import Pool

import cv2

from config import Config
from detectors.scene_detector import SceneDetector
from detectors.template_matcher import TemplateLibrary, TemplateRecognizer
from game_tracker import GameTracker

_rec = _scenes = None


def _init(config_path, templates, scenes):
    global _rec, _scenes
    _rec = TemplateRecognizer(Config.load(config_path).detection, TemplateLibrary.load(templates))
    _scenes = SceneDetector.load(scenes)


def _features(path):
    frame = cv2.imread(path)
    t = float(path.split('/')[-1][:-4])
    reading = _rec.read(frame)
    overlay = _rec.base.has_purple_band(frame)
    return {'t': t, 'scene': _scenes.classify(frame),
            'reading': list(reading) if overlay else None}


def extract(args):
    paths = sorted(glob.glob(f"{args.frame_dir}/*.png"))
    with Pool(initializer=_init, initargs=(args.config, args.templates, args.scenes)) as pool:
        rows = pool.map(_features, paths, chunksize=8)
    json.dump(rows, open(args.out, 'w'))
    print(f"{len(rows)} frames -> {args.out}")


def run(args):
    rows = json.load(open(args.features))
    tracker = GameTracker()
    for row in rows:
        reading = tuple(row['reading']) if row['reading'] else None
        for e in tracker.update(row['scene'], reading):
            detail = {k: v for k, v in e.__dict__.items() if v is not None and k != 'kind'}
            print(f"{row['t']:7.1f}s {e.kind:7s} {detail}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('extract')
    e.add_argument('frame_dir')
    e.add_argument('--templates', required=True)
    e.add_argument('--scenes', default='templates/scenes.npz')
    e.add_argument('--config', default='config.yaml')
    e.add_argument('--out', required=True)
    r = sub.add_parser('run')
    r.add_argument('features')
    args = p.parse_args()
    extract(args) if args.cmd == 'extract' else run(args)


if __name__ == '__main__':
    main()
