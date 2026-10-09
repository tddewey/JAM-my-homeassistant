#!/usr/bin/env python3
"""Record lossless frames from the capture card to build a labeled OCR dataset.

Frames are saved as PNG (no compression artifacts) and near-duplicates of the
previous saved frame are skipped, so static screens don't fill the disk.
Sort the output into folders (q1/, halftime/, ...) afterwards to label them.
"""

import argparse
import signal
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import VideoConfig
from video_capture import VideoCapture


def main():
    parser = argparse.ArgumentParser(description="Record raw capture frames for dataset building")
    parser.add_argument('--device', default='/dev/video0')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--fourcc', default='YUYV')
    parser.add_argument('--interval', type=float, default=0.5, help='Seconds between saved frames')
    parser.add_argument('--duration', type=float, default=0, help='Stop after N seconds (0 = until Ctrl+C)')
    parser.add_argument('--diff-threshold', type=float, default=0.5,
                        help='Skip a frame if its mean abs difference from the last saved frame '
                             '(on a 1/8 scale grayscale copy) is below this. 0 disables.')
    parser.add_argument('--out', default='dataset/raw', help='Output directory')
    args = parser.parse_args()

    out_dir = Path(args.out) / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = VideoConfig(
        device_path=args.device,
        frame_interval=args.interval,
        capture_width=args.width,
        capture_height=args.height,
        use_grayscale=False,
        fourcc=args.fourcc,
    )
    cap = VideoCapture(cfg)
    if not cap.open():
        raise SystemExit(1)

    stop = False

    def handle_sigint(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_sigint)

    print(f"Saving to {out_dir} (Ctrl+C to stop)")
    start = time.time()
    last_small = None
    last_ts = 0.0
    saved = skipped = 0

    while not stop and (args.duration <= 0 or time.time() - start < args.duration):
        result = cap.read_frame()
        if result is None:
            time.sleep(0.01)
            continue
        frame, ts = result
        if ts == last_ts:  # cached frame, no new capture yet
            time.sleep(0.01)
            continue
        last_ts = ts

        if args.diff_threshold > 0:
            small = cv2.cvtColor(cv2.resize(frame, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA),
                                 cv2.COLOR_BGR2GRAY).astype(np.int16)
            if last_small is not None and np.mean(np.abs(small - last_small)) < args.diff_threshold:
                skipped += 1
                continue
            last_small = small

        name = f"{ts - start:08.2f}.png"
        cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        saved += 1
        if saved % 20 == 0:
            print(f"  {saved} saved, {skipped} skipped ({ts - start:.0f}s elapsed)")

    cap.release()
    print(f"Done: {saved} frames saved, {skipped} near-duplicates skipped -> {out_dir}")


if __name__ == '__main__':
    main()
