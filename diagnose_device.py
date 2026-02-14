#!/usr/bin/env python3
"""Diagnostic script to check capture device codec and capabilities."""

import sys
import cv2
import subprocess
from typing import Optional, Dict


def get_codec_info(cap: cv2.VideoCapture) -> Optional[Dict[str, str]]:
    """Get codec information from capture device.
    
    Args:
        cap: OpenCV VideoCapture object
        
    Returns:
        Dictionary with codec information or None
    """
    try:
        # Get FourCC code
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
        
        # Map FourCC to human-readable names
        codec_map = {
            'MJPG': ('MJPEG', 'Compressed, moderate CPU for decoding. Good for Pi Zero.'),
            'YUYV': ('YUYV 4:2:2', 'Uncompressed, high bandwidth but fast processing. Best for Pi Zero.'),
            'YU12': ('YUV 4:2:0', 'Uncompressed YUV format. Good for processing.'),
            'YV12': ('YV12', 'Uncompressed YVU format. Good for processing.'),
            'H264': ('H.264', 'Highly compressed, may require hardware decoding. May be CPU-intensive on Pi Zero.'),
            'X264': ('x264', 'H.264 variant, may require hardware decoding.'),
            'I420': ('I420', 'YUV 4:2:0 planar. Good for processing.'),
        }
        
        name, note = codec_map.get(fourcc, (fourcc, 'Unknown codec format.'))
        
        return {
            'fourcc': fourcc,
            'name': name,
            'note': note
        }
    except Exception as e:
        print(f"Warning: Could not detect codec: {e}")
        return None


def get_v4l2_info(device_path: str) -> Optional[str]:
    """Get device information using v4l2-ctl if available.
    
    Args:
        device_path: Path to video device
        
    Returns:
        v4l2-ctl output or None
    """
    try:
        result = subprocess.run(
            ['v4l2-ctl', '--device', device_path, '--list-formats'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    return None


def main():
    """Main diagnostic function."""
    if len(sys.argv) < 2:
        print("Usage: python diagnose_device.py <device_path>")
        print("Example: python diagnose_device.py /dev/video0")
        sys.exit(1)
    
    device_path = sys.argv[1]
    
    print("=" * 60)
    print("Capture Device Diagnostic")
    print("=" * 60)
    print(f"Device: {device_path}\n")
    
    # Try to open device
    cap = cv2.VideoCapture(device_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open device {device_path}")
        sys.exit(1)
    
    # Get basic properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print("Device Properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    
    # Get codec information
    codec_info = get_codec_info(cap)
    if codec_info:
        print(f"\nCodec Information:")
        print(f"  FourCC: {codec_info['fourcc']}")
        print(f"  Name: {codec_info['name']}")
        print(f"  Note: {codec_info['note']}")
    else:
        print("\nCodec Information: Could not detect")
    
    # Try v4l2-ctl for more info
    v4l2_output = get_v4l2_info(device_path)
    if v4l2_output:
        print(f"\nv4l2-ctl Output:")
        print(v4l2_output)
    
    # Performance recommendations
    print("\nPerformance Recommendations:")
    if codec_info:
        codec = codec_info['fourcc']
        if codec == 'H264' or codec == 'X264':
            print("  ⚠️  H.264 codec detected - may be CPU-intensive on Pi Zero W2")
            print("     Consider a capture device that supports MJPEG or YUYV")
        elif codec == 'MJPG':
            print("  ✓ MJPEG codec - good balance for Pi Zero W2")
        elif codec == 'YUYV':
            print("  ✓ YUYV codec - excellent for Pi Zero W2 (fastest processing)")
        else:
            print(f"  Codec {codec} - performance impact unknown")
    else:
        print("  Could not determine codec - check device compatibility")
    
    print("\nSuggested Configuration:")
    if width >= 1920:
        print("  Consider reducing resolution to 1280x720 or 640x480 for better performance")
    print("  Set use_grayscale: true in config.yaml")
    print("  Start with frame_interval: 1.0 and reduce if CPU allows")
    
    cap.release()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

