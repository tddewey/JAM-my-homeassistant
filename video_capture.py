"""Video capture module for HDMI capture card."""

import cv2
import numpy as np
import time
from typing import Optional, Tuple, Dict
from config import VideoConfig


class VideoCapture:
    """Wrapper for OpenCV VideoCapture with frame sampling."""

    def __init__(self, config: VideoConfig):
        """Initialize video capture.
        
        Args:
            config: Video configuration
        """
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.last_frame_time = 0.0
        self.last_frame: Optional[np.ndarray] = None
        self.last_frame_gray: Optional[np.ndarray] = None
        self.codec_info: Optional[Dict[str, str]] = None

    def open(self) -> bool:
        """Open video capture device.
        
        Returns:
            True if device opened successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.config.device_path)
            if not self.cap.isOpened():
                print(f"Error: Could not open video device {self.config.device_path}")
                return False
            
            # Set resolution from config
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.capture_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.capture_height)
            
            # Detect codec
            self.codec_info = self.get_codec_info()
            
            print(f"Video capture opened: {self.config.device_path}")
            print(f"  Resolution: {self.config.capture_width}x{self.config.capture_height}")
            if self.codec_info:
                print(f"  Codec: {self.codec_info['name']} ({self.codec_info['fourcc']})")
                if self.codec_info.get('note'):
                    print(f"  Note: {self.codec_info['note']}")
            
            return True
        except Exception as e:
            print(f"Error opening video device: {e}")
            return False

    def read_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        """Read a frame if enough time has passed since last frame.
        
        Returns:
            Tuple of (frame, timestamp) if frame was read, None otherwise
        """
        if self.cap is None or not self.cap.isOpened():
            return None

        current_time = time.time()
        time_since_last = current_time - self.last_frame_time

        # Only read frame if enough time has passed
        if time_since_last >= self.config.frame_interval:
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame
                # Convert to grayscale if enabled
                if self.config.use_grayscale:
                    self.last_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    self.last_frame_gray = None
                self.last_frame_time = current_time
                return (frame, current_time)
            else:
                print("Warning: Failed to read frame from video device")
                return None

        # Return cached frame if available
        if self.last_frame is not None:
            return (self.last_frame, self.last_frame_time)

        return None

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame (may be cached).
        
        Returns:
            Frame array or None
        """
        result = self.read_frame()
        if result:
            return result[0]
        return None

    def get_frame_grayscale(self) -> Optional[np.ndarray]:
        """Get the most recent frame as grayscale (may be cached).
        
        Returns:
            Grayscale frame array or None
        """
        if self.config.use_grayscale and self.last_frame_gray is not None:
            return self.last_frame_gray
        elif self.last_frame is not None:
            return cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
        return None

    def get_codec_info(self) -> Optional[Dict[str, str]]:
        """Get codec information from capture device.
        
        Returns:
            Dictionary with codec information or None
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        
        try:
            # Get FourCC code
            fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
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

    def release(self):
        """Release video capture device."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Video capture released")

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()

