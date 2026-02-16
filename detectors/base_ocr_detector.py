"""Base OCR detector with shared utilities for score and quarter detection."""

import cv2
import numpy as np
from typing import Optional, Union
from config import DetectionConfig, ScoreRegion, TextRegion


class BaseOCRDetector:
    """Base class with shared OCR utilities for score and quarter detection."""

    def __init__(self, config: DetectionConfig, debug: bool = False, save_screenshots: bool = False):
        """Initialize base OCR detector.
        
        Args:
            config: Detection configuration
            debug: Enable debug logging
            save_screenshots: Enable saving debug images
        """
        self.config = config
        self.debug = debug
        self.save_screenshots = save_screenshots
        self.screenshot_dir = None  # Will be set by main if screenshots enabled

    def extract_region(self, frame: np.ndarray, region: Union[ScoreRegion, TextRegion]) -> Optional[np.ndarray]:
        """Extract region from frame (generic, works with ScoreRegion or TextRegion).
        
        Args:
            frame: Full video frame
            region: Region coordinates (ScoreRegion or TextRegion)
            
        Returns:
            Extracted region as grayscale image, or None if region is invalid/empty
        """
        # Check if frame is valid
        if frame is None or frame.size == 0:
            return None
        
        h, w = frame.shape[:2]
        x1 = max(0, region.x)
        y1 = max(0, region.y)
        x2 = min(w, region.x + region.width)
        y2 = min(h, region.y + region.height)
        
        # Check if region is valid (has positive dimensions)
        if x2 <= x1 or y2 <= y1:
            return None
        
        roi = frame[y1:y2, x1:x2]
        
        # Check if extracted region is empty
        if roi.size == 0:
            return None
        
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        return roi

    def has_purple_band(self, frame: np.ndarray) -> bool:
        """Check if purple band is present in quarter text region.
        
        Args:
            frame: Full video frame (must be color/BGR, not grayscale)
            
        Returns:
            True if purple band detected, False otherwise
        """
        if self.config.quarter_text_region is None:
            if self.debug:
                print("  Purple check: No quarter_text_region configured")
            return False
        
        # Extract region from color frame (don't convert to grayscale)
        h, w = frame.shape[:2]
        region = self.config.quarter_text_region
        x1 = max(0, region.x)
        y1 = max(0, region.y)
        x2 = min(w, region.x + region.width)
        y2 = min(h, region.y + region.height)
        
        if x2 <= x1 or y2 <= y1:
            if self.debug:
                print(f"  Purple check: Invalid region bounds ({x1},{y1}) to ({x2},{y2})")
            return False
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or len(roi.shape) != 3:
            if self.debug:
                print(f"  Purple check: Invalid ROI (size={roi.size}, shape={roi.shape if roi.size > 0 else 'empty'})")
            return False
        
        # Purple in BGR: B and R are high, G is low
        # Typical purple: B=100-220, G=0-80, R=100-220
        # Check if significant portion of region is purple
        purple_mask = (
            (roi[:, :, 0] >= 100) & (roi[:, :, 0] <= 220) &  # B channel
            (roi[:, :, 1] >= 0) & (roi[:, :, 1] <= 80) &     # G channel (low)
            (roi[:, :, 2] >= 100) & (roi[:, :, 2] <= 220)    # R channel
        )
        
        purple_ratio = np.sum(purple_mask) / roi.size
        
        if self.debug:
            b_mean = np.mean(roi[:, :, 0])
            g_mean = np.mean(roi[:, :, 1])
            r_mean = np.mean(roi[:, :, 2])
            print(f"  Purple check: {np.sum(purple_mask)}/{roi.size} pixels match ({purple_ratio*100:.1f}%), threshold=8%")
            print(f"  Purple check: Result={purple_ratio > 0.08}")
        
        return purple_ratio > 0.08  # At least 8% of region is purple

    def preprocess_for_ocr(self, region: np.ndarray, strategy: int = 0) -> Optional[np.ndarray]:
        """Preprocess image region for better OCR with multiple strategies.
        
        Args:
            region: Grayscale image region
            strategy: Preprocessing strategy (0=adaptive, 1=OTSU, 2=aggressive)
            
        Returns:
            Preprocessed binary image, or None if region is invalid
        """
        # Check if region is valid
        if region is None or region.size == 0:
            return None
        
        # Resize if too small
        h, w = region.shape
        if h < 20 or w < 40:
            region = cv2.resize(region, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        
        if strategy == 0:
            # Adaptive thresholding (default)
            binary = cv2.adaptiveThreshold(
                region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        elif strategy == 1:
            # OTSU threshold
            _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:  # strategy == 2
            # Aggressive: Gaussian blur + OTSU + morphological operations
            blurred = cv2.GaussianBlur(region, (3, 3), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Remove small noise
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Invert if needed (white text on dark background)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
        
        return binary

    def add_border(self, image: np.ndarray, border_size: int = 10) -> np.ndarray:
        """Add white border around image for Tesseract OCR.
        
        Tesseract 4.x works better with images that have borders.
        
        Args:
            image: Binary image
            border_size: Size of border in pixels (default 10)
            
        Returns:
            Image with white border added
        """
        return cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size,
                                  cv2.BORDER_CONSTANT, value=255)

    def ensure_dark_on_light(self, binary: np.ndarray) -> np.ndarray:
        """Ensure image has dark text on light background (Tesseract 4.x requirement).
        
        If mean pixel value < 127, text is dark on light (good).
        If mean pixel value > 127, text is light on dark (invert).
        
        Args:
            binary: Binary image
            
        Returns:
            Binary image with dark text on light background
        """
        if np.mean(binary) > 127:
            return cv2.bitwise_not(binary)
        return binary

    def detect_text_color(self, color_region: np.ndarray) -> str:
        """Detect if text in region is white or red.
        
        Uses percentile-based detection to find the brightest pixels (likely text)
        rather than mean values which can be skewed by background.
        
        Args:
            color_region: Color/BGR image region
            
        Returns:
            'white', 'red', or 'unknown'
        """
        if color_region is None or color_region.size == 0 or len(color_region.shape) != 3:
            return 'unknown'
        
        # Get the top 30% brightest pixels (more lenient - likely to be text)
        # Convert to grayscale for brightness detection
        gray = cv2.cvtColor(color_region, cv2.COLOR_BGR2GRAY)
        threshold = np.percentile(gray, 70)  # Top 30% brightest pixels
        bright_mask = gray >= threshold
        
        if np.sum(bright_mask) == 0:
            return 'unknown'
        
        # Calculate mean values for bright pixels only (likely text)
        b_mean = np.mean(color_region[bright_mask, 0])
        g_mean = np.mean(color_region[bright_mask, 1])
        r_mean = np.mean(color_region[bright_mask, 2])
        
        if self.debug:
            print(f"  Text color analysis: B={b_mean:.1f}, G={g_mean:.1f}, R={r_mean:.1f} (bright pixels only)")
        
        # White text: Lowered threshold to 140 (was 180) to catch medium-brightness text
        if b_mean > 140 and g_mean > 140 and r_mean > 140:
            return 'white'
        
        # Red text: More lenient - R should be higher than B and G
        if r_mean > 140 and r_mean > (b_mean + 30) and r_mean > (g_mean + 30):
            return 'red'
        
        return 'unknown'

