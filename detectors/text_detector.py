"""Text detection module using OCR for state detection."""

import cv2
import numpy as np
import pytesseract
from typing import Optional, Dict, Tuple
from config import DetectionConfig, TextRegion


class TextDetector:
    """Detects text from video frames using OCR for state detection."""

    def __init__(self, config: DetectionConfig, save_screenshots: bool = False):
        """Initialize text detector.
        
        Args:
            config: Detection configuration
            save_screenshots: Enable saving Tesseract debug images
        """
        self.config = config
        self.save_screenshots = save_screenshots
        self.screenshot_dir = None  # Will be set by main if screenshots enabled
        self.screenshot_dir = None  # Will be set by main if screenshots enabled

    def has_purple_band(self, frame: np.ndarray) -> bool:
        """Check if purple band is present in quarter text region.
        
        Args:
            frame: Full video frame (must be color/BGR, not grayscale)
            
        Returns:
            True if purple band detected, False otherwise
        """
        if self.config.quarter_text_region is None:
            return False
        
        # Extract region from color frame (don't convert to grayscale)
        h, w = frame.shape[:2]
        region = self.config.quarter_text_region
        x1 = max(0, region.x)
        y1 = max(0, region.y)
        x2 = min(w, region.x + region.width)
        y2 = min(h, region.y + region.height)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or len(roi.shape) != 3:
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
        return purple_ratio > 0.1  # At least 10% of region is purple

    def extract_region(self, frame: np.ndarray, region: TextRegion) -> Optional[np.ndarray]:
        """Extract text region from frame.
        
        Args:
            frame: Full video frame
            region: Text region coordinates
            
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

    def detect_text(self, region: np.ndarray) -> str:
        """Detect text from image region using OCR.
        
        Args:
            region: Preprocessed image region
            
        Returns:
            Detected text (uppercase, stripped)
        """
        # Check if region is valid
        if region is None or region.size == 0:
            return ""
        
        try:
            # Use OCR with appropriate PSM mode
            # PSM 7: Treat image as a single text line
            # PSM 8: Treat image as a single word
            # PSM 6: Assume a single uniform block of text
            # Add tessedit_write_images=1 if screenshots enabled to save debug images
            ocr_config = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            if self.save_screenshots:
                ocr_config += ' -c tessedit_write_images=1'
            
            text = pytesseract.image_to_string(
                region,
                config=ocr_config
            ).strip().upper()
            
            return text
        except Exception as e:
            # OCR failed
            return ""

    def detect_quarter_text(self, frame: np.ndarray) -> Optional[Dict[str, str]]:
        """Detect quarter/period text from frame with multiple OCR strategies.
        
        Args:
            frame: Full video frame (must be color/BGR for purple detection)
            
        Returns:
            Dictionary with detected quarter info, or None if not detected.
            Keys: 'text' (raw detected text), 'quarter' (1-4 or None), 
                  'state' ('quarter', 'end_of_quarter', 'halftime', 'final')
        """
        if self.config.quarter_text_region is None:
            return None
        
        # Check for purple band first - only run OCR if overlay is present
        if not self.has_purple_band(frame):
            return None
        
        # Extract region (convert to grayscale for OCR)
        region = self.extract_region(frame, self.config.quarter_text_region)
        
        # Check if region extraction failed
        if region is None:
            return None
        
        # Use fastest strategy (0: adaptive threshold) for speed
        binary = self.preprocess_for_ocr(region, strategy=0)
        if binary is None:
            return None
        
        # Save preprocessed image for debugging if screenshots enabled
        if self.save_screenshots and self.screenshot_dir is not None:
            self._save_preprocessed_image(binary, "quarter_preprocessed")
        
        # Detect text
        text = self.detect_text(binary)
        
        if not text:
            return None
        
        # Save preprocessed image for debugging if screenshots enabled
        if self.save_screenshots and self.screenshot_dir is not None:
            self._save_preprocessed_image(binary, "quarter_preprocessed")
        
        # Parse quarter/period information
        text_lower = text.lower()
        
        result = {
            'text': text,
            'quarter': None,
            'state': None
        }
        
        # Check for "final"
        if 'final' in text_lower:
            result['state'] = 'final'
            return result
        
        # Check for "halftime"
        if 'halftime' in text_lower or 'half time' in text_lower:
            result['state'] = 'halftime'
            return result
        
        # Check for "end of" quarter
        if 'end of' in text_lower or 'end' in text_lower:
            result['state'] = 'end_of_quarter'
            # Try to extract quarter number
            if 'first' in text_lower or '1st' in text_lower or '1' in text:
                result['quarter'] = 1
            elif 'second' in text_lower or '2nd' in text_lower or '2' in text:
                result['quarter'] = 2
            elif 'third' in text_lower or '3rd' in text_lower or '3' in text:
                result['quarter'] = 3
            elif 'fourth' in text_lower or '4th' in text_lower or '4' in text:
                result['quarter'] = 4
            return result
        
        # Check for quarter indicators - exact format "XND QUARTER", "XST QUARTER", etc.
        if 'quarter' in text_lower:
            result['state'] = 'quarter'
            # Look for exact patterns: "1ST QUARTER", "2ND QUARTER", "3RD QUARTER", "4TH QUARTER"
            # Also handle variations like "1ST", "2ND", etc. before "QUARTER"
            text_upper = text.upper()
            if '1ST' in text_upper or ('1' in text and 'ST' in text_upper):
                result['quarter'] = 1
            elif '2ND' in text_upper or ('2' in text and 'ND' in text_upper):
                result['quarter'] = 2
            elif '3RD' in text_upper or ('3' in text and 'RD' in text_upper):
                result['quarter'] = 3
            elif '4TH' in text_upper or ('4' in text and 'TH' in text_upper):
                result['quarter'] = 4
            # Fallback to simple number matching if exact pattern not found
            elif result['quarter'] is None:
                if 'first' in text_lower or '1' in text:
                    result['quarter'] = 1
                elif 'second' in text_lower or '2' in text:
                    result['quarter'] = 2
                elif 'third' in text_lower or '3' in text:
                    result['quarter'] = 3
                elif 'fourth' in text_lower or '4' in text:
                    result['quarter'] = 4
            return result
        
        # If we got text but couldn't parse it, return None
        return None

    def detect_team_selection_heading(self, frame: np.ndarray) -> Optional[str]:
        """Detect team selection heading text from frame.
        
        Args:
            frame: Full video frame
            
        Returns:
            Detected heading text, or None if not detected.
            Returns "PRESS SHOOT TO SELECT" for pre-game to trigger PLAYING state
        """
        if self.config.team_selection_heading_region is None:
            return None
        
        # Extract region
        region = self.extract_region(frame, self.config.team_selection_heading_region)
        
        # Check if region extraction failed
        if region is None:
            return None
        
        # Try multiple preprocessing strategies for better detection
        texts_found = []
        for strategy in range(3):
            binary = self.preprocess_for_ocr(region, strategy)
            if binary is None:
                continue
            
            # Detect text
            text = self.detect_text(binary)
            if text:
                texts_found.append(text)
        
        # Use first non-empty text found
        text = texts_found[0] if texts_found else None
        
        if not text:
            return None
        
        # Check for pre-game heading text
        text_upper = text.upper()
        
        # Pre-game heading
        if 'PRESS' in text_upper and 'SHOOT' in text_upper and 'SELECT' in text_upper:
            return "PRESS SHOOT TO SELECT"
        
        return None

    def _save_preprocessed_image(self, binary_image: np.ndarray, prefix: str) -> None:
        """Save preprocessed image for debugging.
        
        Args:
            binary_image: Preprocessed binary image
            prefix: Prefix for filename (e.g., "score_preprocessed", "quarter_preprocessed")
        """
        if not self.save_screenshots or self.screenshot_dir is None:
            return
        
        try:
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]  # Include milliseconds
            filename = f"{prefix}_{timestamp_str}.png"
            dest_path = self.screenshot_dir / filename
            cv2.imwrite(str(dest_path), binary_image)
        except Exception:
            pass  # Don't fail on debug image save

