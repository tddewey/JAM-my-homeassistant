"""Text detection module using OCR for state detection."""

import cv2
import numpy as np
import pytesseract
from typing import Optional, Dict, Tuple
from config import DetectionConfig, TextRegion


class TextDetector:
    """Detects text from video frames using OCR for state detection."""

    def __init__(self, config: DetectionConfig):
        """Initialize text detector.
        
        Args:
            config: Detection configuration
        """
        self.config = config

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

    def preprocess_for_ocr(self, region: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess image region for better OCR.
        
        Args:
            region: Grayscale image region
            
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
        
        # Apply adaptive thresholding for better text extraction
        # This works better than simple threshold for varying backgrounds
        binary = cv2.adaptiveThreshold(
            region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
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
            text = pytesseract.image_to_string(
                region,
                config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
            ).strip().upper()
            
            return text
        except Exception as e:
            # OCR failed
            return ""

    def detect_quarter_text(self, frame: np.ndarray) -> Optional[Dict[str, str]]:
        """Detect quarter/period text from frame.
        
        Args:
            frame: Full video frame
            
        Returns:
            Dictionary with detected quarter info, or None if not detected.
            Keys: 'text' (raw detected text), 'quarter' (1-4 or None), 
                  'state' ('quarter', 'end_of_quarter', 'halftime', 'final')
        """
        if self.config.quarter_text_region is None:
            return None
        
        # Extract region
        region = self.extract_region(frame, self.config.quarter_text_region)
        
        # Check if region extraction failed
        if region is None:
            return None
        
        # Preprocess for OCR
        binary = self.preprocess_for_ocr(region)
        
        # Check if preprocessing failed
        if binary is None:
            return None
        
        # Detect text
        text = self.detect_text(binary)
        
        if not text:
            return None
        
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
        
        # Check for quarter indicators
        if 'quarter' in text_lower:
            result['state'] = 'quarter'
            # Extract quarter number
            if 'first' in text_lower or '1st' in text_lower or '1' in text:
                result['quarter'] = 1
            elif 'second' in text_lower or '2nd' in text_lower or '2' in text:
                result['quarter'] = 2
            elif 'third' in text_lower or '3rd' in text_lower or '3' in text:
                result['quarter'] = 3
            elif 'fourth' in text_lower or '4th' in text_lower or '4' in text:
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
        
        # Preprocess for OCR
        binary = self.preprocess_for_ocr(region)
        
        # Check if preprocessing failed
        if binary is None:
            return None
        
        # Detect text
        text = self.detect_text(binary)
        
        if not text:
            return None
        
        # Check for pre-game heading text
        text_upper = text.upper()
        
        # Pre-game heading
        if 'PRESS' in text_upper and 'SHOOT' in text_upper and 'SELECT' in text_upper:
            return "PRESS SHOOT TO SELECT"
        
        return None

