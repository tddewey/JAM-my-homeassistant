"""Game quarter detection module using OCR for state detection."""

import cv2
import numpy as np
import pytesseract
from typing import Optional, Dict
from enum import Enum
from datetime import datetime
from pathlib import Path
from config import DetectionConfig
from detectors.base_ocr_detector import BaseOCRDetector


class GameState(Enum):
    """Game state enumeration."""
    NOT_PLAYING = "not_playing"  # Default state
    PLAYING = "playing"           # Triggered by team selection screen or rest periods
    Q1 = "q1"                     # Triggered by "1st quarter" text
    Q2 = "q2"                     # Triggered by "2nd quarter" text
    HALFTIME = "halftime"         # Triggered by "halftime" text
    Q3 = "q3"                     # Triggered by "3rd quarter" text
    Q4 = "q4"                     # Triggered by "4th quarter" text
    GAME_OVER = "game_over"       # Triggered by "Final" text


class GameQuarterDetector(BaseOCRDetector):
    """Detects game quarter/state from video frames using OCR."""

    def __init__(self, config: DetectionConfig, save_screenshots: bool = False):
        """Initialize game quarter detector.
        
        Args:
            config: Detection configuration
            save_screenshots: Enable saving debug images
        """
        super().__init__(config, debug=False, save_screenshots=save_screenshots)
        self.current_state = GameState.NOT_PLAYING

    def detect_quarter_text(self, frame: np.ndarray) -> Optional[Dict[str, str]]:
        """Detect quarter/period text from frame.
        
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
        
        # Use adaptive threshold strategy
        binary = self.preprocess_for_ocr(region, strategy=0)
        if binary is None:
            return None
        
        # Ensure dark text on light background (Tesseract 4.x requirement)
        binary = self.ensure_dark_on_light(binary)
        
        # Add border padding (Tesseract requirement)
        binary = self.add_border(binary)
        
        # Save preprocessed image for debugging if screenshots enabled
        if self.save_screenshots and self.screenshot_dir is not None:
            self._save_preprocessed_image(binary, "quarter_preprocessed")
        
        # Detect text using OCR
        text = self._detect_text(binary)
        
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
        
        # Check for quarter indicators - exact format "XND QUARTER", "XST QUARTER", etc.
        if 'quarter' in text_lower:
            result['state'] = 'quarter'
            # Look for exact patterns: "1ST QUARTER", "2ND QUARTER", "3RD QUARTER", "4TH QUARTER"
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

    def _detect_text(self, region: np.ndarray) -> str:
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
            # PSM 6: Assume a single uniform block of text
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

    def detect_state(self, frame: np.ndarray) -> GameState:
        """Detect current game state from quarter text.
        
        Args:
            frame: Video frame
            
        Returns:
            Detected game state
        """
        # Try to detect quarter/period text
        quarter_info = self.detect_quarter_text(frame)
        
        if quarter_info:
            state = quarter_info.get('state')
            quarter = quarter_info.get('quarter')
            
            if state == 'final':
                self.current_state = GameState.GAME_OVER
            elif state == 'halftime':
                self.current_state = GameState.HALFTIME
            elif state == 'end_of_quarter':
                # End of quarter - set to PLAYING (rest period)
                self.current_state = GameState.PLAYING
            elif state == 'quarter' and quarter is not None:
                # Set specific quarter state based on quarter number
                if quarter == 1:
                    self.current_state = GameState.Q1
                elif quarter == 2:
                    self.current_state = GameState.Q2
                elif quarter == 3:
                    self.current_state = GameState.Q3
                elif quarter == 4:
                    self.current_state = GameState.Q4
        
        # If no text detected, maintain current state
        return self.current_state

    def _save_preprocessed_image(self, binary_image: np.ndarray, prefix: str) -> None:
        """Save preprocessed image for debugging.
        
        Args:
            binary_image: Preprocessed binary image
            prefix: Prefix for filename (e.g., "quarter_preprocessed")
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

