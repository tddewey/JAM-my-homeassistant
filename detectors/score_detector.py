"""Score detection module using OCR."""

import cv2
import numpy as np
import pytesseract
from typing import Optional, Dict, Tuple
from datetime import datetime
from pathlib import Path
from config import DetectionConfig, ScoreRegion
from detectors.base_ocr_detector import BaseOCRDetector


class ScoreDetector(BaseOCRDetector):
    """Detects scores from video frames using OCR."""

    def __init__(self, config: DetectionConfig, debug: bool = False, save_screenshots: bool = False):
        """Initialize score detector.
        
        Args:
            config: Detection configuration
            debug: Enable debug logging
            save_screenshots: Enable saving debug images
        """
        super().__init__(config, debug=debug, save_screenshots=save_screenshots)

    def preprocess_red_text(self, color_region: np.ndarray) -> Optional[np.ndarray]:
        """Special preprocessing for red text to enhance contrast.
        
        Args:
            color_region: Color/BGR image region
            
        Returns:
            Binary image (will be inverted by ensure_dark_on_light in detect_with_ocr)
        """
        if color_region is None or color_region.size == 0 or len(color_region.shape) != 3:
            return None
        
        # Create mask for red pixels (more lenient thresholds)
        red_mask = (
            (color_region[:, :, 2] > 150) &  # R channel high (lowered from 200)
            (color_region[:, :, 0] < 120) &  # B channel low (raised from 100)
            (color_region[:, :, 1] < 120)    # G channel low (raised from 100)
        )
        
        # Convert to grayscale
        gray = cv2.cvtColor(color_region, cv2.COLOR_BGR2GRAY)
        
        # Convert red pixels to white (enhance contrast)
        gray[red_mask] = 255
        
        # Also enhance any bright pixels (likely text) - top 30% brightest
        bright_threshold = np.percentile(gray, 70)
        bright_mask = gray >= bright_threshold
        gray[bright_mask] = 255
        
        # Apply OTSU threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # DON'T call ensure_dark_on_light here - let detect_with_ocr handle it
        # This prevents double inversion
        return binary

    def detect_with_ocr(self, region: np.ndarray, color_region: Optional[np.ndarray] = None) -> Tuple[Optional[int], Optional[str]]:
        """Detect score using OCR with multiple preprocessing strategies.
        
        Args:
            region: Score region image (grayscale)
            color_region: Color/BGR region for red text detection (optional)
            
        Returns:
            Tuple of (detected score, raw OCR text) or (None, None) if not detected
        """
        if region is None or region.size == 0:
            return None, None
        
        # Detect text color if color region available
        text_color = 'unknown'
        if color_region is not None:
            text_color = self.detect_text_color(color_region)
            if self.debug:
                print(f"  Text color detected: {text_color}")
        
        # Try multiple preprocessing strategies
        strategies = []
        
        # Strategy 0: White text (normal OTSU)
        if text_color in ['white', 'unknown']:
            binary = self.preprocess_for_ocr(region, strategy=1)  # OTSU
            if binary is not None:
                binary = self.ensure_dark_on_light(binary)
                binary = self.add_border(binary)
                strategies.append(('white', binary))
        
        # Strategy 1: Red text (special preprocessing)
        if text_color in ['red', 'unknown']:
            if color_region is not None:
                red_binary = self.preprocess_red_text(color_region)
                if red_binary is not None:
                    red_binary = self.ensure_dark_on_light(red_binary)
                    red_binary = self.add_border(red_binary)
                    strategies.append(('red', red_binary))
            else:
                # Fallback: try aggressive preprocessing
                binary = self.preprocess_for_ocr(region, strategy=2)  # Aggressive
                if binary is not None:
                    binary = self.ensure_dark_on_light(binary)
                    binary = self.add_border(binary)
                    strategies.append(('red_fallback', binary))
        
        # Strategy 2: Adaptive threshold (fallback)
        if text_color == 'unknown' or len(strategies) == 0:
            binary = self.preprocess_for_ocr(region, strategy=0)  # Adaptive
            if binary is not None:
                binary = self.ensure_dark_on_light(binary)
                binary = self.add_border(binary)
                strategies.append(('adaptive', binary))
        
        # Strategy 3: Aggressive preprocessing for unknown text
        if text_color == 'unknown':
            binary = self.preprocess_for_ocr(region, strategy=2)  # Aggressive
            if binary is not None:
                binary = self.ensure_dark_on_light(binary)
                binary = self.add_border(binary)
                strategies.append(('aggressive', binary))
        
        # Try each strategy in sequence
        ocr_config = '--psm 7 -c tessedit_char_whitelist=0123456789'
        
        for strategy_name, binary in strategies:
            # Save preprocessed image for debugging if screenshots enabled
            if self.save_screenshots and self.screenshot_dir is not None:
                self._save_preprocessed_image(binary, f"score_{strategy_name}")
            
            # Additional debug info for failed detections
            if self.debug:
                # Check image statistics
                mean_val = np.mean(binary)
                std_val = np.std(binary)
                print(f"  Strategy {strategy_name}: image stats - mean={mean_val:.1f}, std={std_val:.1f}, shape={binary.shape}")
            
            try:
                text = pytesseract.image_to_string(binary, config=ocr_config).strip()
                
                if text:
                    # Try to parse full number
                    try:
                        score = int(text)
                        if 0 <= score <= 999:
                            if self.debug:
                                print(f"  OCR ({strategy_name}): detected={score}, raw_text='{text}'")
                            return score, text
                    except ValueError:
                        # Try first sequence of digits
                        digits = ''.join(c for c in text if c.isdigit())
                        if digits:
                            try:
                                score = int(digits)
                                if 0 <= score <= 999:
                                    if self.debug:
                                        print(f"  OCR ({strategy_name}): detected={score}, raw_text='{text}'")
                                    return score, text
                            except ValueError:
                                pass
                
                if self.debug:
                    print(f"  OCR ({strategy_name}): no valid score found, raw_text='{text}'")
            except Exception as e:
                if self.debug:
                    print(f"  OCR ({strategy_name}): exception={e}")
        
        return None, None

    def detect_score(self, frame: np.ndarray, color_frame: Optional[np.ndarray], player: str) -> Optional[int]:
        """Detect score for a specific player (direct detection, no validation).
        
        Args:
            frame: Grayscale video frame
            color_frame: Color/BGR frame for color detection (optional)
            player: 'player1' or 'player2'
            
        Returns:
            Detected score or None if not detected
        """
        if player not in ['player1', 'player2']:
            return None
        
        region_config = self.config.score_regions.get(player)
        if region_config is None:
            return None
        
        # Extract score region (grayscale)
        region = self.extract_region(frame, region_config)
        if region is None:
            if self.debug:
                print(f"  {player}: Region extraction failed")
            return None
        
        # Extract color region if available
        color_region = None
        if color_frame is not None:
            h, w = color_frame.shape[:2]
            x1 = max(0, region_config.x)
            y1 = max(0, region_config.y)
            x2 = min(w, region_config.x + region_config.width)
            y2 = min(h, region_config.y + region_config.height)
            if x2 > x1 and y2 > y1:
                color_region = color_frame[y1:y2, x1:x2]
        
        # Try OCR detection with multiple strategies
        score, raw_text = self.detect_with_ocr(region, color_region)
        
        if self.debug:
            print(f"  {player}: OCR raw='{raw_text}', detected={score}")
        
        # Validate score range (0-999)
        if score is not None and (score < 0 or score > 999):
            if self.debug:
                print(f"  {player}: Score {score} out of range (0-999)")
            return None
        
        return score

    def detect_scores(self, frame: np.ndarray, color_frame: Optional[np.ndarray] = None) -> Dict[str, Optional[int]]:
        """Detect scores for both players.
        
        Args:
            frame: Video frame (grayscale or color) for OCR
            color_frame: Color/BGR frame for purple detection and color detection (if None, uses frame)
            
        Returns:
            Dictionary with 'player1' and 'player2' scores (None if not detected)
        """
        # Early bail: Check for purple band before any expensive operations
        check_frame = color_frame if color_frame is not None else frame
        
        # Only proceed if we have a color frame and purple is detected
        if check_frame is not None and len(check_frame.shape) == 3:
            if not self.has_purple_band(check_frame):
                if self.debug:
                    print("  No purple band detected, skipping OCR")
                # Return None for both players (no fallback to cached values)
                return {
                    'player1': None,
                    'player2': None
                }
        
        # Purple detected (or no color frame available) - proceed with detection
        return {
            'player1': self.detect_score(frame, color_frame, 'player1'),
            'player2': self.detect_score(frame, color_frame, 'player2')
        }

    def _save_preprocessed_image(self, binary_image: np.ndarray, prefix: str) -> None:
        """Save preprocessed image for debugging.
        
        Args:
            binary_image: Preprocessed binary image
            prefix: Prefix for filename (e.g., "score_white", "score_red")
        """
        if not self.save_screenshots or self.screenshot_dir is None:
            return
        
        try:
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]  # Include milliseconds
            filename = f"{prefix}_{timestamp_str}.png"
            dest_path = self.screenshot_dir / filename
            cv2.imwrite(str(dest_path), binary_image)
            
            if self.debug:
                print(f"  Saved preprocessed image: {filename}")
        except Exception as e:
            if self.debug:
                print(f"  Warning: Failed to save preprocessed image: {e}")
