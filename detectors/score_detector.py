"""Score detection module using template matching and OCR."""

import cv2
import numpy as np
import pytesseract
from typing import Optional, Dict, Tuple
from config import DetectionConfig, ScoreRegion


class ScoreDetector:
    """Detects scores from video frames using template matching and OCR."""

    def __init__(self, config: DetectionConfig):
        """Initialize score detector.
        
        Args:
            config: Detection configuration
        """
        self.config = config
        self.last_frames: Dict[str, np.ndarray] = {}
        self.last_scores: Dict[str, Optional[int]] = {
            'player1': None,
            'player2': None
        }

    def extract_region(self, frame: np.ndarray, region: ScoreRegion) -> np.ndarray:
        """Extract score region from frame.
        
        Args:
            frame: Full video frame
            region: Score region coordinates
            
        Returns:
            Extracted region as grayscale image
        """
        h, w = frame.shape[:2]
        x1 = max(0, region.x)
        y1 = max(0, region.y)
        x2 = min(w, region.x + region.width)
        y2 = min(h, region.y + region.height)
        
        roi = frame[y1:y2, x1:x2]
        
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        return roi

    def detect_motion(self, current: np.ndarray, previous: Optional[np.ndarray], 
                     threshold: float) -> bool:
        """Detect motion in score region.
        
        Args:
            current: Current frame region
            previous: Previous frame region (None if first frame)
            threshold: Motion threshold (0.0 to 1.0)
            
        Returns:
            True if motion detected, False otherwise
        """
        if previous is None:
            return True  # First frame, always process
        
        # Calculate frame difference
        diff = cv2.absdiff(current, previous)
        diff_mean = np.mean(diff) / 255.0
        
        return diff_mean > threshold

    def detect_with_ocr(self, region: np.ndarray) -> Optional[int]:
        """Detect score using OCR.
        
        Args:
            region: Score region image (grayscale)
            
        Returns:
            Detected score or None if not detected
        """
        # Preprocess image for better OCR
        # Resize if too small
        h, w = region.shape
        if h < 20 or w < 40:
            region = cv2.resize(region, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        
        # Threshold to binary
        _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if needed (white text on dark background)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
        
        # OCR with digit-only whitelist
        try:
            text = pytesseract.image_to_string(
                binary,
                config='--psm 7 -c tessedit_char_whitelist=0123456789'
            ).strip()
            
            if text:
                # Extract first number found
                score = int(text[0]) if text[0].isdigit() else None
                # Try to parse full number
                try:
                    score = int(text)
                except ValueError:
                    # Try first sequence of digits
                    digits = ''.join(c for c in text if c.isdigit())
                    if digits:
                        score = int(digits)
                    else:
                        score = None
                
                return score
        except Exception as e:
            # OCR failed
            pass
        
        return None

    def detect_score(self, frame: np.ndarray, player: str) -> Optional[int]:
        """Detect score for a specific player.
        
        Args:
            frame: Full video frame
            player: 'player1' or 'player2'
            
        Returns:
            Detected score or None if not visible/detected
        """
        if player not in ['player1', 'player2']:
            return None
        
        region_config = self.config.score_regions.get(player)
        if region_config is None:
            return None
        
        # Extract score region
        region = self.extract_region(frame, region_config)
        
        # Check for motion
        previous_region = self.last_frames.get(player)
        has_motion = self.detect_motion(
            region,
            previous_region,
            self.config.motion_threshold
        )
        
        # Store current frame for next comparison
        self.last_frames[player] = region.copy()
        
        # Only process if motion detected or first frame
        if not has_motion and previous_region is not None:
            # No motion, return cached score
            return self.last_scores[player]
        
        # Try OCR detection
        score = self.detect_with_ocr(region)
        
        # Validate score (reasonable range for arcade game)
        if score is not None and (score < 0 or score > 999):
            score = None
        
        # Update cached score
        if score is not None:
            self.last_scores[player] = score
        # Note: We don't clear cached score if None - let it persist
        
        return score

    def detect_scores(self, frame: np.ndarray) -> Dict[str, Optional[int]]:
        """Detect scores for both players.
        
        Args:
            frame: Full video frame
            
        Returns:
            Dictionary with 'player1' and 'player2' scores (None if not detected)
        """
        return {
            'player1': self.detect_score(frame, 'player1'),
            'player2': self.detect_score(frame, 'player2')
        }

