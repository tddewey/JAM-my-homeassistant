"""Score detection module using template matching and OCR."""

import cv2
import numpy as np
import pytesseract
from typing import Optional, Dict, Tuple, List
from collections import deque, Counter
from config import DetectionConfig, ScoreRegion


class ScoreDetector:
    """Detects scores from video frames using template matching and OCR."""

    def __init__(self, config: DetectionConfig, debug: bool = False):
        """Initialize score detector.
        
        Args:
            config: Detection configuration
            debug: Enable debug logging
        """
        self.config = config
        self.debug = debug
        self.last_frames: Dict[str, np.ndarray] = {}
        self.last_scores: Dict[str, Optional[int]] = {
            'player1': None,
            'player2': None
        }
        # Multi-frame consensus: track recent detections
        self.score_history: Dict[str, deque] = {
            'player1': deque(maxlen=3),  # Require 2-3 consistent detections
            'player2': deque(maxlen=3)
        }
        # Track last valid scores for temporal validation
        self.last_valid_scores: Dict[str, Optional[int]] = {
            'player1': None,
            'player2': None
        }

    def extract_region(self, frame: np.ndarray, region: ScoreRegion) -> Optional[np.ndarray]:
        """Extract score region from frame.
        
        Args:
            frame: Full video frame
            region: Score region coordinates
            
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

    def detect_motion(self, current: Optional[np.ndarray], previous: Optional[np.ndarray], 
                     threshold: float) -> bool:
        """Detect motion in score region.
        
        Args:
            current: Current frame region
            previous: Previous frame region (None if first frame)
            threshold: Motion threshold (0.0 to 1.0)
            
        Returns:
            True if motion detected, False otherwise
        """
        if current is None:
            return False  # Invalid current frame, no motion
        
        if previous is None:
            return True  # First frame, always process
        
        # Calculate frame difference
        diff = cv2.absdiff(current, previous)
        diff_mean = np.mean(diff) / 255.0
        
        return diff_mean > threshold

    def preprocess_for_ocr(self, region: np.ndarray, strategy: int = 0) -> Optional[np.ndarray]:
        """Preprocess image for OCR with multiple strategies.
        
        Args:
            region: Score region image (grayscale)
            strategy: Preprocessing strategy (0=default, 1=aggressive, 2=adaptive)
            
        Returns:
            Preprocessed binary image or None
        """
        if region is None or region.size == 0:
            return None
        
        # Resize if too small
        h, w = region.shape
        if h < 20 or w < 40:
            region = cv2.resize(region, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        
        if strategy == 0:
            # Default: OTSU threshold
            _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(binary) > 127:
                binary = cv2.bitwise_not(binary)
        elif strategy == 1:
            # Aggressive: Morphological operations for noise reduction
            # Apply Gaussian blur first
            blurred = cv2.GaussianBlur(region, (3, 3), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(binary) > 127:
                binary = cv2.bitwise_not(binary)
            # Remove small noise
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        else:  # strategy == 2
            # Adaptive threshold
            binary = cv2.adaptiveThreshold(
                region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            if np.mean(binary) > 127:
                binary = cv2.bitwise_not(binary)
        
        return binary

    def detect_with_ocr(self, region: np.ndarray) -> Tuple[Optional[int], Optional[str]]:
        """Detect score using OCR with fastest preprocessing strategy.
        
        Args:
            region: Score region image (grayscale)
            
        Returns:
            Tuple of (detected score, raw OCR text) or (None, None) if not detected
        """
        if region is None or region.size == 0:
            return None, None
        
        # Use fastest strategy (0: OTSU threshold) for speed
        binary = self.preprocess_for_ocr(region, strategy=0)
        if binary is None:
            return None, None
        
        # OCR with digit-only whitelist
        try:
            text = pytesseract.image_to_string(
                binary,
                config='--psm 7 -c tessedit_char_whitelist=0123456789'
            ).strip()
            
            if text:
                # Try to parse full number
                try:
                    score = int(text)
                    if 0 <= score <= 999:
                        if self.debug:
                            print(f"  OCR: detected={score}, raw_text='{text}'")
                        return score, text
                except ValueError:
                    # Try first sequence of digits
                    digits = ''.join(c for c in text if c.isdigit())
                    if digits:
                        try:
                            score = int(digits)
                            if 0 <= score <= 999:
                                if self.debug:
                                    print(f"  OCR: detected={score}, raw_text='{text}'")
                                return score, text
                        except ValueError:
                            pass
            
            if self.debug:
                print(f"  OCR: no valid score found, raw_text='{text}'")
            return None, text
        except Exception as e:
            if self.debug:
                print(f"  OCR: exception={e}")
            return None, None

    def validate_score_temporal(self, score: Optional[int], player: str) -> Tuple[Optional[int], str]:
        """Validate score using temporal consistency checks.
        
        Args:
            score: Detected score to validate
            player: 'player1' or 'player2'
            
        Returns:
            Tuple of (validated score, reason) - score may be None if invalid
        """
        if score is None:
            return None, "no score detected"
        
        last_valid = self.last_valid_scores[player]
        
        # First valid score is always accepted
        if last_valid is None:
            return score, "first valid score"
        
        # Score shouldn't decrease (unless game reset - but we'll be conservative)
        if score < last_valid:
            reason = f"score decreased ({last_valid} -> {score})"
            if self.debug:
                print(f"  Validation failed for {player}: {reason}")
            return None, reason
        
        # Score shouldn't increase by more than reasonable amount per detection
        # NBA Jam: typical score increments are 1-3 points, rarely more
        # Allow up to 10 points increase to account for missed detections
        max_increase = 10
        if score > last_valid + max_increase:
            reason = f"score increased too much ({last_valid} -> {score}, max {max_increase})"
            if self.debug:
                print(f"  Validation failed for {player}: {reason}")
            return None, reason
        
        return score, "valid"
    
    def get_consensus_score(self, player: str) -> Optional[int]:
        """Get consensus score from recent detections.
        
        Args:
            player: 'player1' or 'player2'
            
        Returns:
            Consensus score if enough consistent detections, None otherwise
        """
        history = self.score_history[player]
        if len(history) < 2:
            return None
        
        # Count occurrences of each score
        score_counts = Counter(history)
        most_common = score_counts.most_common(1)[0]
        
        # Require at least 2 consistent detections
        if most_common[1] >= 2:
            return most_common[0]
        
        return None

    def detect_score(self, frame: np.ndarray, player: str) -> Optional[int]:
        """Detect score for a specific player with temporal validation and consensus.
        
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
        
        # Check if region extraction failed
        if region is None:
            if self.debug:
                print(f"  {player}: Region extraction failed, returning last valid={self.last_valid_scores[player]}")
            # Return last valid score if available
            return self.last_valid_scores[player]
        
        # REMOVED: Motion detection - always process for speed
        # Motion detection was mostly always true and added overhead
        
        # Try OCR detection
        score, raw_text = self.detect_with_ocr(region)
        
        if self.debug:
            print(f"  {player}: OCR raw='{raw_text}', detected={score}")
        
        # Validate score range (0-999)
        if score is not None and (score < 0 or score > 999):
            if self.debug:
                print(f"  {player}: Score {score} out of range (0-999)")
            score = None
        
        # Add to history for consensus
        if score is not None:
            self.score_history[player].append(score)
            if self.debug:
                print(f"  {player}: Added to history, history={list(self.score_history[player])}")
        
        # Get consensus score from recent history
        consensus_score = self.get_consensus_score(player)
        
        if self.debug:
            print(f"  {player}: Consensus score={consensus_score}, current score={score}")
        
        # Use consensus if available, otherwise use current detection
        final_score = consensus_score if consensus_score is not None else score
        
        # Temporal validation
        if final_score is not None:
            validated_score, reason = self.validate_score_temporal(final_score, player)
            if validated_score is not None:
                # Update last valid score
                self.last_valid_scores[player] = validated_score
                if self.debug:
                    print(f"  {player}: Score {validated_score} accepted ({reason})")
                return validated_score
            elif self.debug:
                print(f"  {player}: Score {final_score} rejected ({reason}), keeping {self.last_valid_scores[player]}")
        elif self.debug:
            print(f"  {player}: No final score to validate, returning last valid={self.last_valid_scores[player]}")
        
        # Return last valid score if current detection failed
        return self.last_valid_scores[player]

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

