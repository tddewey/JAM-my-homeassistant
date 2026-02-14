"""Game state detection module using text-based OCR detection."""

from typing import Optional
from enum import Enum
from detectors.text_detector import TextDetector
from config import DetectionConfig


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


class StateDetector:
    """Detects game state from video frames using text-based OCR detection."""

    def __init__(self, config: DetectionConfig):
        """Initialize state detector.
        
        Args:
            config: Detection configuration
        """
        self.config = config
        self.text_detector = TextDetector(config)
        self.current_state = GameState.NOT_PLAYING

    def detect_state(self, frame) -> GameState:
        """Detect current game state using text-based detection.
        
        Args:
            frame: Video frame
            
        Returns:
            Detected game state
        """
        # Try to detect team selection heading first
        heading_text = self.text_detector.detect_team_selection_heading(frame)
        
        if heading_text == "PRESS SHOOT TO SELECT":
            # Pre-game team selection - trigger PLAYING state
            self.current_state = GameState.PLAYING
            return self.current_state
        
        # Try to detect quarter/period text
        quarter_info = self.text_detector.detect_quarter_text(frame)
        
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
        # This is safer than guessing with unreliable heuristics
        return self.current_state

