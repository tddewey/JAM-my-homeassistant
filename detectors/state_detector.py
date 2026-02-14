"""Game state detection module using text-based OCR detection."""

from typing import Optional
from enum import Enum
from detectors.text_detector import TextDetector
from config import DetectionConfig


class GameState(Enum):
    """Game state enumeration."""
    TEAM_SELECTION = "team_selection"
    IN_PROGRESS = "in_progress"
    HALF_TIME = "half_time"
    GAME_OVER = "game_over"


class StateDetector:
    """Detects game state from video frames using text-based OCR detection."""

    def __init__(self, config: DetectionConfig):
        """Initialize state detector.
        
        Args:
            config: Detection configuration
        """
        self.config = config
        self.text_detector = TextDetector(config)
        self.current_state = GameState.TEAM_SELECTION

    def detect_state(self, frame) -> GameState:
        """Detect current game state using text-based detection.
        
        Args:
            frame: Video frame
            
        Returns:
            Detected game state
        """
        # Try to detect team selection heading first
        heading_text = self.text_detector.detect_team_selection_heading(frame)
        
        if heading_text:
            if heading_text == "PRESS SHOOT TO SELECT":
                # Pre-game team selection
                self.current_state = GameState.TEAM_SELECTION
                return self.current_state
            elif heading_text == "SUBSTITUTIONS":
                # Halftime team selection
                self.current_state = GameState.HALF_TIME
                return self.current_state
        
        # Try to detect quarter/period text
        quarter_info = self.text_detector.detect_quarter_text(frame)
        
        if quarter_info:
            state = quarter_info.get('state')
            
            if state == 'final':
                self.current_state = GameState.GAME_OVER
            elif state == 'halftime':
                self.current_state = GameState.HALF_TIME
            elif state in ('quarter', 'end_of_quarter'):
                # Game is in progress
                self.current_state = GameState.IN_PROGRESS
        
        # If no text detected, maintain current state
        # This is safer than guessing with unreliable heuristics
        return self.current_state

    def get_state(self) -> GameState:
        """Get current state without detection.
        
        Returns:
            Current game state
        """
        return self.current_state
