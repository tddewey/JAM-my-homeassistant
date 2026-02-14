"""Screenshot capture and management module."""

import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict
from config import ScreenshotConfig, DetectionConfig, ScoreRegion, TextRegion
from detectors.state_detector import GameState


class ScreenshotManager:
    """Manages screenshot capture, annotation, and cleanup."""

    def __init__(self, config: ScreenshotConfig):
        """Initialize screenshot manager.
        
        Args:
            config: Screenshot configuration
        """
        self.config = config
        self.screenshot_dir = Path(config.directory)
        self.screenshot_count = 0
        
        # Create screenshot directory if it doesn't exist
        try:
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create screenshot directory: {e}")

    def capture_screenshot(
        self,
        frame: np.ndarray,
        state: GameState,
        scores: Dict[str, Optional[int]],
        detection_config: DetectionConfig
    ) -> Optional[str]:
        """Capture and save an annotated screenshot.
        
        Args:
            frame: Video frame to capture
            state: Current game state
            scores: Dictionary with player1 and player2 scores
            detection_config: Detection configuration for annotation
            
        Returns:
            Path to saved screenshot, or None if capture failed
        """
        if frame is None or frame.size == 0:
            return None
        
        try:
            # Annotate frame
            annotated_frame = self._annotate_frame(frame, state, scores, detection_config)
            
            # Generate filename
            timestamp = datetime.now()
            filename = self._get_screenshot_path(state, scores, timestamp)
            filepath = self.screenshot_dir / filename
            
            # Save screenshot
            cv2.imwrite(str(filepath), annotated_frame)
            
            self.screenshot_count += 1
            
            # Cleanup periodically (every 10 screenshots or if approaching max_count)
            if self.screenshot_count % 10 == 0 or (
                self.config.max_count > 0 and 
                self._count_screenshots() > int(self.config.max_count * 0.9)
            ):
                self.cleanup_old_screenshots()
            
            return str(filepath)
        except Exception as e:
            print(f"Warning: Failed to capture screenshot: {e}")
            return None

    def _annotate_frame(
        self,
        frame: np.ndarray,
        state: GameState,
        scores: Dict[str, Optional[int]],
        detection_config: DetectionConfig
    ) -> np.ndarray:
        """Annotate frame with detection regions and info.
        
        Args:
            frame: Video frame
            state: Current game state
            scores: Dictionary with player1 and player2 scores
            detection_config: Detection configuration
            
        Returns:
            Annotated frame
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw score regions
        p1_region = detection_config.score_regions.get('player1')
        p2_region = detection_config.score_regions.get('player2')
        
        # Player 1 region (green)
        if p1_region:
            cv2.rectangle(
                overlay,
                (p1_region.x, p1_region.y),
                (p1_region.x + p1_region.width, p1_region.y + p1_region.height),
                (0, 255, 0),  # Green
                2
            )
            # Label
            p1_score = scores.get('player1')
            score_text = str(p1_score) if p1_score is not None else "N/A"
            cv2.putText(
                overlay,
                f"P1: {score_text}",
                (p1_region.x, p1_region.y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        # Player 2 region (blue)
        if p2_region:
            cv2.rectangle(
                overlay,
                (p2_region.x, p2_region.y),
                (p2_region.x + p2_region.width, p2_region.y + p2_region.height),
                (255, 0, 0),  # Blue
                2
            )
            # Label
            p2_score = scores.get('player2')
            score_text = str(p2_score) if p2_score is not None else "N/A"
            cv2.putText(
                overlay,
                f"P2: {score_text}",
                (p2_region.x, p2_region.y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )
        
        # Draw quarter text region (yellow)
        if detection_config.quarter_text_region:
            cv2.rectangle(
                overlay,
                (detection_config.quarter_text_region.x, detection_config.quarter_text_region.y),
                (detection_config.quarter_text_region.x + detection_config.quarter_text_region.width,
                 detection_config.quarter_text_region.y + detection_config.quarter_text_region.height),
                (0, 255, 255),  # Yellow
                2
            )
            cv2.putText(
                overlay,
                "Quarter Text",
                (detection_config.quarter_text_region.x, detection_config.quarter_text_region.y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
        
        # Draw team selection heading region (magenta)
        if detection_config.team_selection_heading_region:
            cv2.rectangle(
                overlay,
                (detection_config.team_selection_heading_region.x, detection_config.team_selection_heading_region.y),
                (detection_config.team_selection_heading_region.x + detection_config.team_selection_heading_region.width,
                 detection_config.team_selection_heading_region.y + detection_config.team_selection_heading_region.height),
                (255, 0, 255),  # Magenta
                2
            )
            cv2.putText(
                overlay,
                "Team Selection Heading",
                (detection_config.team_selection_heading_region.x, detection_config.team_selection_heading_region.y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2
            )
        
        # Add info overlay
        overlay = self._draw_info_overlay(overlay, state, scores, w, h)
        
        return overlay

    def _draw_info_overlay(
        self,
        frame: np.ndarray,
        state: GameState,
        scores: Dict[str, Optional[int]],
        width: int,
        height: int
    ) -> np.ndarray:
        """Draw information overlay on frame.
        
        Args:
            frame: Video frame
            state: Current game state
            scores: Dictionary with player1 and player2 scores
            width: Frame width
            height: Frame height
            
        Returns:
            Frame with info overlay
        """
        overlay = frame.copy()
        
        # Create semi-transparent background for text
        overlay_bg = overlay.copy()
        cv2.rectangle(overlay_bg, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay_bg, 0.7, overlay, 0.3, 0, overlay)
        
        # Text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 1
        line_height = 25
        y_start = 35
        
        # Game state
        state_color = {
            GameState.NOT_PLAYING: (128, 128, 128),  # Gray
            GameState.PLAYING: (255, 255, 0),        # Cyan
            GameState.Q1: (0, 255, 0),               # Green
            GameState.Q2: (0, 255, 0),               # Green
            GameState.Q3: (0, 255, 0),               # Green
            GameState.Q4: (0, 255, 0),               # Green
            GameState.HALFTIME: (0, 165, 255),       # Orange
            GameState.GAME_OVER: (0, 0, 255)         # Red
        }.get(state, (255, 255, 255))
        
        cv2.putText(
            overlay,
            f"State: {state.value}",
            (20, y_start),
            font,
            font_scale,
            state_color,
            thickness + 1
        )
        y_start += line_height
        
        # Scores
        p1_score = scores.get('player1')
        p2_score = scores.get('player2')
        p1_text = str(p1_score) if p1_score is not None else "N/A"
        p2_text = str(p2_score) if p2_score is not None else "N/A"
        cv2.putText(
            overlay,
            f"Scores: P1={p1_text}, P2={p2_text}",
            (20, y_start),
            font,
            font_scale,
            color,
            thickness
        )
        y_start += line_height
        
        # Timestamp
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            overlay,
            f"Time: {timestamp_str}",
            (20, y_start),
            font,
            font_scale,
            color,
            thickness
        )
        y_start += line_height
        
        # Resolution
        cv2.putText(
            overlay,
            f"Resolution: {width}x{height}",
            (20, y_start),
            font,
            font_scale,
            color,
            thickness
        )
        
        return overlay

    def _get_screenshot_path(
        self,
        state: GameState,
        scores: Dict[str, Optional[int]],
        timestamp: datetime
    ) -> str:
        """Generate screenshot filename.
        
        Args:
            state: Current game state
            scores: Dictionary with player1 and player2 scores
            timestamp: Timestamp for filename
            
        Returns:
            Filename string
        """
        timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        state_str = state.value
        
        p1_score = scores.get('player1')
        p2_score = scores.get('player2')
        p1_str = f"p1-{p1_score}" if p1_score is not None else "p1-NA"
        p2_str = f"p2-{p2_score}" if p2_score is not None else "p2-NA"
        
        return f"screenshot_{timestamp_str}_{state_str}_{p1_str}_{p2_str}.jpg"

    def cleanup_old_screenshots(self) -> None:
        """Clean up old screenshots based on max_count and max_age_days."""
        try:
            # Get all screenshot files
            screenshot_files = list(self.screenshot_dir.glob("screenshot_*.jpg"))
            
            if not screenshot_files:
                return
            
            # Sort by modification time (oldest first)
            screenshot_files.sort(key=lambda f: f.stat().st_mtime)
            
            current_time = datetime.now()
            deleted_count = 0
            
            # Delete by age
            if self.config.max_age_days > 0:
                max_age = timedelta(days=self.config.max_age_days)
                for filepath in screenshot_files[:]:
                    file_age = current_time - datetime.fromtimestamp(filepath.stat().st_mtime)
                    if file_age > max_age:
                        try:
                            filepath.unlink()
                            screenshot_files.remove(filepath)
                            deleted_count += 1
                        except Exception:
                            pass
            
            # Delete by count (after age-based deletion)
            if self.config.max_count > 0:
                remaining_files = [f for f in screenshot_files if f.exists()]
                if len(remaining_files) > self.config.max_count:
                    # Delete oldest files until we're under the limit
                    files_to_delete = remaining_files[:-self.config.max_count]
                    for filepath in files_to_delete:
                        try:
                            filepath.unlink()
                            deleted_count += 1
                        except Exception:
                            pass
            
            if deleted_count > 0:
                print(f"Cleaned up {deleted_count} old screenshot(s)")
        except Exception as e:
            print(f"Warning: Failed to cleanup screenshots: {e}")

    def _count_screenshots(self) -> int:
        """Count current number of screenshots.
        
        Returns:
            Number of screenshot files
        """
        try:
            return len(list(self.screenshot_dir.glob("screenshot_*.jpg")))
        except Exception:
            return 0

