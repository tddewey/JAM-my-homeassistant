"""Visualization and calibration tools for detection regions."""

import cv2
import numpy as np
from typing import Dict, Optional
from config import DetectionConfig, ScoreRegion
from detectors.state_detector import GameState


def draw_detection_regions(
    frame: np.ndarray,
    config: DetectionConfig,
    scores: Dict[str, Optional[int]],
    state: GameState
) -> np.ndarray:
    """Draw detection regions and information on frame.
    
    Args:
        frame: Video frame
        config: Detection configuration
        scores: Dictionary with player1 and player2 scores
        state: Current game state
        
    Returns:
        Frame with overlays
    """
    overlay = frame.copy()
    
    # Draw score regions
    p1_region = config.score_regions.get('player1')
    p2_region = config.score_regions.get('player2')
    
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
    if config.quarter_text_region:
        cv2.rectangle(
            overlay,
            (config.quarter_text_region.x, config.quarter_text_region.y),
            (config.quarter_text_region.x + config.quarter_text_region.width,
             config.quarter_text_region.y + config.quarter_text_region.height),
            (0, 255, 255),  # Yellow
            2
        )
        cv2.putText(
            overlay,
            "Quarter Text",
            (config.quarter_text_region.x, config.quarter_text_region.y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
    
    # Draw team selection heading region (magenta)
    if config.team_selection_heading_region:
        cv2.rectangle(
            overlay,
            (config.team_selection_heading_region.x, config.team_selection_heading_region.y),
            (config.team_selection_heading_region.x + config.team_selection_heading_region.width,
             config.team_selection_heading_region.y + config.team_selection_heading_region.height),
            (255, 0, 255),  # Magenta
            2
        )
        cv2.putText(
            overlay,
            "Team Selection Heading",
            (config.team_selection_heading_region.x, config.team_selection_heading_region.y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2
        )
    
    return overlay


def draw_info_overlay(
    frame: np.ndarray,
    state: GameState,
    scores: Dict[str, Optional[int]],
    fps: float,
    cpu_percent: float,
    resolution: tuple,
    codec_info: Optional[Dict[str, str]] = None
) -> np.ndarray:
    """Draw information overlay on frame.
    
    Args:
        frame: Video frame
        state: Current game state
        scores: Dictionary with player1 and player2 scores
        fps: Frames per second
        cpu_percent: CPU usage percentage
        resolution: Frame resolution (width, height)
        codec_info: Optional codec information
        
    Returns:
        Frame with info overlay
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
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
        GameState.TEAM_SELECTION: (255, 255, 0),  # Cyan
        GameState.IN_PROGRESS: (0, 255, 0),      # Green
        GameState.HALF_TIME: (0, 165, 255),       # Orange
        GameState.GAME_OVER: (0, 0, 255)          # Red
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
    
    # Performance info
    cv2.putText(
        overlay,
        f"FPS: {fps:.1f}",
        (20, y_start),
        font,
        font_scale,
        color,
        thickness
    )
    y_start += line_height
    
    cv2.putText(
        overlay,
        f"CPU: {cpu_percent:.1f}%",
        (20, y_start),
        font,
        font_scale,
        (0, 255, 0) if cpu_percent < 80 else (0, 0, 255),
        thickness
    )
    y_start += line_height
    
    # Resolution
    cv2.putText(
        overlay,
        f"Resolution: {resolution[0]}x{resolution[1]}",
        (20, y_start),
        font,
        font_scale,
        color,
        thickness
    )
    y_start += line_height
    
    # Codec info
    if codec_info:
        cv2.putText(
            overlay,
            f"Codec: {codec_info['name']}",
            (20, y_start),
            font,
            font_scale,
            color,
            thickness
        )
    
    return overlay


def create_visualization_frame(
    frame: np.ndarray,
    config: DetectionConfig,
    scores: Dict[str, Optional[int]],
    state: GameState,
    fps: float,
    cpu_percent: float,
    codec_info: Optional[Dict[str, str]] = None
) -> np.ndarray:
    """Create complete visualization frame with all overlays.
    
    Args:
        frame: Video frame
        config: Detection configuration
        scores: Dictionary with player1 and player2 scores
        state: Current game state
        fps: Frames per second
        cpu_percent: CPU usage percentage
        codec_info: Optional codec information
        
    Returns:
        Frame with all visualizations
    """
    # Draw detection regions
    vis_frame = draw_detection_regions(frame, config, scores, state)
    
    # Draw info overlay
    resolution = (frame.shape[1], frame.shape[0])
    vis_frame = draw_info_overlay(
        vis_frame,
        state,
        scores,
        fps,
        cpu_percent,
        resolution,
        codec_info
    )
    
    return vis_frame

