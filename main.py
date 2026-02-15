#!/usr/bin/env python3
"""Main entry point for NBA Jam video detection system."""

import sys
import os
import time
import signal
import argparse
import cv2
from datetime import datetime
from pathlib import Path
from colorama import init, Fore, Style

from config import Config
from video_capture import VideoCapture
from detectors.score_detector import ScoreDetector
from detectors.state_detector import StateDetector, GameState
from mqtt_client import MQTTClient
from performance_monitor import PerformanceMonitor
from screenshot_manager import ScreenshotManager

# Initialize colorama for colored console output
init(autoreset=True)


class NBAJamDetector:
    """Main detector class that orchestrates all components."""

    def __init__(self, config: Config, monitor_cpu: bool = False, 
                 metrics_interval: int = 10, save_screenshots: bool = False,
                 debug: bool = False):
        """Initialize detector.
        
        Args:
            config: Configuration object
            monitor_cpu: Override config to enable CPU monitoring
            metrics_interval: Override config for metrics interval
            save_screenshots: Override config to enable screenshot capture
            debug: Enable debug logging for score detection
        """
        self.config = config
        self.running = False
        
        # Override config with CLI args if provided
        self.monitor_cpu = monitor_cpu or config.tuning.monitor_cpu
        self.metrics_interval = metrics_interval or config.tuning.metrics_interval
        self.save_screenshots = save_screenshots or config.screenshots.enabled
        
        # Initialize components
        self.video_capture = VideoCapture(config.video)
        self.score_detector = ScoreDetector(config.detection, debug=debug)
        self.state_detector = StateDetector(config.detection)
        self.mqtt_client = MQTTClient(config.mqtt)
        self.performance_monitor = PerformanceMonitor() if self.monitor_cpu else None
        self.screenshot_manager = ScreenshotManager(config.screenshots) if self.save_screenshots else None
        
        # Track last published values
        self.last_published_state = None
        self.last_published_scores = {'player1': None, 'player2': None}
        self.last_metrics_display = time.time()
        
        # Track last state for screenshot capture
        self.last_state = None
        # Track last saved scores for screenshot capture on score changes
        self.last_saved_scores = {'player1': None, 'player2': None}

    def log(self, message: str, color: str = ""):
        """Log message with timestamp.
        
        Args:
            message: Message to log
            color: Colorama color code
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if color:
            print(f"{color}[{timestamp}] {message}{Style.RESET_ALL}")
        else:
            print(f"[{timestamp}] {message}")

    def log_detection(self, state: GameState, scores: dict, processing_time: float = 0.0):
        """Log detection results.
        
        Args:
            state: Current game state
            scores: Dictionary with player1 and player2 scores
            processing_time: Time taken to process frame
        """
        state_str = state.value
        state_color = {
            GameState.NOT_PLAYING: "",  # Default/Gray
            GameState.PLAYING: Fore.CYAN,  # Used for rest periods between quarters
            GameState.Q1: Fore.GREEN,
            GameState.Q2: Fore.GREEN,
            GameState.Q3: Fore.GREEN,
            GameState.Q4: Fore.GREEN,
            GameState.HALFTIME: Fore.YELLOW,
            GameState.GAME_OVER: Fore.RED
        }.get(state, "")
        
        # Format scores
        p1_score = scores.get('player1')
        p2_score = scores.get('player2')
        
        p1_str = str(p1_score) if p1_score is not None else "not detected"
        p2_str = str(p2_score) if p2_score is not None else "not detected"
        
        # Log state
        self.log(f"Game State: {state_str}", state_color)
        
        # Log scores
        if p1_score is None:
            self.log(f"  Player 1 Score: {p1_str} (Score not detected)", Fore.YELLOW)
        else:
            self.log(f"  Player 1 Score: {p1_str}", Fore.GREEN)
        
        if p2_score is None:
            self.log(f"  Player 2 Score: {p2_str} (Score not detected)", Fore.YELLOW)
        else:
            self.log(f"  Player 2 Score: {p2_str}", Fore.GREEN)
        
        # Log performance if monitoring
        if self.monitor_cpu and processing_time > 0:
            cpu = self.performance_monitor.get_cpu_percent()
            self.log(f"  Processing: {processing_time*1000:.1f}ms, CPU: {cpu:.1f}%", Fore.CYAN)

    def publish_updates(self, state: GameState, scores: dict):
        """Publish updates to MQTT if enabled.
        
        Args:
            state: Current game state
            scores: Dictionary with player1 and player2 scores
        """
        if not self.config.mqtt.enabled:
            return
        
        # Publish game state (always publish state changes)
        state_str = state.value
        if state_str != self.last_published_state:
            self.mqtt_client.publish_game_status(state_str)
            self.last_published_state = state_str
        
        # Publish scores (only if not None)
        p1_score = scores.get('player1')
        p2_score = scores.get('player2')
        
        if p1_score is not None and p1_score != self.last_published_scores['player1']:
            self.mqtt_client.publish_score('player1', p1_score)
            self.last_published_scores['player1'] = p1_score
        
        if p2_score is not None and p2_score != self.last_published_scores['player2']:
            self.mqtt_client.publish_score('player2', p2_score)
            self.last_published_scores['player2'] = p2_score

    def run(self):
        """Run the main detection loop."""
        self.log("NBA Jam Video Detection System Starting...", Fore.CYAN)
        self.log("=" * 60)
        
        # Open video capture
        if not self.video_capture.open():
            self.log("Failed to open video capture device", Fore.RED)
            return
        
        # Display codec info if available
        if self.video_capture.codec_info:
            codec = self.video_capture.codec_info
            self.log(f"Codec: {codec['name']} ({codec['fourcc']})", Fore.CYAN)
            self.log(f"  {codec['note']}", Fore.YELLOW)
        
        # Connect MQTT if enabled
        if self.config.mqtt.enabled:
            if self.mqtt_client.connect():
                # Wait a moment for connection
                time.sleep(1)
                self.mqtt_client.publish_config()
                self.log("MQTT enabled and connected", Fore.GREEN)
            else:
                self.log("MQTT connection failed, continuing without MQTT", Fore.YELLOW)
        else:
            self.log("MQTT disabled (console mode)", Fore.YELLOW)
        
        if self.monitor_cpu:
            self.log("CPU monitoring enabled", Fore.GREEN)
        
        self.log("Detection started. Press Ctrl+C to stop.", Fore.GREEN)
        self.log("=" * 60)
        
        self.running = True
        
        try:
            while self.running:
                frame_start = time.time()
                
                # Read frame
                frame_result = self.video_capture.read_frame()
                
                if frame_result is None:
                    self.log("Frame read: None (no frame available)", Fore.YELLOW)
                    time.sleep(0.1)
                    continue
                
                frame, timestamp = frame_result
                # Track when frame was read for interval enforcement
                frame_read_time = timestamp
                self.log(f"Frame read: timestamp={timestamp:.3f}", Fore.CYAN)
                
                # Use grayscale frame if available and configured
                if self.config.video.use_grayscale:
                    frame_gray = self.video_capture.get_frame_grayscale()
                    # Use grayscale for score detection (faster), color for state detection
                    score_frame = frame_gray if frame_gray is not None else frame
                    state_frame = frame  # State detection may need color
                else:
                    score_frame = frame
                    state_frame = frame
                
                # Detect game state
                state = self.state_detector.detect_state(state_frame)
                
                # Detect scores - score detector handles both grayscale and color
                scores = self.score_detector.detect_scores(score_frame)
                
                # TEMPORARILY: Save screenshot on every detection for debugging
                if self.screenshot_manager:
                    screenshot_path = self.screenshot_manager.capture_screenshot(
                        frame, state, scores, self.config.detection
                    )
                    if screenshot_path:
                        self.log(f"Screenshot saved (every detection): {screenshot_path}", Fore.CYAN)
                
                # Check for state change
                state_changed = state != self.last_state
                
                # Check for score changes
                p1_score = scores.get('player1')
                p2_score = scores.get('player2')
                score_changed = (
                    p1_score is not None and p1_score != self.last_saved_scores.get('player1')
                ) or (
                    p2_score is not None and p2_score != self.last_saved_scores.get('player2')
                )
                
                # Update tracking (keep this for state/score change tracking)
                if state_changed:
                    self.last_state = state
                if score_changed:
                    self.last_saved_scores['player1'] = p1_score
                    self.last_saved_scores['player2'] = p2_score
                elif self.last_state is None:
                    # Initialize last_state and last_saved_scores on first frame
                    self.last_state = state
                    self.last_saved_scores['player1'] = p1_score
                    self.last_saved_scores['player2'] = p2_score
                
                # Calculate processing time
                processing_time = time.time() - frame_start
                
                # Record performance metrics
                if self.performance_monitor:
                    self.performance_monitor.record_frame(processing_time)
                
                # Log detection results (verbose by default)
                self.log_detection(state, scores, processing_time)
                
                # Publish to MQTT if enabled
                self.publish_updates(state, scores)
                
                # Display metrics periodically
                if self.performance_monitor:
                    current_time = time.time()
                    if current_time - self.last_metrics_display >= self.metrics_interval:
                        self.log("\n" + self.performance_monitor.get_metrics_summary(), Fore.CYAN)
                        self.last_metrics_display = current_time
                        
                        # Warn if CPU is high
                        cpu = self.performance_monitor.get_cpu_percent()
                        if cpu > 80:
                            self.log(f"WARNING: High CPU usage ({cpu:.1f}%)", Fore.RED)
                
                # Enforce frame interval: sleep if we processed faster than configured interval
                elapsed_since_frame = time.time() - frame_read_time
                remaining_time = self.config.video.frame_interval - elapsed_since_frame
                if remaining_time > 0:
                    time.sleep(remaining_time)
                # If processing took longer than frame_interval, continue immediately
                
        except KeyboardInterrupt:
            self.log("\nShutting down...", Fore.YELLOW)
        except Exception as e:
            self.log(f"Error in main loop: {e}", Fore.RED)
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources."""
        self.running = False
        self.video_capture.release()
        self.mqtt_client.disconnect()
        if self.performance_monitor:
            self.log("\n" + self.performance_monitor.get_metrics_summary(), Fore.CYAN)
        self.log("Cleanup complete", Fore.CYAN)

    def stop(self):
        """Stop the detector."""
        self.running = False


def signal_handler(sig, frame, pid_file_path=None):
    """Handle interrupt signal."""
    print("\nInterrupt received, shutting down...")
    if pid_file_path and pid_file_path.exists():
        try:
            pid_file_path.unlink()
        except Exception:
            pass
    sys.exit(0)


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='NBA Jam Video Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Normal mode
  python main.py --monitor-cpu            # With CPU monitoring
  python main.py --save-screenshots       # Enable screenshot capture
  python main.py --debug                  # Enable debug logging for score detection
  python main.py --monitor-cpu --metrics-interval 5     # Custom metrics interval
        """
    )
    parser.add_argument(
        '--monitor-cpu',
        action='store_true',
        help='Enable CPU and performance monitoring'
    )
    parser.add_argument(
        '--metrics-interval',
        type=int,
        default=None,
        help='Interval between metrics display in seconds (default: from config)'
    )
    parser.add_argument(
        '--save-screenshots',
        action='store_true',
        help='Enable screenshot capture on state changes (overrides config)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging for score detection (shows OCR results and validation)'
    )
    parser.add_argument(
        '--pid-file',
        type=str,
        default=None,
        help='Path to PID file (default: ~/.nba-jam-detector.pid)'
    )
    
    args = parser.parse_args()
    
    # Set up PID file
    pid_file_path = None
    if args.pid_file:
        pid_file_path = Path(os.path.expanduser(args.pid_file))
    else:
        pid_file_path = Path.home() / '.nba-jam-detector.pid'
    
    # Write PID file
    try:
        pid_file_path.parent.mkdir(parents=True, exist_ok=True)
        pid_file_path.write_text(str(os.getpid()))
    except Exception as e:
        print(f"Warning: Could not write PID file: {e}")
    
    # Update signal handler to clean up PID file
    def cleanup_signal_handler(sig, frame):
        signal_handler(sig, frame, pid_file_path)
    
    signal.signal(signal.SIGINT, cleanup_signal_handler)
    signal.signal(signal.SIGTERM, cleanup_signal_handler)
    
    # Load configuration
    try:
        config = Config.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create and run detector
    detector = NBAJamDetector(
        config,
        monitor_cpu=args.monitor_cpu,
        metrics_interval=args.metrics_interval,
        save_screenshots=args.save_screenshots,
        debug=args.debug
    )
    
    try:
        detector.run()
    finally:
        # Clean up PID file on exit
        if pid_file_path and pid_file_path.exists():
            try:
                pid_file_path.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    main()
