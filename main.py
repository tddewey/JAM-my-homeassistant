#!/usr/bin/env python3
"""Main entry point for NBA Jam video detection system."""

import sys
import time
import signal
import argparse
import cv2
from datetime import datetime
from colorama import init, Fore, Style

from config import Config
from video_capture import VideoCapture
from detectors.score_detector import ScoreDetector
from detectors.state_detector import StateDetector, GameState
from mqtt_client import MQTTClient
from performance_monitor import PerformanceMonitor
from tuning_tools import create_visualization_frame

# Initialize colorama for colored console output
init(autoreset=True)


class NBAJamDetector:
    """Main detector class that orchestrates all components."""

    def __init__(self, config: Config, monitor_cpu: bool = False, 
                 visualize: bool = False, metrics_interval: int = 10):
        """Initialize detector.
        
        Args:
            config: Configuration object
            monitor_cpu: Override config to enable CPU monitoring
            visualize: Override config to enable visualization
            metrics_interval: Override config for metrics interval
        """
        self.config = config
        self.running = False
        
        # Override config with CLI args if provided
        self.monitor_cpu = monitor_cpu or config.tuning.monitor_cpu
        self.visualize = visualize or config.tuning.visualize
        self.metrics_interval = metrics_interval or config.tuning.metrics_interval
        
        # Initialize components
        self.video_capture = VideoCapture(config.video)
        self.score_detector = ScoreDetector(config.detection)
        self.state_detector = StateDetector(config.detection)
        self.mqtt_client = MQTTClient(config.mqtt)
        self.performance_monitor = PerformanceMonitor() if self.monitor_cpu else None
        
        # Track last published values
        self.last_published_state = None
        self.last_published_scores = {'player1': None, 'player2': None}
        self.last_metrics_display = time.time()

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
            GameState.TEAM_SELECTION: Fore.CYAN,
            GameState.IN_PROGRESS: Fore.GREEN,
            GameState.HALF_TIME: Fore.YELLOW,
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
        
        if self.visualize:
            self.log("Visualization mode enabled (press 'q' to quit)", Fore.GREEN)
        
        self.log("Detection started. Press Ctrl+C to stop.", Fore.GREEN)
        self.log("=" * 60)
        
        self.running = True
        
        try:
            while self.running:
                frame_start = time.time()
                
                # Read frame
                frame_result = self.video_capture.read_frame()
                
                if frame_result is None:
                    time.sleep(0.1)
                    continue
                
                frame, timestamp = frame_result
                
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
                
                # Visualization mode
                if self.visualize:
                    fps = self.performance_monitor.get_fps() if self.performance_monitor else 0.0
                    cpu = self.performance_monitor.get_cpu_percent() if self.performance_monitor else 0.0
                    
                    vis_frame = create_visualization_frame(
                        frame,
                        self.config.detection,
                        scores,
                        state,
                        fps,
                        cpu,
                        self.video_capture.codec_info
                    )
                    
                    cv2.imshow('NBA Jam Detection', vis_frame)
                    
                    # Check for 'q' key to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.log("Quit key pressed", Fore.YELLOW)
                        break
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                
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
        if self.visualize:
            cv2.destroyAllWindows()
        if self.performance_monitor:
            self.log("\n" + self.performance_monitor.get_metrics_summary(), Fore.CYAN)
        self.log("Cleanup complete", Fore.CYAN)

    def stop(self):
        """Stop the detector."""
        self.running = False


def signal_handler(sig, frame):
    """Handle interrupt signal."""
    print("\nInterrupt received, shutting down...")
    sys.exit(0)


def main():
    """Main entry point."""
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='NBA Jam Video Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Normal mode
  python main.py --monitor-cpu            # With CPU monitoring
  python main.py --visualize              # Visualization mode
  python main.py --monitor-cpu --visualize # Both modes
  python main.py --metrics-interval 5     # Custom metrics interval
        """
    )
    parser.add_argument(
        '--monitor-cpu',
        action='store_true',
        help='Enable CPU and performance monitoring'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Enable visualization mode (shows detection regions)'
    )
    parser.add_argument(
        '--metrics-interval',
        type=int,
        default=None,
        help='Interval between metrics display in seconds (default: from config)'
    )
    
    args = parser.parse_args()
    
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
        visualize=args.visualize,
        metrics_interval=args.metrics_interval
    )
    detector.run()


if __name__ == "__main__":
    main()
