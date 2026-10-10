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
from detectors.scene_detector import SceneDetector
from detectors.template_matcher import TemplateLibrary, TemplateRecognizer
from game_tracker import GameTracker, Event
from mqtt_client import MQTTClient
from performance_monitor import PerformanceMonitor
from screenshot_manager import ScreenshotManager

# Initialize colorama for colored console output
init(autoreset=True)

STATUS_COLORS = {
    'not_playing': "", 'team_select': Fore.CYAN, 'starting': Fore.CYAN, 'tipoff': Fore.GREEN,
    'q1': Fore.GREEN, 'q2': Fore.GREEN, 'q3': Fore.GREEN, 'q4': Fore.GREEN,
    'halftime': Fore.YELLOW, 'overtime': Fore.MAGENTA, 'game_over': Fore.RED,
}


class NBAJamDetector:
    """Reads frames, recognises scenes and the score overlay, and publishes game events."""

    def __init__(self, config: Config, monitor_cpu: bool = False,
                 metrics_interval: int = 10, save_screenshots: bool = False,
                 debug: bool = False):
        self.config = config
        self.running = False
        self.debug = debug

        self.monitor_cpu = monitor_cpu or config.tuning.monitor_cpu
        self.metrics_interval = metrics_interval or config.tuning.metrics_interval
        self.save_screenshots = save_screenshots or config.screenshots.enabled

        self.video_capture = VideoCapture(config.video)
        self.scenes = SceneDetector.load(config.detection.scenes_path)
        self.recognizer = TemplateRecognizer(config.detection, TemplateLibrary.load(config.detection.templates_path))
        self.tracker = GameTracker()
        self.mqtt_client = MQTTClient(config.mqtt)
        self.performance_monitor = PerformanceMonitor() if self.monitor_cpu else None
        self.screenshot_manager = ScreenshotManager(config.screenshots) if self.save_screenshots else None
        self.last_metrics_display = time.time()

    def log(self, message: str, color: str = ""):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if color:
            print(f"{color}[{timestamp}] {message}{Style.RESET_ALL}")
        else:
            print(f"[{timestamp}] {message}")

    def handle_event(self, event: Event):
        """Log an event and publish it to MQTT."""
        if event.kind == 'status':
            self.log(f"Game status: {event.status}", STATUS_COLORS.get(event.status, ""))
            self.mqtt_client.publish_game_status(event.status)
        elif event.kind == 'banner':
            if event.banner is not None:
                self.log(f"Banner: {event.banner}", Fore.CYAN)
            self.mqtt_client.publish_banner(event.banner)
        elif event.kind == 'scores':
            if event.player1 is not None and event.player2 is not None:
                self.log(f"Score: {event.player1} - {event.player2}", Fore.GREEN)
                self.mqtt_client.publish_score('player1', event.player1)
                self.mqtt_client.publish_score('player2', event.player2)
        elif event.kind == 'score':
            self.log(f"Player {event.scorer} scored {event.points} "
                     f"({event.player1}-{event.player2}, {event.quarter})", Fore.GREEN)
            self.mqtt_client.publish_score_event(event.scorer, event.points,
                                                 event.player1, event.player2, event.quarter)

    def process_frame(self, frame):
        """Run one frame through scene + overlay recognition and the tracker."""
        scene = self.scenes.classify(frame)
        reading = None
        if self.recognizer.base.has_purple_band(frame):
            reading = self.recognizer.read(frame)
        if self.debug:
            observed = (scene, reading)
            if observed != getattr(self, '_last_observed', None):    # only log changes
                self._last_observed = observed
                self.log(f"  saw scene={scene} overlay={reading}", Fore.BLUE)
        events = self.tracker.update(scene, reading)
        for event in events:
            self.handle_event(event)
        return scene, reading, events

    def run(self):
        """Run the main detection loop."""
        self.log("NBA Jam Video Detection System Starting...", Fore.CYAN)
        self.log("=" * 60)

        if not self.video_capture.open():
            self.log("Failed to open video capture device", Fore.RED)
            return

        if self.config.mqtt.enabled:
            if self.mqtt_client.connect():
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
                frame_result = self.video_capture.read_frame()
                if frame_result is None:
                    time.sleep(0.05)
                    continue
                frame, timestamp = frame_result
                if timestamp == getattr(self, '_last_ts', None):
                    time.sleep(0.01)      # no new frame since the last pass
                    continue
                self._last_ts = timestamp

                scene, reading, events = self.process_frame(frame)

                if self.screenshot_manager and events:
                    scores = {'player1': self.tracker.player1, 'player2': self.tracker.player2}
                    path = self.screenshot_manager.capture_screenshot(
                        frame.copy(), self.tracker.status, scores, self.config.detection)
                    if path:
                        self.log(f"Screenshot saved: {path}", Fore.CYAN)

                processing_time = time.time() - frame_start
                if self.performance_monitor:
                    self.performance_monitor.record_frame(processing_time)
                    now = time.time()
                    if now - self.last_metrics_display >= self.metrics_interval:
                        self.log("\n" + self.performance_monitor.get_metrics_summary(), Fore.CYAN)
                        self.last_metrics_display = now
                        if self.performance_monitor.get_cpu_percent() > 80:
                            self.log("WARNING: High CPU usage", Fore.RED)

                remaining = self.config.video.frame_interval - (time.time() - timestamp)
                if remaining > 0:
                    time.sleep(remaining)

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
        help='Log the scene, overlay reading and status for every frame'
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
