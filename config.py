"""Configuration loader for NBA Jam video detection system."""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ScoreRegion:
    """Score region coordinates."""
    x: int
    y: int
    width: int
    height: int


@dataclass
class TextRegion:
    """Text region coordinates."""
    x: int
    y: int
    width: int
    height: int


@dataclass
class VideoConfig:
    """Video capture configuration."""
    device_path: str
    frame_interval: float
    capture_width: int
    capture_height: int
    use_grayscale: bool
    fourcc: str = "YUYV"
    # Resolution the config regions were measured at; regions are scaled to the
    # capture size on load. None means the regions already match the capture size.
    reference_width: Optional[int] = None
    reference_height: Optional[int] = None
    # Capture rate requested from the card; frames are only processed every frame_interval,
    # so a low rate saves the YUYV->BGR conversion work
    fps: Optional[int] = None


@dataclass
class DetectionConfig:
    """Detection configuration."""
    score_regions: Dict[str, ScoreRegion]
    motion_threshold: float
    confidence_threshold: float
    quarter_text_region: Optional[TextRegion] = None
    team_selection_heading_region: Optional[TextRegion] = None
    # Plain stretch of the purple overlay band (no text) used by the purple gate
    purple_band_region: Optional[TextRegion] = None
    # Learned glyph/phrase templates and full-screen scene references
    templates_path: str = "templates/nbajam.npz"
    scenes_path: str = "templates/scenes.npz"


@dataclass
class MQTTConfig:
    """MQTT configuration."""
    enabled: bool
    broker: str
    port: int
    username: Optional[str]
    password: Optional[str]
    topic_prefix: str


@dataclass
class TuningConfig:
    """Tuning and monitoring configuration."""
    monitor_cpu: bool
    metrics_interval: int


@dataclass
class ScreenshotConfig:
    """Screenshot capture configuration."""
    enabled: bool
    directory: str
    max_count: int  # Maximum number of screenshots to keep
    max_age_days: int  # Maximum age in days before deletion


def _scale_regions(detection: DetectionConfig, video: VideoConfig) -> None:
    """Scale every region from the reference resolution to the capture resolution."""
    ref_w = video.reference_width or video.capture_width
    ref_h = video.reference_height or video.capture_height
    sx = video.capture_width / ref_w
    sy = video.capture_height / ref_h
    if sx == 1 and sy == 1:
        return

    def scale(r):
        if r is None:
            return
        r.x, r.width = round(r.x * sx), round(r.width * sx)
        r.y, r.height = round(r.y * sy), round(r.height * sy)

    for r in detection.score_regions.values():
        scale(r)
    scale(detection.quarter_text_region)
    scale(detection.team_selection_heading_region)
    scale(detection.purple_band_region)


@dataclass
class Config:
    """Main configuration class."""
    video: VideoConfig
    detection: DetectionConfig
    mqtt: MQTTConfig
    tuning: TuningConfig
    screenshots: ScreenshotConfig

    @classmethod
    def load(cls, config_path: str = "config.yaml") -> "Config":
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Please create it from config.yaml.example"
            )

        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)

        # Load video config
        video_data = data.get('video', {})
        video_config = VideoConfig(
            device_path=video_data.get('device_path', '/dev/video0'),
            frame_interval=video_data.get('frame_interval', 1.0),
            capture_width=video_data.get('capture_width', 1920),
            capture_height=video_data.get('capture_height', 1080),
            use_grayscale=video_data.get('use_grayscale', True),
            fourcc=video_data.get('fourcc', 'YUYV'),
            reference_width=video_data.get('reference_width'),
            reference_height=video_data.get('reference_height'),
            fps=video_data.get('fps')
        )

        # Load detection config
        detection_data = data.get('detection', {})
        score_regions_data = detection_data.get('score_regions', {})
        score_regions = {}
        for player, region_data in score_regions_data.items():
            score_regions[player] = ScoreRegion(
                x=region_data.get('x', 0),
                y=region_data.get('y', 0),
                width=region_data.get('width', 0),
                height=region_data.get('height', 0)
            )
        
        # Load text regions
        quarter_text_data = detection_data.get('quarter_text_region')
        quarter_text_region = None
        if quarter_text_data:
            quarter_text_region = TextRegion(
                x=quarter_text_data.get('x', 0),
                y=quarter_text_data.get('y', 0),
                width=quarter_text_data.get('width', 0),
                height=quarter_text_data.get('height', 0)
            )
        
        team_selection_heading_data = detection_data.get('team_selection_heading_region')
        team_selection_heading_region = None
        if team_selection_heading_data:
            team_selection_heading_region = TextRegion(
                x=team_selection_heading_data.get('x', 0),
                y=team_selection_heading_data.get('y', 0),
                width=team_selection_heading_data.get('width', 0),
                height=team_selection_heading_data.get('height', 0)
            )
        
        purple_band_data = detection_data.get('purple_band_region')
        purple_band_region = None
        if purple_band_data:
            purple_band_region = TextRegion(
                x=purple_band_data.get('x', 0),
                y=purple_band_data.get('y', 0),
                width=purple_band_data.get('width', 0),
                height=purple_band_data.get('height', 0)
            )

        detection_config = DetectionConfig(
            score_regions=score_regions,
            motion_threshold=detection_data.get('motion_threshold', 0.1),
            confidence_threshold=detection_data.get('confidence_threshold', 0.7),
            quarter_text_region=quarter_text_region,
            team_selection_heading_region=team_selection_heading_region,
            purple_band_region=purple_band_region,
            templates_path=detection_data.get('templates_path', 'templates/nbajam.npz'),
            scenes_path=detection_data.get('scenes_path', 'templates/scenes.npz')
        )
        _scale_regions(detection_config, video_config)

        # Load MQTT config
        mqtt_data = data.get('mqtt', {})
        username = mqtt_data.get('username')
        password = mqtt_data.get('password')
        if username == 'null' or username == '':
            username = None
        if password == 'null' or password == '':
            password = None

        mqtt_config = MQTTConfig(
            enabled=mqtt_data.get('enabled', False),
            broker=mqtt_data.get('broker', 'localhost'),
            port=mqtt_data.get('port', 1883),
            username=username,
            password=password,
            topic_prefix=mqtt_data.get('topic_prefix', 'nba_jam')
        )

        # Load tuning config
        tuning_data = data.get('tuning', {})
        tuning_config = TuningConfig(
            monitor_cpu=tuning_data.get('monitor_cpu', False),
            metrics_interval=tuning_data.get('metrics_interval', 10)
        )

        # Load screenshot config
        screenshot_data = data.get('screenshots', {})
        screenshot_config = ScreenshotConfig(
            enabled=screenshot_data.get('enabled', False),
            directory=os.path.expanduser(screenshot_data.get('directory', '~/nba-jam-screenshots')),
            max_count=screenshot_data.get('max_count', 100),
            max_age_days=screenshot_data.get('max_age_days', 7)
        )

        return cls(
            video=video_config,
            detection=detection_config,
            mqtt=mqtt_config,
            tuning=tuning_config,
            screenshots=screenshot_config
        )

