# NBA Jam Video Detection System for Home Assistant

This system captures video from an NBA Jam arcade cabinet via HDMI capture card, detects game state and scores, and publishes MQTT messages for Home Assistant integration.

## Features

- **Game State Detection**: Detects `team_selection`, `in_progress`, `half_time`, and `game_over` states
- **Score Detection**: Tracks Player 1 and Player 2 scores (handles cases where scores are off-screen)
- **MQTT Integration**: Home Assistant auto-discovery format with enable/disable flag
- **Console Logging**: Verbose logging for testing and validation
- **Performance Optimizations**: Reduced resolution, grayscale conversion, optimized processing
- **CPU Monitoring**: Real-time CPU and performance metrics
- **Visualization Mode**: Visual overlay showing detection regions and metrics
- **Codec Detection**: Identify capture device codec (MJPEG, YUYV, H.264, etc.)
- **Cleanup Script**: Remove MQTT entities after use

## Requirements

- Raspberry Pi Zero W2 (or compatible)
- HDMI capture card (USB-based)
- Python 3.8+
- Tesseract OCR installed on system

## Installation

1. Install system dependencies:
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr libtesseract-dev
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the system by editing `config.yaml` (create from `config.yaml.example`). This includes confirming the detection regions, which may be different depending on your video shaders and other retroarch configurations.

## Usage

### Device Diagnostics

Before running the main script, check your capture device:
```bash
python diagnose_device.py /dev/video0
```

This will show:
- Device codec (MJPEG, YUYV, H.264, etc.)
- Supported resolutions
- Performance recommendations

### Main Detection Script

Run the main detection script:
```bash
python main.py
```

**Command-line options:**
```bash
# Normal mode
python main.py

# With CPU monitoring
python main.py --monitor-cpu

# Visualization mode (shows detection regions)
python main.py --visualize

# Combined modes
python main.py --monitor-cpu --visualize

# Custom metrics interval
python main.py --monitor-cpu --metrics-interval 5
```

The script will:
- Capture video from the configured device
- Detect game state and scores
- Log all detections to console (verbose by default)
- Display codec information at startup
- Show performance metrics periodically (if enabled)
- Display visualization window (if enabled, press 'q' to quit)
- Optionally publish to MQTT if enabled in config

### Cleanup Script

After testing or party cleanup, remove MQTT entities:
```bash
python cleanup_mqtt.py
```

This publishes empty config messages to remove all NBA Jam entities from Home Assistant.

## Performance Tuning

### Optimize Frame Rate

1. **Check your codec**: Run `diagnose_device.py` to see what codec your device uses
   - **YUYV**: Best for Pi Zero W2 (fastest processing)
   - **MJPEG**: Good balance (moderate CPU)
   - **H.264**: May be too CPU-intensive, consider a different dongle

2. **Reduce resolution**: Lower `capture_width` and `capture_height` in config
   - 1280x720: ~44% faster than 1920x1080
   - 640x480: ~89% faster than 1920x1080

3. **Enable grayscale**: Set `use_grayscale: true` (default)
   - Reduces memory and processing time

4. **Monitor CPU**: Use `--monitor-cpu` flag to see performance
   - Aim for <80% CPU usage
   - Adjust `frame_interval` based on CPU load

5. **Use visualization**: Run with `--visualize` to see detection regions
   - Helps calibrate score region coordinates
   - Shows real-time FPS and CPU usage

## Text-Based State Detection

The system uses OCR (Optical Character Recognition) to detect game state by reading on-screen text. This is more reliable than heuristic-based detection.

### Configuration

Configure text detection regions in `config.yaml`:

```yaml
detection:
  # Region where quarter/period text appears
  quarter_text_region:
    x: 600
    y: 100
    width: 200
    height: 50
  
  # Region for team selection heading
  team_selection_heading_region:
    x: 400
    y: 50
    width: 300
    height: 40
```

### How It Works

- **Quarter/Period Text**: Detects text like "1st quarter", "2nd quarter", "halftime", "3rd quarter", "4th quarter", "final", "end of first quarter", etc.
- **Team Selection Heading**: Distinguishes pre-game ("PRESS SHOOT TO SELECT") from halftime ("SUBSTITUTIONS")

The text appears reliably when:
- Scores update
- Quarters begin
- Quarters end

## Testing

1. **Diagnose device**: Run `diagnose_device.py` to check codec
2. **Configure text regions**: Set up `quarter_text_region` and `team_selection_heading_region` in config
3. **Start with MQTT disabled**: Set `mqtt.enabled: false` in config
4. **Use visualization mode**: Run `python main.py --visualize` to calibrate regions
5. **Monitor performance**: Use `--monitor-cpu` to tune frame_interval
6. **Validate detection**: Watch console output to verify accuracy
7. **Adjust regions**: Fine-tune score region coordinates in config
8. **Enable MQTT**: Once validated, set `mqtt.enabled: true`
9. **Cleanup**: Use cleanup script to remove entities when done

## Project Structure

```
.
├── main.py                 # Main entry point
├── config.py               # Configuration loader
├── video_capture.py        # Video capture wrapper
├── mqtt_client.py          # MQTT publisher
├── cleanup_mqtt.py         # MQTT cleanup script
├── diagnose_device.py      # Device codec diagnostic tool
├── performance_monitor.py   # CPU and performance metrics
├── tuning_tools.py         # Visualization utilities
├── detectors/
│   ├── __init__.py
│   ├── score_detector.py   # Score detection using OCR
│   ├── state_detector.py    # State detection using text OCR
│   └── text_detector.py      # Text detection utilities
├── requirements.txt
├── config.yaml             # Configuration file (create from config.yaml.example)
├── config.yaml.example     # Example configuration
└── README.md
```

