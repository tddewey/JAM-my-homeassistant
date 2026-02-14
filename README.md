# NBA Jam Video Detection System for Home Assistant

This system captures video from an NBA Jam arcade cabinet via HDMI capture card, detects game state and scores, and publishes MQTT messages for Home Assistant integration.

**GitHub Repository:** https://github.com/tddewey/JAM-my-homeassistant

## Features

- **Game State Detection**: Detects `team_selection`, `in_progress`, `half_time`, and `game_over` states
- **Score Detection**: Tracks Player 1 and Player 2 scores (handles cases where scores are off-screen)
- **MQTT Integration**: Home Assistant auto-discovery format with enable/disable flag
- **Console Logging**: Verbose logging for testing and validation
- **Performance Optimizations**: Reduced resolution, grayscale conversion, optimized processing
- **CPU Monitoring**: Real-time CPU and performance metrics
- **Codec Detection**: Identify capture device codec (MJPEG, YUYV, H.264, etc.)
- **Cleanup Script**: Remove MQTT entities after use

## Requirements

- Raspberry Pi Zero W2 (or compatible)
- HDMI capture card (USB-based)
- Python 3.8+
- Tesseract OCR installed on system

## Installation

### Clone the Repository

On your Raspberry Pi (Debian Trixie), clone the repository to a suitable location:

**Recommended location:** `/opt/nba-jam-detector` (for system-wide installation)
Requires `sudo` for initial setup, but provides better organization.
```bash
sudo git clone https://github.com/tddewey/JAM-my-homeassistant.git /opt/nba-jam-detector
sudo chown -R $USER:$USER /opt/nba-jam-detector
cd /opt/nba-jam-detector
```

**Alternative location:** User home directory (for user-specific installation)
Suitable for user-specific installation and testing. Easier to manage without `sudo`.
```bash
cd ~
git clone https://github.com/tddewey/JAM-my-homeassistant.git
cd JAM-my-homeassistant
```


### Install Dependencies

1. Install system dependencies:
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr libtesseract-dev python3-pip python3-venv python3-full
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

**Note:** You'll need to activate the virtual environment each time you work with the project:
```bash
source venv/bin/activate
```

To deactivate the virtual environment when you're done:
```bash
deactivate
```

**Troubleshooting:** If you get an "externally-managed-environment" error even after activating the venv, the venv was likely created before `python3-full` was installed. Fix it by:

1. Deactivate the venv (if active):
```bash
deactivate
```

2. Remove the old venv and recreate it:
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Make sure `python3-full` is installed before recreating the venv:
```bash
sudo apt-get install -y python3-full
```

### Configuration

3. Create your configuration file:
```bash
cp config.yaml.example config.yaml
```

4. Edit `config.yaml` to configure:
   - Video capture device path (typically `/dev/video0` or `/dev/video1`)
   - Score detection regions (player1 and player2)
   - Quarter/period text detection region
   - Team selection heading detection region
   - MQTT broker settings (if enabled)
   - Performance tuning settings

   **Note:** Detection regions may vary depending on your video capture setup. Adjust the coordinates in the config file based on your screen layout.



## Usage

### Device Diagnostics

Before running the main script, check your capture device:
```bash
source venv/bin/activate
python diagnose_device.py /dev/video0
```

This will show:
- Device codec (MJPEG, YUYV, H.264, etc.)
- Supported resolutions
- Performance recommendations

### Main Detection Script

Run the main detection script:
```bash
source venv/bin/activate
python main.py
```

**Command-line options:**
```bash
# Normal mode
python main.py

# With CPU monitoring
python main.py --monitor-cpu

# Enable screenshot capture
python main.py --save-screenshots

# Custom metrics interval
python main.py --monitor-cpu --metrics-interval 5
```

The script will:
- Capture video from the configured device
- Detect game state and scores
- Log all detections to console (verbose by default)
- Display codec information at startup
- Show performance metrics periodically (if enabled)
- Optionally publish to MQTT if enabled in config

### How to Stop the Detector

**Foreground Mode (running without `&`):**
- Press `Ctrl+C` to stop the detector gracefully

**Background Mode (running with `&`):**
When running in the background with `python main.py &`, Ctrl+C won't work. Use one of these methods:

1. **Using PID file (recommended):**
   ```bash
   kill $(cat ~/.nba-jam-detector.pid)
   ```

2. **Using process name:**
   ```bash
   pkill -f "python.*main.py"
   ```

3. **Find and kill manually:**
   ```bash
   ps aux | grep "python.*main.py"
   kill <PID>
   ```

The PID file is automatically created at `~/.nba-jam-detector.pid` (or custom path with `--pid-file`). It's automatically cleaned up when the process exits normally.

### Screenshot Capture

The detector can capture annotated screenshots when game state changes. This is useful for debugging detection regions and verifying accuracy.

**Enable Screenshot Capture:**

1. **Via CLI flag:**
   ```bash
   python main.py --save-screenshots
   ```

2. **Via config file:**
   Edit `config.yaml` and set:
   ```yaml
   screenshots:
     enabled: true
   ```

**Configuration Options:**

Screenshots are configured in `config.yaml`:
```yaml
screenshots:
  enabled: false                    # Enable/disable (can override with --save-screenshots)
  directory: ~/nba-jam-screenshots  # Directory to save screenshots
  max_count: 100                    # Maximum number of screenshots to keep (0 = unlimited)
  max_age_days: 7                   # Maximum age in days before deletion (0 = no age limit)
```

**Screenshot Features:**
- Screenshots are captured automatically when game state changes
- Each screenshot is annotated with:
  - Detection region rectangles (score regions, text regions)
  - Current game state
  - Current scores
  - Timestamp
  - Frame resolution
- Old screenshots are automatically cleaned up based on `max_count` and `max_age_days`
- Screenshots are saved in the configured directory (default: `~/nba-jam-screenshots`)

**Screenshot Naming:**
Format: `screenshot_YYYY-MM-DD_HH-MM-SS_STATE_p1-SCORE_p2-SCORE.jpg`

Example: `screenshot_2026-02-14_15-30-45_q1_p1-10_p2-5.jpg`

**Note:** The screenshot directory defaults to your home directory (`~/nba-jam-screenshots`) for easy access via SFTP.

### Cleanup Script

After testing or party cleanup, remove MQTT entities:
```bash
source venv/bin/activate
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
- **Team Selection Heading**: Detects pre-game ("PRESS SHOOT TO SELECT") to trigger PLAYING state

The text appears reliably when:
- Scores update
- Quarters begin
- Quarters end

## Testing

1. **Diagnose device**: Run `diagnose_device.py` to check codec
2. **Configure text regions**: Set up `quarter_text_region` and `team_selection_heading_region` in config
3. **Start with MQTT disabled**: Set `mqtt.enabled: false` in config
4. **Monitor performance**: Use `--monitor-cpu` to tune frame_interval
5. **Validate detection**: Watch console output to verify accuracy
6. **Adjust regions**: Fine-tune score region coordinates in config based on console output
7. **Enable MQTT**: Once validated, set `mqtt.enabled: true`
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

