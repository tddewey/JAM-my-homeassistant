#!/usr/bin/env python3
"""Cleanup script to remove MQTT auto-discovery entities from Home Assistant."""

import sys
import time
from config import Config
from mqtt_client import MQTTClient


def main():
    """Main cleanup function."""
    print("NBA Jam MQTT Cleanup Script")
    print("=" * 40)
    
    # Load configuration
    try:
        config = Config.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Enable MQTT temporarily for cleanup (even if disabled in config)
    original_enabled = config.mqtt.enabled
    config.mqtt.enabled = True
    
    # Create MQTT client
    mqtt_client = MQTTClient(config.mqtt)
    
    # Connect
    print(f"Connecting to MQTT broker: {config.mqtt.broker}:{config.mqtt.port}...")
    if not mqtt_client.connect():
        print("Failed to connect to MQTT broker")
        sys.exit(1)
    
    # Wait a moment for connection to stabilize
    time.sleep(1)
    
    # Publish cleanup configs
    print("Publishing cleanup configs to remove entities...")
    mqtt_client.cleanup_configs()
    
    # Wait for messages to be sent
    time.sleep(2)
    
    # Disconnect
    mqtt_client.disconnect()
    
    # Restore original config
    config.mqtt.enabled = original_enabled
    
    print("Cleanup complete!")
    print("Entities should be removed from Home Assistant.")


if __name__ == "__main__":
    main()

