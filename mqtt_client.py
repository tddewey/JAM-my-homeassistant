"""MQTT client for Home Assistant auto-discovery."""

import json
import paho.mqtt.client as mqtt
from typing import Optional
from config import MQTTConfig


class MQTTClient:
    """MQTT client for Home Assistant integration."""

    def __init__(self, config: MQTTConfig):
        """Initialize MQTT client.
        
        Args:
            config: MQTT configuration
        """
        self.config = config
        self.client: Optional[mqtt.Client] = None
        self.connected = False
        self.config_published = False

    def connect(self) -> bool:
        """Connect to MQTT broker.
        
        Returns:
            True if connected successfully, False otherwise
        """
        if not self.config.enabled:
            print("MQTT is disabled in configuration")
            return False

        try:
            self.client = mqtt.Client(client_id="nba_jam_detector")
            
            # Set credentials if provided
            if self.config.username and self.config.password:
                self.client.username_pw_set(
                    self.config.username,
                    self.config.password
                )
            
            # Set callbacks
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            
            # Connect
            self.client.connect(
                self.config.broker,
                self.config.port,
                keepalive=60
            )
            
            # Start loop
            self.client.loop_start()
            
            # Wait for connection
            import time
            timeout = 5
            start = time.time()
            while not self.connected and (time.time() - start) < timeout:
                time.sleep(0.1)
            
            if self.connected:
                print(f"Connected to MQTT broker: {self.config.broker}:{self.config.port}")
                return True
            else:
                print("Failed to connect to MQTT broker (timeout)")
                return False
                
        except Exception as e:
            print(f"Error connecting to MQTT broker: {e}")
            return False

    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            self.connected = True
            print("MQTT connection established")
        else:
            print(f"MQTT connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback."""
        self.connected = False
        print("MQTT disconnected")

    def publish_config(self):
        """Publish Home Assistant auto-discovery configuration."""
        if not self.config.enabled or not self.connected:
            return

        base_topic = f"homeassistant/sensor/{self.config.topic_prefix}"
        device_info = {
            "identifiers": [f"{self.config.topic_prefix}_detector"],
            "name": "NBA Jam Detector",
            "model": "Raspberry Pi Zero W2",
            "manufacturer": "Custom"
        }

        # Game status sensor
        status_config = {
            "name": "NBA Jam Game Status",
            "state_topic": f"{base_topic}_game_status/state",
            "unique_id": f"{self.config.topic_prefix}_game_status",
            "device": device_info,
            "value_template": "{{ value_json.state }}"
        }
        self._publish(f"{base_topic}_game_status/config", json.dumps(status_config))

        # Player 1 score sensor
        p1_config = {
            "name": "NBA Jam Player 1 Score",
            "state_topic": f"{base_topic}_player1_score/state",
            "unique_id": f"{self.config.topic_prefix}_player1_score",
            "device": device_info,
            "unit_of_measurement": "points",
            "value_template": "{{ value_json.state }}"
        }
        self._publish(f"{base_topic}_player1_score/config", json.dumps(p1_config))

        # Player 2 score sensor
        p2_config = {
            "name": "NBA Jam Player 2 Score",
            "state_topic": f"{base_topic}_player2_score/state",
            "unique_id": f"{self.config.topic_prefix}_player2_score",
            "device": device_info,
            "unit_of_measurement": "points",
            "value_template": "{{ value_json.state }}"
        }
        self._publish(f"{base_topic}_player2_score/config", json.dumps(p2_config))

        self.config_published = True
        print("MQTT auto-discovery config published")

    def publish_game_status(self, state: str):
        """Publish game status.
        
        Args:
            state: Game state string
        """
        if not self.config.enabled or not self.connected:
            return

        base_topic = f"homeassistant/sensor/{self.config.topic_prefix}"
        payload = json.dumps({"state": state})
        self._publish(f"{base_topic}_game_status/state", payload)

    def publish_score(self, player: str, score: Optional[int]):
        """Publish score for a player.
        
        Args:
            player: 'player1' or 'player2'
            score: Score value or None (None values are not published)
        """
        if not self.config.enabled or not self.connected:
            return

        # Don't publish None scores - Home Assistant will retain last value
        if score is None:
            return

        base_topic = f"homeassistant/sensor/{self.config.topic_prefix}"
        payload = json.dumps({"state": score})
        
        if player == "player1":
            self._publish(f"{base_topic}_player1_score/state", payload)
        elif player == "player2":
            self._publish(f"{base_topic}_player2_score/state", payload)

    def _publish(self, topic: str, payload: str):
        """Internal publish method.
        
        Args:
            topic: MQTT topic
            payload: Message payload
        """
        if self.client and self.connected:
            result = self.client.publish(topic, payload, retain=True)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                print(f"Warning: Failed to publish to {topic}")

    def disconnect(self):
        """Disconnect from MQTT broker."""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            print("MQTT disconnected")

    def cleanup_configs(self):
        """Publish empty configs to remove entities from Home Assistant."""
        if not self.config.enabled or not self.connected:
            return

        base_topic = f"homeassistant/sensor/{self.config.topic_prefix}"
        
        # Publish empty configs to remove entities
        topics = [
            f"{base_topic}_game_status/config",
            f"{base_topic}_player1_score/config",
            f"{base_topic}_player2_score/config"
        ]
        
        for topic in topics:
            self._publish(topic, "")  # Empty payload removes entity
        
        print("MQTT cleanup configs published")

