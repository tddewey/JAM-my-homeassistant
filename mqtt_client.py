"""MQTT client for Home Assistant auto-discovery."""

import json
import time
from typing import Dict, Optional

import paho.mqtt.client as mqtt

from config import MQTTConfig

# Every value game_status can take (also the enum options HA shows)
STATUS_OPTIONS = [
    "not_playing", "team_select", "starting", "tipoff",
    "q1", "q2", "halftime", "q3", "q4", "overtime", "game_over",
]


class MQTTClient:
    """Publishes game state to Home Assistant over MQTT discovery.

    Entities (all under one device):
      sensor  game_status      enum, see STATUS_OPTIONS
      sensor  banner           exact overlay text, e.g. "END OF 4TH QUARTER"
      sensor  player1_score / player2_score
      event   score            fires per scoring play (player1_scored / player2_scored)
    plus an availability topic with a last-will, so HA shows the entities as
    unavailable if the detector dies. Sensor values are retained and re-sent on reconnect.
    """

    def __init__(self, config: MQTTConfig):
        self.config = config
        self.client: Optional[mqtt.Client] = None
        self.connected = False
        self.config_published = False
        self._state: Dict[str, str] = {}      # topic suffix -> last payload, for resync on reconnect

    # ---- topics ----
    @property
    def prefix(self) -> str:
        return self.config.topic_prefix

    def _sensor_topic(self, name: str, leaf: str) -> str:
        return f"homeassistant/sensor/{self.prefix}_{name}/{leaf}"

    @property
    def availability_topic(self) -> str:
        return f"{self.prefix}/availability"

    @property
    def event_topic(self) -> str:
        return f"{self.prefix}/score_event"

    # ---- connection ----
    def connect(self) -> bool:
        """Connect to the broker. Returns True once connected."""
        if not self.config.enabled:
            print("MQTT is disabled in configuration")
            return False
        try:
            version = getattr(mqtt, 'CallbackAPIVersion', None)
            if version is not None:                       # paho-mqtt 2.x
                self.client = mqtt.Client(version.VERSION2, client_id=f"{self.prefix}_detector")
            else:                                         # paho-mqtt 1.x
                self.client = mqtt.Client(client_id=f"{self.prefix}_detector")
            if self.config.username and self.config.password:
                self.client.username_pw_set(self.config.username, self.config.password)
            self.client.will_set(self.availability_topic, "offline", retain=True)
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.reconnect_delay_set(min_delay=1, max_delay=30)
            self.client.connect(self.config.broker, self.config.port, keepalive=60)
            self.client.loop_start()

            start = time.time()
            while not self.connected and time.time() - start < 5:
                time.sleep(0.1)
            if self.connected:
                print(f"Connected to MQTT broker: {self.config.broker}:{self.config.port}")
                return True
            print("Failed to connect to MQTT broker (timeout)")
            return False
        except Exception as e:
            print(f"Error connecting to MQTT broker: {e}")
            return False

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        # paho 2.x passes a ReasonCode (falsy-compatible via .is_failure); 1.x passes an int
        failed = rc.is_failure if hasattr(rc, 'is_failure') else rc != 0
        if failed:
            print(f"MQTT connection failed: {rc}")
            return
        self.connected = True
        print("MQTT connection established")
        self._resync()

    def _on_disconnect(self, client, userdata, *args):
        self.connected = False
        print("MQTT disconnected")

    def _resync(self):
        """Re-announce discovery, availability and the last known values (also on reconnect)."""
        self.publish_config()
        self._raw(self.availability_topic, "online", retain=True)
        for topic, payload in self._state.items():
            self._raw(topic, payload, retain=True)

    # ---- discovery ----
    def publish_config(self):
        """Publish Home Assistant auto-discovery configuration."""
        if not self.config.enabled or not self.connected:
            return
        device = {
            "identifiers": [f"{self.prefix}_detector"],
            "name": "NBA Jam Detector",
            "model": "Dell OptiPlex 7060",
            "manufacturer": "Custom",
        }
        common = {
            "device": device,
            "availability_topic": self.availability_topic,
            "value_template": "{{ value_json.state }}",
        }

        def sensor(name, label, extra=None):
            cfg = dict(common, name=label, unique_id=f"{self.prefix}_{name}",
                       state_topic=self._sensor_topic(name, "state"))
            cfg.update(extra or {})
            self._raw(self._sensor_topic(name, "config"), json.dumps(cfg), retain=True)

        sensor("game_status", "NBA Jam Game Status",
               {"device_class": "enum", "options": STATUS_OPTIONS, "icon": "mdi:basketball"})
        sensor("banner", "NBA Jam Banner", {"icon": "mdi:text-box-outline"})
        sensor("player1_score", "NBA Jam Player 1 Score",
               {"unit_of_measurement": "points", "state_class": "measurement"})
        sensor("player2_score", "NBA Jam Player 2 Score",
               {"unit_of_measurement": "points", "state_class": "measurement"})

        event_cfg = {
            "name": "NBA Jam Score",
            "unique_id": f"{self.prefix}_score_event",
            "state_topic": self.event_topic,
            "event_types": ["player1_scored", "player2_scored"],
            "device": device,
            "availability_topic": self.availability_topic,
        }
        self._raw(f"homeassistant/event/{self.prefix}_score/config", json.dumps(event_cfg), retain=True)

        self.config_published = True
        print("MQTT auto-discovery config published")

    # ---- state ----
    def publish_game_status(self, state: str):
        self._sensor_state("game_status", state)

    def publish_banner(self, banner: Optional[str]):
        if banner is not None:      # keep the last banner until a new one replaces it
            self._sensor_state("banner", banner)

    def publish_score(self, player: str, score: Optional[int]):
        """player is 'player1' or 'player2'. None is never published."""
        if score is None or player not in ("player1", "player2"):
            return
        self._sensor_state(f"{player}_score", score)

    def publish_score_event(self, scorer: int, points: int, player1: int, player2: int, quarter: str):
        """Fire the per-play event. Not retained: an old play must not replay on reconnect."""
        payload = json.dumps({
            "event_type": f"player{scorer}_scored",
            "player": scorer,
            "points": points,
            "player1_score": player1,
            "player2_score": player2,
            "quarter": quarter,
        })
        self._raw(self.event_topic, payload, retain=False)

    def _sensor_state(self, name: str, value):
        topic = self._sensor_topic(name, "state")
        payload = json.dumps({"state": value})
        self._state[topic] = payload      # remembered even while disconnected, sent on reconnect
        self._raw(topic, payload, retain=True)

    def _raw(self, topic: str, payload: str, retain: bool):
        if not (self.config.enabled and self.client and self.connected):
            return
        result = self.client.publish(topic, payload, retain=retain)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            print(f"Warning: Failed to publish to {topic}")

    # ---- teardown ----
    def disconnect(self):
        if self.client:
            if self.connected:
                self._raw(self.availability_topic, "offline", retain=True)
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            print("MQTT disconnected")

    def cleanup_configs(self):
        """Publish empty configs to remove the entities from Home Assistant."""
        if not self.config.enabled or not self.connected:
            return
        for name in ("game_status", "banner", "player1_score", "player2_score"):
            self._raw(self._sensor_topic(name, "config"), "", retain=True)
        self._raw(f"homeassistant/event/{self.prefix}_score/config", "", retain=True)
        self._raw(self.availability_topic, "", retain=True)
        print("MQTT cleanup configs published")
