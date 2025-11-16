"""
WebSocket Manager for Real-time ZELDA Data Streaming
Handles multiple WebSocket connections and broadcasts RF data
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Set, Dict, List, Any
from datetime import datetime
import asyncio
import json
import numpy as np
from loguru import logger


class ConnectionManager:
    """Manages WebSocket connections and broadcasts"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.subscription_map: Dict[WebSocket, Set[str]] = {}

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.subscription_map[websocket] = set()
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections.discard(websocket)
        self.subscription_map.pop(websocket, None)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    def subscribe(self, websocket: WebSocket, channel: str):
        """Subscribe websocket to a specific channel"""
        if websocket in self.subscription_map:
            self.subscription_map[websocket].add(channel)
            logger.debug(f"WebSocket subscribed to {channel}")

    def unsubscribe(self, websocket: WebSocket, channel: str):
        """Unsubscribe websocket from a channel"""
        if websocket in self.subscription_map:
            self.subscription_map[websocket].discard(channel)
            logger.debug(f"WebSocket unsubscribed from {channel}")

    async def send_personal(self, message: dict, websocket: WebSocket):
        """Send message to specific websocket"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending to websocket: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict, channel: str = None):
        """Broadcast message to all or specific channel subscribers"""
        disconnected = set()

        for connection in self.active_connections:
            # If channel specified, only send to subscribers
            if channel and channel not in self.subscription_map.get(connection, set()):
                continue

            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting: {e}")
                disconnected.add(connection)

        # Clean up disconnected
        for connection in disconnected:
            self.disconnect(connection)

    async def broadcast_spectrum_data(self, spectrum_data: dict):
        """Broadcast spectrum analyzer data"""
        message = {
            "type": "spectrum",
            "data": spectrum_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(message, channel="spectrum")

    async def broadcast_detection(self, detection: dict):
        """Broadcast signal detection"""
        message = {
            "type": "detection",
            "data": detection,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(message, channel="detections")

    async def broadcast_threat(self, threat: dict):
        """Broadcast threat alert"""
        message = {
            "type": "threat_alert",
            "data": threat,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(message, channel="threats")

    async def broadcast_mission_update(self, mission_data: dict):
        """Broadcast mission status update"""
        message = {
            "type": "mission_update",
            "data": mission_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(message, channel="missions")

    async def broadcast_receiver_status(self, receiver_data: dict):
        """Broadcast receiver status change"""
        message = {
            "type": "receiver_status",
            "data": receiver_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(message, channel="receivers")


# Global manager instance
manager = ConnectionManager()


class DataSimulator:
    """Simulates real-time RF data for demonstration"""

    @staticmethod
    def generate_spectrum_data(
        center_freq: float = 915e6,
        bandwidth: float = 40e6,
        num_bins: int = 512
    ) -> dict:
        """Generate simulated spectrum data"""

        frequencies = np.linspace(
            center_freq - bandwidth/2,
            center_freq + bandwidth/2,
            num_bins
        )

        # Simulate power spectrum with some peaks
        powers = -80 + np.random.randn(num_bins) * 10

        # Add signal peaks
        for _ in range(np.random.randint(2, 5)):
            peak_idx = np.random.randint(0, num_bins)
            peak_width = np.random.randint(10, 30)
            peak_power = np.random.uniform(-30, -10)

            for i in range(max(0, peak_idx - peak_width), min(num_bins, peak_idx + peak_width)):
                dist = abs(i - peak_idx)
                powers[i] = peak_power - (dist ** 2) / 10

        return {
            "frequencies": frequencies.tolist(),
            "powers": powers.tolist(),
            "center_freq": center_freq,
            "bandwidth": bandwidth,
            "sample_rate": bandwidth,
            "timestamp": datetime.now().isoformat()
        }

    @staticmethod
    def generate_detection() -> dict:
        """Generate simulated signal detection"""

        signal_types = ["WiFi", "Bluetooth", "LoRa", "Zigbee", "Cellular", "Unknown"]
        modulations = ["QPSK", "QAM16", "QAM64", "OFDM", "FSK", "GFSK"]

        freq_base = 915e6
        freq_offset = np.random.uniform(-20e6, 20e6)

        return {
            "id": f"det_{datetime.now().timestamp()}",
            "frequency": freq_base + freq_offset,
            "signal_type": np.random.choice(signal_types),
            "confidence": np.random.uniform(0.7, 0.99),
            "power": np.random.uniform(-60, -20),
            "bandwidth": np.random.uniform(1e6, 10e6),
            "modulation": np.random.choice(modulations),
            "timestamp": datetime.now().isoformat()
        }

    @staticmethod
    def generate_threat() -> dict:
        """Generate simulated threat alert"""

        threat_types = [
            "gps_spoofing",
            "jamming_barrage",
            "jamming_spot",
            "imsi_catcher",
            "rogue_ap"
        ]

        severities = ["low", "medium", "high", "critical"]

        return {
            "id": f"threat_{datetime.now().timestamp()}",
            "type": np.random.choice(threat_types),
            "severity": np.random.choice(severities),
            "confidence": np.random.uniform(0.7, 0.99),
            "description": "Simulated threat detection for demonstration",
            "location": {
                "latitude": 37.7749 + np.random.uniform(-0.01, 0.01),
                "longitude": -122.4194 + np.random.uniform(-0.01, 0.01)
            },
            "recommended_action": "Investigate signal source",
            "timestamp": datetime.now().isoformat()
        }

    @staticmethod
    def generate_mission_update(mission_id: str = "mission_001") -> dict:
        """Generate simulated mission update"""

        statuses = ["pending", "active", "paused", "completed"]

        return {
            "mission_id": mission_id,
            "status": np.random.choice(statuses),
            "detections_count": np.random.randint(0, 100),
            "threats_count": np.random.randint(0, 10),
            "uptime_seconds": np.random.randint(0, 10000),
            "timestamp": datetime.now().isoformat()
        }

    @staticmethod
    def generate_receiver_status(receiver_id: str = "rx_001") -> dict:
        """Generate simulated receiver status"""

        statuses = ["online", "offline", "error"]

        return {
            "receiver_id": receiver_id,
            "status": np.random.choice(statuses, p=[0.8, 0.1, 0.1]),
            "frequency": 915e6,
            "sample_rate": 40e6,
            "gain": 20.0,
            "temperature": np.random.uniform(25, 45),
            "signal_strength": np.random.uniform(-80, -40),
            "timestamp": datetime.now().isoformat()
        }


async def start_data_streaming(app):
    """Background task to generate and broadcast simulated data"""

    logger.info("Starting real-time data streaming")

    simulator = DataSimulator()
    spectrum_interval = 0.1  # 10 Hz for spectrum
    detection_interval = 2.0  # Detection every 2s
    threat_interval = 10.0  # Threat every 10s
    mission_interval = 5.0  # Mission update every 5s

    last_detection = datetime.now()
    last_threat = datetime.now()
    last_mission = datetime.now()

    while True:
        try:
            now = datetime.now()

            # Spectrum data (high frequency)
            if len(manager.active_connections) > 0:
                spectrum = simulator.generate_spectrum_data()
                await manager.broadcast_spectrum_data(spectrum)

            # Signal detection (periodic)
            if (now - last_detection).total_seconds() >= detection_interval:
                detection = simulator.generate_detection()
                await manager.broadcast_detection(detection)
                last_detection = now

            # Threat alert (periodic)
            if (now - last_threat).total_seconds() >= threat_interval:
                threat = simulator.generate_threat()
                await manager.broadcast_threat(threat)
                last_threat = now

            # Mission update (periodic)
            if (now - last_mission).total_seconds() >= mission_interval:
                mission = simulator.generate_mission_update()
                await manager.broadcast_mission_update(mission)
                last_mission = now

            await asyncio.sleep(spectrum_interval)

        except Exception as e:
            logger.error(f"Error in data streaming: {e}")
            await asyncio.sleep(1.0)
