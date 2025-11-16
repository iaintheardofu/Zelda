"""
WebSocket Routes for Real-time ZELDA Data
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from .websocket_manager import manager
from loguru import logger
import json

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for real-time ZELDA data

    Clients can subscribe to channels:
    - spectrum: RF spectrum waterfall data
    - detections: Signal detections
    - threats: Threat alerts
    - missions: Mission status updates
    - receivers: Receiver status changes
    - all: Subscribe to everything
    """

    await manager.connect(websocket)

    try:
        # Send welcome message
        await manager.send_personal({
            "type": "connected",
            "message": "Connected to ZELDA WebSocket",
            "channels": ["spectrum", "detections", "threats", "missions", "receivers"],
            "instructions": {
                "subscribe": "Send: {\"action\": \"subscribe\", \"channel\": \"spectrum\"}",
                "unsubscribe": "Send: {\"action\": \"unsubscribe\", \"channel\": \"spectrum\"}"
            }
        }, websocket)

        # Main message loop
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)

                action = message.get("action")
                channel = message.get("channel")

                if action == "subscribe" and channel:
                    manager.subscribe(websocket, channel)
                    await manager.send_personal({
                        "type": "subscribed",
                        "channel": channel
                    }, websocket)

                elif action == "unsubscribe" and channel:
                    manager.unsubscribe(websocket, channel)
                    await manager.send_personal({
                        "type": "unsubscribed",
                        "channel": channel
                    }, websocket)

                elif action == "ping":
                    await manager.send_personal({
                        "type": "pong",
                        "timestamp": message.get("timestamp")
                    }, websocket)

                else:
                    await manager.send_personal({
                        "type": "error",
                        "message": f"Unknown action: {action}"
                    }, websocket)

            except json.JSONDecodeError:
                await manager.send_personal({
                    "type": "error",
                    "message": "Invalid JSON"
                }, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@router.websocket("/ws/spectrum")
async def websocket_spectrum(websocket: WebSocket):
    """Dedicated WebSocket for spectrum data only"""

    await manager.connect(websocket)
    manager.subscribe(websocket, "spectrum")

    try:
        await manager.send_personal({
            "type": "connected",
            "channel": "spectrum",
            "message": "Connected to spectrum data stream"
        }, websocket)

        # Keep connection alive
        while True:
            try:
                await websocket.receive_text()
            except Exception:
                pass

    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.websocket("/ws/detections")
async def websocket_detections(websocket: WebSocket):
    """Dedicated WebSocket for signal detections"""

    await manager.connect(websocket)
    manager.subscribe(websocket, "detections")

    try:
        await manager.send_personal({
            "type": "connected",
            "channel": "detections",
            "message": "Connected to detections stream"
        }, websocket)

        while True:
            try:
                await websocket.receive_text()
            except Exception:
                pass

    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.websocket("/ws/threats")
async def websocket_threats(websocket: WebSocket):
    """Dedicated WebSocket for threat alerts"""

    await manager.connect(websocket)
    manager.subscribe(websocket, "threats")

    try:
        await manager.send_personal({
            "type": "connected",
            "channel": "threats",
            "message": "Connected to threat alerts stream"
        }, websocket)

        while True:
            try:
                await websocket.receive_text()
            except Exception:
                pass

    except WebSocketDisconnect:
        manager.disconnect(websocket)
