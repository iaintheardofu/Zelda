"""
FastAPI Application for Zelda TDOA System
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
import json
from datetime import datetime
from loguru import logger

# Import Zelda components
# (In a full implementation, these would be imported from the actual modules)


# Pydantic models for API
class ReceiverConfig(BaseModel):
    receiver_id: str
    driver: str
    center_freq: float
    sample_rate: float
    latitude: float
    longitude: float
    altitude: float
    gain: float = 20.0


class SystemConfig(BaseModel):
    receivers: List[ReceiverConfig]
    tdoa_method: str = "gcc-phat"
    multilateration_method: str = "taylor"


class GeolocationData(BaseModel):
    latitude: float
    longitude: float
    altitude: float
    accuracy: float
    timestamp: str
    confidence: float


class SystemStatus(BaseModel):
    status: str  # "idle", "running", "error"
    num_receivers: int
    uptime_seconds: float
    total_fixes: int
    last_fix_time: Optional[str]


# Create FastAPI app
def create_app() -> FastAPI:
    """Create and configure FastAPI application"""

    app = FastAPI(
        title="Zelda TDOA System API",
        description="Advanced Time Difference of Arrival Electronic Warfare Platform",
        version="0.1.0-alpha",
    )

    # CORS middleware for web frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify actual origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global state (in production, use proper state management)
    app.state.system_running = False
    app.state.receivers = {}
    app.state.geolocation_history = []
    app.state.websocket_clients = set()
    app.state.start_time = None

    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "name": "Zelda TDOA System",
            "version": "0.1.0-alpha",
            "status": "operational",
            "endpoints": {
                "status": "/api/status",
                "receivers": "/api/receivers",
                "start": "POST /api/start",
                "stop": "POST /api/stop",
                "latest": "/api/positions/latest",
                "history": "/api/positions/history",
                "websocket": "ws://host:port/ws/positions",
            }
        }

    @app.get("/api/status", response_model=SystemStatus)
    async def get_status():
        """Get system status"""

        uptime = 0.0
        if app.state.start_time:
            uptime = (datetime.now() - app.state.start_time).total_seconds()

        last_fix = None
        if app.state.geolocation_history:
            last_fix = app.state.geolocation_history[-1]["timestamp"]

        return SystemStatus(
            status="running" if app.state.system_running else "idle",
            num_receivers=len(app.state.receivers),
            uptime_seconds=uptime,
            total_fixes=len(app.state.geolocation_history),
            last_fix_time=last_fix,
        )

    @app.get("/api/receivers")
    async def get_receivers():
        """Get list of receivers"""
        return {
            "receivers": list(app.state.receivers.values()),
            "count": len(app.state.receivers),
        }

    @app.post("/api/receivers")
    async def add_receiver(config: ReceiverConfig):
        """Add a new receiver"""

        if config.receiver_id in app.state.receivers:
            raise HTTPException(status_code=400, detail="Receiver already exists")

        app.state.receivers[config.receiver_id] = config.dict()

        logger.info(f"Added receiver: {config.receiver_id}")

        return {"status": "success", "receiver_id": config.receiver_id}

    @app.delete("/api/receivers/{receiver_id}")
    async def remove_receiver(receiver_id: str):
        """Remove a receiver"""

        if receiver_id not in app.state.receivers:
            raise HTTPException(status_code=404, detail="Receiver not found")

        del app.state.receivers[receiver_id]

        logger.info(f"Removed receiver: {receiver_id}")

        return {"status": "success"}

    @app.post("/api/start")
    async def start_system(config: Optional[SystemConfig] = None):
        """Start the TDOA system"""

        if app.state.system_running:
            raise HTTPException(status_code=400, detail="System already running")

        # If config provided, update receivers
        if config:
            app.state.receivers = {
                r.receiver_id: r.dict() for r in config.receivers
            }

        if len(app.state.receivers) < 3:
            raise HTTPException(
                status_code=400,
                detail="Need at least 3 receivers"
            )

        app.state.system_running = True
        app.state.start_time = datetime.now()

        logger.info(f"System started with {len(app.state.receivers)} receivers")

        # Start background processing task
        asyncio.create_task(process_tdoa_background(app))

        return {
            "status": "started",
            "num_receivers": len(app.state.receivers),
        }

    @app.post("/api/stop")
    async def stop_system():
        """Stop the TDOA system"""

        if not app.state.system_running:
            raise HTTPException(status_code=400, detail="System not running")

        app.state.system_running = False

        logger.info("System stopped")

        return {"status": "stopped"}

    @app.get("/api/positions/latest")
    async def get_latest_position():
        """Get the most recent geolocation"""

        if not app.state.geolocation_history:
            raise HTTPException(status_code=404, detail="No positions available")

        return app.state.geolocation_history[-1]

    @app.get("/api/positions/history")
    async def get_position_history(limit: int = 100):
        """Get historical geolocations"""

        history = app.state.geolocation_history[-limit:]

        return {
            "positions": history,
            "count": len(history),
        }

    @app.websocket("/ws/positions")
    async def websocket_positions(websocket: WebSocket):
        """WebSocket endpoint for real-time position updates"""

        await websocket.accept()
        app.state.websocket_clients.add(websocket)

        logger.info(f"WebSocket client connected ({len(app.state.websocket_clients)} total)")

        try:
            # Keep connection alive and send updates
            while True:
                # Wait for message or timeout
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass

                # Send ping to keep alive
                await websocket.send_json({"type": "ping"})

        except WebSocketDisconnect:
            app.state.websocket_clients.remove(websocket)
            logger.info(f"WebSocket client disconnected ({len(app.state.websocket_clients)} total)")

    @app.get("/api/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}

    return app


async def process_tdoa_background(app: FastAPI):
    """
    Background task for TDOA processing.

    In a real implementation, this would:
    1. Read samples from receivers
    2. Calculate TDOA
    3. Perform multilateration
    4. Track targets
    5. Send results to WebSocket clients
    """

    logger.info("Background TDOA processing started")

    import random

    while app.state.system_running:
        # Simulate TDOA processing (replace with real processing)
        await asyncio.sleep(1.0)

        # Generate simulated result
        result = {
            "latitude": 37.7749 + random.uniform(-0.01, 0.01),
            "longitude": -122.4194 + random.uniform(-0.01, 0.01),
            "altitude": random.uniform(0, 100),
            "accuracy": random.uniform(5, 20),
            "confidence": random.uniform(0.7, 0.99),
            "timestamp": datetime.now().isoformat(),
            "algorithm": "gcc-phat+taylor",
        }

        # Store in history
        app.state.geolocation_history.append(result)

        # Limit history size
        if len(app.state.geolocation_history) > 1000:
            app.state.geolocation_history = app.state.geolocation_history[-1000:]

        # Broadcast to WebSocket clients
        disconnected = set()
        for client in app.state.websocket_clients:
            try:
                await client.send_json({
                    "type": "position_update",
                    "data": result
                })
            except Exception as e:
                logger.error(f"Error sending to WebSocket client: {e}")
                disconnected.add(client)

        # Remove disconnected clients
        app.state.websocket_clients -= disconnected

    logger.info("Background TDOA processing stopped")


# Run with: uvicorn backend.api.app:app --reload
if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
