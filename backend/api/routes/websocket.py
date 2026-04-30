"""
WebSocket route for real-time updates.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from typing import List, Dict
from datetime import datetime
import asyncio
import logging
import json

from backend.api.proxy_auth import ProxyAuthError, validate_proxy_headers
from backend.shared.config import system_config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, event_type: str, data: Dict):
        """Broadcast message to all connected clients."""
        message = json.dumps({
            "type": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Failed to send to websocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    if system_config.generic_mode:
        try:
            validate_proxy_headers(
                websocket.headers,
                method="GET",
                path=websocket.url.path,
                expected_instance_id=system_config.instance_id,
                shared_secret=system_config.internal_proxy_secret or "",
            )
        except ProxyAuthError as exc:
            logger.warning("Rejected generic-mode websocket connection: %s", exc.detail)
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION,
                reason=exc.detail,
            )
            return

    await manager.connect(websocket)
    
    try:
        # Keep connection alive
        while True:
            # Receive messages (ping/pong)
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# Broadcaster function to be used by coordinator
async def broadcast_event(event_type: str, data: Dict):
    """Broadcast an event to all WebSocket clients."""
    await manager.broadcast(event_type, data)

