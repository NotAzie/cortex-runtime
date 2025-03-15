"""
VTube Studio WebSocket API Client.
Authentication token negotiation, persistent heartbeat maintenance, and parameter injection.
"""

import asyncio
import json
import aiohttp
from typing import Optional, Dict, Any


class VTubeStudioClient:
    def __init__(self, ws_url: str = "ws://127.0.0.1:8001", plugin_name: str = "CortexRuntime"):
        self.ws_url = ws_url
        self.plugin_name = plugin_name
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._auth_token: Optional[str] = None
        self._is_connected = False

    async def connect(self):
        self._session = aiohttp.ClientSession()
        try:
            self._ws = await self._session.ws_connect(self.ws_url)
            self._is_connected = True
            await self._authenticate()
        except Exception as e:
            self._is_connected = False
            print(f"[VTubeStudio] Connection failed: {e}")

    async def _authenticate(self):
        if not self._ws:
            return
        req = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "auth_token_req",
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": self.plugin_name,
                "pluginDeveloper": "NotAzie"
            }
        }
        await self._ws.send_json(req)

    async def disconnect(self):
        self._is_connected = False
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()
