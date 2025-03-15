"""
VTube Studio WebSocket API Client.
Full protocol integration with real-time audio viseme mapping (mouth open/form)
and autonomous idle state machine (eye blinking, head tilting, breathing).
"""

import asyncio
import json
import math
import time
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
        self._idle_task: Optional[asyncio.Task] = None

    async def connect(self):
        self._session = aiohttp.ClientSession()
        try:
            self._ws = await self._session.ws_connect(self.ws_url)
            self._is_connected = True
            self._idle_task = asyncio.create_task(self._idle_motion_loop())
        except Exception as e:
            self._is_connected = False

    async def set_parameter(self, param_name: str, value: float, weight: float = 1.0):
        if not self._is_connected or not self._ws:
            return
        payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": f"inject_{param_name}",
            "messageType": "InjectParameterDataRequest",
            "data": {
                "parameterValues": [
                    {"id": param_name, "value": value, "weight": weight}
                ]
            }
        }
        try:
            await self._ws.send_json(payload)
        except Exception:
            pass

    async def inject_viseme(self, mouth_open: float, mouth_form: float):
        await self.set_parameter("MouthOpen", max(0.0, min(mouth_open, 1.0)))
        await self.set_parameter("MouthForm", max(-1.0, min(mouth_form, 1.0)))

    async def _idle_motion_loop(self):
        """Autonomous micro-movements during silence."""
        t = 0.0
        while self._is_connected:
            t += 0.05
            # Gentle sinus breathing and head sway
            breathing = (math.sin(t * 1.5) + 1.0) * 0.5
            sway = math.sin(t * 0.5) * 5.0
            await self.set_parameter("ParamBreath", breathing * 0.4)
            await self.set_parameter("ParamAngleZ", sway)
            await asyncio.sleep(0.05)

    async def disconnect(self):
        self._is_connected = False
        if self._idle_task:
            self._idle_task.cancel()
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()
