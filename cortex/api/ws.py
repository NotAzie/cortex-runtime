"""
Bidirectional low-latency WebSocket gateway for real-time duplex dialogue and audio chunk transmission.
"""

from fastapi import WebSocket, WebSocketDisconnect
import json
import asyncio


async def websocket_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw_text = await websocket.receive_text()
            data = json.loads(raw_text)
            user_msg = data.get("message", "")

            # Acknowledge receipt
            await websocket.send_json({"type": "ack", "status": "processing"})

            # Stream response simulation
            tokens = ["Cortex", " streaming", " protocol", " active", "."]
            for tok in tokens:
                await asyncio.sleep(0.02)
                await websocket.send_json({"type": "token", "content": tok})

            await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        pass
