# CortexRuntime API Reference

## Authentication
CortexRuntime supports Bearer Token (JWT RS256/HS256) and API Key authentication headers.

```http
Authorization: Bearer <JWT_TOKEN>
X-Cortex-API-Key: <SECRET_KEY>
```

## REST Endpoints

### 1. Health Probe
- **Method**: `GET /v1/healthz`
- **Response**: `{"status": "ok", "timestamp": 1730648400.0}`

### 2. Chat Completions
- **Method**: `POST /v1/chat/completions`
- **Payload**:
  ```json
  {
    "message": "User query string",
    "session_id": "guild_1234_chan_5678",
    "max_tokens": 128,
    "temperature": 0.72
  }
  ```
- **Response**:
  ```json
  {
    "session_id": "guild_1234_chan_5678",
    "reply": "Synthesized agent response",
    "latency_ms": 28.4,
    "tokens_generated": 14
  }
  ```

## WebSocket Protocol

### Duplex Streaming Connection
- **Endpoint**: `WS /v1/stream`
- **Frames**:
  - Client -> Server: `{"type": "user_input", "message": "query"}`
  - Server -> Client: `{"type": "token", "content": "Hello"}`
  - Server -> Client: `{"type": "pcm_audio", "format": "raw_s16le_24khz", "data": "<base64_pcm>"}`
  - Server -> Client: `{"type": "done"}`
