# NeuroLite — Full System Redesign
### Offline-First, Low-Resource, Real-Time Conversational AI Agent

---

## 0. Engineering Mindset

> "Every millisecond spent outside of inference is waste. Every abstraction layer that doesn't earn its keep is debt."

The guiding rule for this redesign: **flatten where it doesn't cost intelligence, cache where it doesn't cost memory, stream where it doesn't cost correctness.**

---

## 1. Final Optimized Architecture

```
Neuro/
├── llm_engine.py          # Single inference path. llama.cpp / Ollama backend.
├── context_builder.py     # Deterministic prompt assembly. <1ms target.
├── memory.py              # 3-tier hybrid memory. No vector DB.
├── signals.py             # Thin event bus. Replaces polling everywhere.
├── constants.py           # All config constants. Loaded once.
├── Neuro.yaml             # Personality + system config.
│
├── modules/
│   ├── discord_client.py  # Only external platform. Event-driven.
│   ├── audio_player.py    # Queue + interrupt-safe playback.
│   └── vtube_studio.py    # Optional. Lazy-loaded. Never in hot path.
│
├── stt.py                 # Non-blocking. Runs in isolated thread.
├── tts.py                 # Streaming-aware. Feeds audio_player queue.
├── streamingSink.py       # Token → TTS bridge. Chunked sentence flushing.
│
└── main.py                # Boot sequence. Lazy loads non-critical modules.
```

**Total core modules in hot path: 5**
`llm_engine → context_builder → memory → tts → audio_player`

---

## 2. What Changed and Why

### Removed

| File | Reason |
|---|---|
| `llmWrappers/abstractLLMWrapper.py` | Multi-provider abstraction = overhead with zero offline benefit. One brain, one path. |
| `llmWrappers/imageLLMWrapper.py` | Multimodal cloud dependency. Incompatible with offline-first constraint. |
| `modules/twitchClient.py` | Spec is Discord-only. Dead weight. |
| `modules/multimodal.py` | Same as image wrapper. Removed entirely. |
| `socketioServer.py` | External socket server adds latency surface and complexity with no stated requirement. |
| `modules/injection.py` | Logic absorbed into `context_builder.py`. Was a thin pass-through. |

### Merged

| Before | After | Reason |
|---|---|---|
| `textLLMWrapper.py` + `llmState.py` + `abstractLLMWrapper.py` | `llm_engine.py` | Three files managing one thing. Collapsed to single stateful inference engine. |
| `prompter.py` + `injection.py` + `modules/customPrompt.py` | `context_builder.py` | Prompt assembly is one deterministic pipeline. Splitting it across files creates hidden coupling. |

### Rewritten

| File | Reason |
|---|---|
| `memory.py` | Full 3-tier redesign. Old flat memory breaks personality continuity at scale. |
| `stt.py` | Must be non-blocking. Blocking STT stalls the inference loop — unacceptable. |
| `tts.py` | Must consume streaming tokens. Waiting for full response before speaking = dead air. |
| `audio_player.py` | Needs interrupt logic. Without it, queued audio can't be cut off when user speaks. |
| `main.py` | Boot order restructured. Non-critical modules (VTube) lazy-loaded after core is warm. |

---

## 3. Inference Pipeline — Internal Flow

```
[User Input Event]
        │
        ▼
[signals.py: INPUT_RECEIVED]
        │
        ▼
[context_builder.py]
  ├─ Pull personality core (cached, never reloaded)
  ├─ Pull short-term window from memory (last N turns, in RAM)
  ├─ Pull episodic fragments (importance-scored, top-K by relevance)
  └─ Assemble final prompt (deterministic, <1ms)
        │
        ▼
[llm_engine.py]
  ├─ Submit prompt to llama.cpp or Ollama HTTP local endpoint
  ├─ Stream tokens as they arrive (do NOT wait for completion)
  └─ Emit token chunks via signals.py: TOKEN_CHUNK
        │
        ├──────────────────────────┐
        ▼                          ▼
[streamingSink.py]         [memory.py: async write]
  ├─ Buffer tokens into        ├─ Append to short-term window
  │  sentence chunks           └─ Score + episodic store if salient
  └─ Emit SENTENCE_READY
        │
        ▼
[tts.py]
  ├─ Convert sentence chunk to audio
  └─ Push audio buffer to audio_player queue
        │
        ▼
[audio_player.py]
  ├─ Plays queued audio buffers
  ├─ Monitors interrupt signal (new user input → flush queue)
  └─ Emits PLAYBACK_DONE when finished
```

**Key design decisions:**
- Memory write happens **async** — it never sits in the inference hot path
- Sentence chunking in `streamingSink.py` means TTS starts speaking at the **first sentence**, not after full generation
- Interrupt signal from STT immediately flushes the audio queue — the agent "stops talking" the moment the user speaks

---

## 4. Memory System Design

Three tiers. No vector DB. No embeddings. Runs entirely on structured Python data + lightweight heuristics.

### Tier 1 — Short-Term Window (Hot, In-RAM)
```python
# Implemented as a fixed-size deque
short_term: deque[dict] = deque(maxlen=SHORT_TERM_LIMIT)  # default: 20 turns

# Each entry:
{
  "role": "user" | "assistant",
  "content": str,
  "timestamp": float
}
```
- Always included in context, in full
- Evicted automatically when maxlen exceeded (sliding window)
- Zero retrieval cost — it's just a slice of a deque

### Tier 2 — Episodic Memory (Warm, Importance-Scored)
```python
# Stored as a flat JSON list, loaded into RAM at boot
episodes: list[dict]

# Each entry:
{
  "summary": str,          # compressed representation
  "importance": float,     # 0.0 – 1.0, scored at write time
  "tags": list[str],       # lightweight semantic labels (no embeddings)
  "timestamp": float,
  "access_count": int      # boosted on retrieval — recency + frequency
}
```

**Importance scoring at write time (no ML required):**
```python
def score_importance(turn: str) -> float:
    score = 0.0
    if contains_name_or_preference(turn):   score += 0.4
    if contains_question_or_request(turn):  score += 0.2
    if len(turn) > LONG_TURN_THRESHOLD:     score += 0.2
    if emotional_keywords_present(turn):    score += 0.2
    return min(score, 1.0)
```

**Retrieval:** At context build time, select top-K episodes by `importance + recency_decay(timestamp) + access_count`. No cosine similarity. No embeddings. Just arithmetic.

### Tier 3 — Long-Term Summaries (Cold, Compressed)
```python
# Written when episodic list exceeds EPISODIC_LIMIT (default: 200 entries)
# Triggered async, never blocking
long_term: list[dict]

# Each entry:
{
  "summary": str,   # LLM-generated compression of N episodic entries
  "period": str,    # "2025-W03" — week/month granularity
  "themes": list[str]
}
```

Long-term summaries are generated by calling the **local LLM itself** with a compression prompt when the episodic list is too large — no separate summarization service needed.

**Retrieval strategy across all tiers:**
```
Context budget = MAX_CONTEXT_TOKENS - personality_tokens - short_term_tokens
Remaining budget filled greedily: episodic top-K first, then long-term if space allows.
```

---

## 5. Prompt Construction System

Handled entirely by `context_builder.py`. One function. Deterministic. No branching logic at runtime.

### Structure (in order, top to bottom in prompt)

```
[SYSTEM BLOCK]        ← personality core + behavioral rules (loaded once, cached)
[LONG-TERM BLOCK]     ← 1-3 thematic summaries if context budget allows
[EPISODIC BLOCK]      ← top-K episodic fragments, formatted as "[Memory]: ..."
[CONVERSATION BLOCK]  ← short-term window, last N turns verbatim
[CURRENT INPUT]       ← user's current message
```

### Personality Core (Cached at Boot)
```python
# Loaded from Neuro.yaml once at startup, never reloaded
PERSONALITY_CORE: str = load_personality()  # ~200-400 tokens, stays constant
```

### Context Budget Enforcement
```python
def build_context(user_input: str) -> str:
    budget = MAX_TOKENS - count_tokens(PERSONALITY_CORE) - RESPONSE_RESERVE

    short_term_text = format_short_term(memory.short_term)
    budget -= count_tokens(short_term_text)

    episodes = memory.retrieve_top_k(budget // 2)   # give half remaining to episodic
    episode_text = format_episodes(episodes)
    budget -= count_tokens(episode_text)

    long_term_text = memory.retrieve_long_term(budget) if budget > MIN_LT_THRESHOLD else ""

    return assemble(PERSONALITY_CORE, long_term_text, episode_text, short_term_text, user_input)
```

**Token counting** uses a cheap character-ratio estimate (`len(text) // 4`) rather than a full tokenizer call — accurate enough for budget enforcement, zero overhead.

### Prompt Compression Rules
1. **Episodic fragments are pre-compressed at write time** — never stored verbatim at full length
2. **Long-term summaries are aggressively short** — max 100 tokens per summary
3. **Short-term window is the only verbatim content** — everything else is compressed
4. If total context still exceeds budget: drop oldest episodic entries first, then reduce long-term count

---

## 6. Discord Integration Flow

Single-file. Event-driven. Minimal overhead.

```python
# discord_client.py — simplified core logic

@client.event
async def on_message(message):
    if not should_respond(message):   # blacklist, bot filter, rate limit check
        return

    signals.emit(SIGNALS.INPUT_RECEIVED, {
        "content": message.content,
        "author": str(message.author),
        "channel_id": message.channel.id,
        "reply_to": message.id
    })
```

**`should_respond()` filter chain (in order, cheapest check first):**
1. Is it from a bot? → skip (O(1))
2. Is author in blacklist? → skip (O(1) set lookup)
3. Is it within rate limit window? → skip (token bucket check)
4. Does it mention the agent or pass relevance heuristic? → pass through

**Response routing:**
```python
# Registered as signal handler at boot
signals.on(SIGNALS.RESPONSE_READY, send_discord_reply)

async def send_discord_reply(payload):
    channel = client.get_channel(payload["channel_id"])
    await channel.send(payload["text"], reference=payload["reply_to"])
```

**No polling. No background thread scanning messages. Discord.py's async event loop handles everything natively.**

Rate limiting: token bucket — 1 response per 3 seconds per channel. Burst of 2 allowed. Implemented with a simple `(last_sent_time, token_count)` tuple per channel — no external library needed.

---

## 7. Audio Pipeline

### STT — Non-Blocking Design
```python
# stt.py runs in a dedicated daemon thread
# Communicates only via signals — never touches inference thread directly

def stt_listener_thread():
    while running:
        audio_chunk = mic_stream.read(CHUNK_SIZE)
        if vad.is_speech(audio_chunk):          # Voice Activity Detection gate
            text = whisper_model.transcribe(audio_chunk)
            if text.strip():
                signals.emit(SIGNALS.INPUT_RECEIVED, {"content": text, "source": "voice"})
                signals.emit(SIGNALS.USER_SPEAKING)  # triggers audio interrupt
```

**VAD gate is critical** — without it, Whisper runs on every chunk of silence, wasting CPU constantly. Use `webrtcvad` (lightweight, C extension) to gate Whisper activation.

**Whisper model selection for weak hardware:**
- `whisper-tiny` (~40MB, ~150ms on CPU) — minimum latency
- `whisper-base` (~150MB, ~300ms on CPU) — acceptable quality/speed tradeoff
- Never `whisper-medium` or above on CPU-only systems

### TTS — Streaming-Aware Design
```python
# tts.py consumes sentence chunks from streamingSink
# Starts converting BEFORE full response is generated

signals.on(SIGNALS.SENTENCE_READY, on_sentence_ready)

def on_sentence_ready(sentence: str):
    audio_bytes = tts_engine.synthesize(sentence)   # Piper / Coqui TTS
    audio_player.enqueue(audio_bytes)
```

**Recommended TTS engines (offline, lightweight):**
| Engine | RAM | Quality | Latency |
|---|---|---|---|
| Piper TTS | ~50MB | Good | ~80ms/sentence |
| Coqui XTTS (small model) | ~300MB | Excellent | ~200ms/sentence |
| espeak-ng | ~5MB | Robotic | ~10ms/sentence |

Piper TTS is the default recommendation — best latency/quality tradeoff for offline use.

### Audio Player — Queue + Interrupt
```python
# audio_player.py

class AudioPlayer:
    def __init__(self):
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._interrupted = threading.Event()
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()

    def enqueue(self, audio: bytes):
        self._queue.put(audio)

    def interrupt(self):
        self._interrupted.set()
        with self._queue.mutex:
            self._queue.queue.clear()   # flush pending audio immediately

    def _playback_loop(self):
        while True:
            audio = self._queue.get()
            if self._interrupted.is_set():
                self._interrupted.clear()
                continue
            sounddevice.play(audio, ...)
            sounddevice.wait()
```

**Interrupt is wired to `SIGNALS.USER_SPEAKING`** — the moment STT detects voice activity, the audio queue flushes. The agent stops mid-sentence. This is what makes it feel reactive rather than robotic.

### streamingSink — Token-to-Sentence Bridge
```python
# streamingSink.py — sits between llm_engine and tts

SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s')

class StreamingSink:
    def __init__(self):
        self._buffer = ""

    def push(self, token: str):
        self._buffer += token
        parts = SENTENCE_ENDINGS.split(self._buffer, maxsplit=1)
        if len(parts) > 1:
            complete_sentence = parts[0]
            self._buffer = parts[1]
            signals.emit(SIGNALS.SENTENCE_READY, complete_sentence)

    def flush(self):   # called when LLM stream ends
        if self._buffer.strip():
            signals.emit(SIGNALS.SENTENCE_READY, self._buffer.strip())
            self._buffer = ""
```

---

## 8. Key Bottlenecks and Eliminations

### Bottleneck 1 — Multi-Wrapper LLM Abstraction
**Problem:** Every inference call passes through 2-3 wrapper layers, each adding function call overhead and potential state confusion.
**Fix:** Single `llm_engine.py`. One function: `stream_completion(prompt) → AsyncGenerator[str]`. Direct HTTP to local Ollama or direct `llama_cpp.Llama()` call. No routing, no provider switching.

### Bottleneck 2 — Blocking STT
**Problem:** If STT runs synchronously in the main loop, every voice input stalls inference for 150-400ms.
**Fix:** STT runs in a dedicated daemon thread. Communicates via signals only. Inference loop never waits on it.

### Bottleneck 3 — Full-Response TTS
**Problem:** Waiting for complete LLM response before starting TTS synthesis = dead air equal to the entire generation time (often 3-15 seconds on weak hardware).
**Fix:** `streamingSink.py` flushes sentence boundaries immediately. TTS starts on sentence 1 while the LLM is still generating sentence 3. Perceived latency drops to ~first-sentence generation time only.

### Bottleneck 4 — Memory Retrieval at Context Build Time
**Problem:** If memory retrieval does anything expensive (embedding lookup, disk scan) during prompt assembly, it adds latency directly to every single inference cycle.
**Fix:** Episodic memory lives in RAM as a sorted list. Retrieval is a single `sorted(..., key=score_fn)[:K]` call. Importance scores are pre-computed at write time. Context build time stays under 1ms.

### Bottleneck 5 — Personality Reloading
**Problem:** If the personality prompt is re-read from disk or re-parsed on every inference call, it wastes I/O and CPU for content that never changes.
**Fix:** `PERSONALITY_CORE` is loaded once at boot, stored as a module-level constant string. Never touched again at runtime.

### Bottleneck 6 — Audio Queue Congestion
**Problem:** Without interrupt logic, queued TTS audio from a long response continues playing while the user is already speaking again — creating an unresponsive, out-of-sync experience.
**Fix:** `SIGNALS.USER_SPEAKING` immediately flushes the audio queue. `audio_player.interrupt()` clears both the queue and the currently playing buffer.

### Bottleneck 7 — Cold Memory on Long Sessions
**Problem:** After hours of conversation, if all episodic memory is scanned for every context build, retrieval time grows linearly with session length.
**Fix:** Episodic list is capped at `EPISODIC_LIMIT` (default 200). When exceeded, oldest low-importance entries are compressed into long-term summaries via a local LLM call triggered **async, never blocking inference**. The active episodic list stays bounded and fast.

---

## 9. signals.py — Thin Event Bus

This replaces all polling loops. It's not a framework — it's 40 lines.

```python
# signals.py
from enum import Enum
from collections import defaultdict
from typing import Callable

class SIGNALS(str, Enum):
    INPUT_RECEIVED  = "input_received"
    TOKEN_CHUNK     = "token_chunk"
    SENTENCE_READY  = "sentence_ready"
    RESPONSE_READY  = "response_ready"
    USER_SPEAKING   = "user_speaking"
    PLAYBACK_DONE   = "playback_done"

_handlers: dict[str, list[Callable]] = defaultdict(list)

def on(signal: SIGNALS, fn: Callable):
    _handlers[signal].append(fn)

def emit(signal: SIGNALS, payload=None):
    for fn in _handlers[signal]:
        fn(payload)
```

Every module registers its handlers at boot. Nothing polls. Nothing sleeps in a loop waiting for state to change.

---

## 10. main.py — Boot Sequence

```python
# main.py
import constants
import signals
from llm_engine import LLMEngine
from context_builder import ContextBuilder
from memory import Memory
from tts import TTS
from stt import start_stt_thread
from streamingSink import StreamingSink
from audio_player import AudioPlayer
from modules.discord_client import DiscordClient

def boot():
    # 1. Load config (fail fast if malformed)
    cfg = load_yaml("Neuro.yaml")

    # 2. Init core — order matters
    memory   = Memory(cfg)
    builder  = ContextBuilder(cfg, memory)
    engine   = LLMEngine(cfg)
    sink     = StreamingSink()
    player   = AudioPlayer()
    tts      = TTS(cfg)

    # 3. Wire signals
    signals.on(SIGNALS.INPUT_RECEIVED, lambda p: handle_input(p, builder, engine, sink))
    signals.on(SIGNALS.TOKEN_CHUNK,    sink.push)
    signals.on(SIGNALS.SENTENCE_READY, tts.synthesize_and_enqueue)
    signals.on(SIGNALS.USER_SPEAKING,  player.interrupt)

    # 4. Start non-blocking services
    start_stt_thread()          # daemon thread, never blocks main loop
    DiscordClient(cfg).start()  # async discord event loop

    # 5. Lazy-load optional modules AFTER core is warm
    if cfg.get("vtube_enabled"):
        from modules.vtube_studio import VTubeStudio
        VTubeStudio(cfg).connect()

if __name__ == "__main__":
    boot()
```

**VTube Studio is always lazy-loaded.** It never sits in the import graph of the hot path. If it fails to connect, the core agent continues running unaffected.

---

## 11. Hardware Budget Estimates

| Component | RAM | VRAM | Notes |
|---|---|---|---|
| llama3-q4 (8B) via llama.cpp | ~4.5GB | 0 (CPU) or ~5GB | CPU inference ~15 tok/s |
| Whisper-tiny STT | ~150MB | 0 | Acceptable quality |
| Piper TTS | ~60MB | 0 | Fast, offline |
| Memory system (200 episodes) | ~10MB | 0 | Pure Python |
| Discord.py client | ~30MB | 0 | Async, low overhead |
| **Total (CPU-only estimate)** | **~5GB** | **0** | Fits 8GB RAM system |
| **Total (GPU offload, 6GB VRAM)** | **~1.5GB** | **~5GB** | ~40-60 tok/s |

---

## 12. File-by-File Implementation Priority

| Priority | File | Effort | Blocks |
|---|---|---|---|
| 1 | `llm_engine.py` | Medium | Everything |
| 2 | `context_builder.py` | Low | Memory, Discord |
| 3 | `memory.py` | High | Context builder |
| 4 | `streamingSink.py` | Low | TTS pipeline |
| 5 | `tts.py` | Medium | Audio player |
| 6 | `audio_player.py` | Medium | TTS |
| 7 | `stt.py` | Medium | Interrupt behavior |
| 8 | `modules/discord_client.py` | Low | Deploy |
| 9 | `main.py` | Low | Integration |
| 10 | `modules/vtube_studio.py` | Low | Optional, last |
