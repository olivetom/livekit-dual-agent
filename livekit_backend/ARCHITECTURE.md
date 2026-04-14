# LiveKit Dual-Agent Voice Backend

## Overview

Implementation of the **dual-agent architecture** described by Vocal Bridge:
a _foreground agent_ for real-time voice conversation and a _background agent_
for reasoning, guardrails, and tool calls.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Browser / Mobile Client                       │
│            (publishes microphone, subscribes to agent audio)         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  WebRTC (LiveKit room)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        LiveKit SFU / Cloud                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  WebSocket dispatch
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Python Worker (main.py)                          │
│                                                                      │
│  ┌──────────────────────────────┐   AgentMessageBus (asyncio)       │
│  │      ForegroundAgent         │◄──────────────────────────────┐   │
│  │                              │                               │   │
│  │  Deepgram STT                │  UserUtterance (published)    │   │
│  │     │                        │──────────────────────────────►│   │
│  │     ▼                        │                               │   │
│  │  gpt-4o-mini (fast LLM)      │  BackgroundResult (received)  │   │
│  │     │                        │◄──────────────────────────────│   │
│  │     ▼                        │                               │   │
│  │  ElevenLabs TTS              │               ┌───────────────┴─┐ │
│  │     │                        │               │ BackgroundAgent  │ │
│  │     ▼                        │               │                 │ │
│  │  LiveKit audio track ────────┼──► user       │  Claude Sonnet  │ │
│  └──────────────────────────────┘   hears it   │  (reasoning +   │ │
│                                                 │   guardrails +  │ │
│                                                 │   tool calls)   │ │
│                                                 │                 │ │
│                                                 │  Tools:         │ │
│                                                 │  • calculate()  │ │
│                                                 │  • knowledge_   │ │
│                                                 │    lookup()     │ │
│                                                 │  • get_time()   │ │
│                                                 └─────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### Why two agents?

| Concern | Foreground | Background |
|---|---|---|
| Latency | ~300–500 ms | 2–8 s acceptable |
| Model | gpt-4o-mini (fast) | claude-sonnet-4-6 (capable) |
| Speaks? | Yes (TTS) | No |
| Tools | No | Yes (agentic loop) |
| Guardrails | Defers to BG | Authoritative |
| Context | Injects BG results | Produces context |

### Message Bus

`AgentMessageBus` is a lightweight asyncio pub/sub channel:

```
UserUtterance  →  BackgroundAgent  →  BackgroundResult  →  ForegroundAgent
```

Both agents share a reference to the bus; no HTTP calls or external queues
needed inside a single worker process.

### Guardrails Flow

```
User speaks
    │
    ▼
ForegroundAgent publishes utterance ──────────────────────────────────────┐
    │                                                                      │
    │  (fast path: FG immediately starts generating a reply)              │
    │                                                                      │
    │                                               BackgroundAgent        │
    │                                               checks guardrails      │
    │                                                      │               │
    │                                           ┌──────────┴──────────┐   │
    │                                           │                     │   │
    │                                      PASS ▼                FAIL ▼   │
    │                                    enrich context     publish block  │
    │                                           │                 │       │
    │◄──────────────────────────────────────────┘                 │       │
    │                                                             │       │
    ▼                                                             ▼       │
ForegroundAgent speaks reply                    ForegroundAgent interrupts│
(with enriched context if available)            and speaks refusal        │
```

### Extended Thinking

The background agent enables Anthropic's **extended thinking** on the initial
reasoning pass (`budget_tokens=512`).  This improves guardrail accuracy and
context quality at the cost of a small extra latency budget that is hidden from
the user.

## File Structure

```
livekit_backend/
├── main.py                  # Worker entrypoint + FastAPI HTTP server
├── config.py                # Environment-based configuration
├── message_bus.py           # Async pub/sub bridge between agents
├── agents/
│   ├── foreground_agent.py  # Real-time voice pipeline (STT→LLM→TTS)
│   └── background_agent.py  # Reasoning, guardrails, tool calls (Claude)
├── tools/
│   └── definitions.py       # Tool registry + concrete implementations
├── requirements.txt
└── .env.example
```

## Running Locally

```bash
cd livekit_backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your keys
python main.py dev     # LiveKit dev mode
```

The FastAPI HTTP server starts automatically on port 8000.
Request a token from your frontend:

```bash
curl -X POST http://localhost:8000/token \
  -H 'Content-Type: application/json' \
  -d '{"room_name": "my-room", "participant_identity": "alice"}'
```

## Adding Tools

Open `tools/definitions.py` and decorate a new async function:

```python
@tool(
    description="Fetch weather for a city.",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)
async def get_weather(city: str) -> dict:
    # call your weather API here
    return {"city": city, "temp_c": 22}
```

The background agent picks it up automatically on the next restart.
