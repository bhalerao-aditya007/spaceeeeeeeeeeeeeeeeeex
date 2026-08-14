"""
backend/api.py
--------------
FastAPI server that:
  1. Subscribes to all Redis channels (perception.out, cognition.out, action.out, interface.out)
  2. Broadcasts every message to all connected WebSocket clients in real-time
  3. Accepts human override commands from the frontend and publishes to human.in
  4. Exposes a POST /scenario endpoint to trigger scenarios programmatically

Run:
    pip install fastapi uvicorn redis
    python backend/api.py

Requires Redis on localhost:6379 (same as integration.py)
"""

import json
import asyncio
import threading
from datetime import datetime

import redis
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ─── Config ───────────────────────────────────────────────────────────────────
REDIS_HOST = "localhost"
REDIS_PORT = 6379
SUBSCRIBE_CHANNELS = [
    "perception.out",
    "cognition.out",
    "action.out",
    "interface.out",
    "orchestrator.status",
]
HUMAN_IN_CHANNEL = "human.in"

# ─── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="Spacecraft Autonomy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── State ────────────────────────────────────────────────────────────────────
connected_clients: list[WebSocket] = []
latest_state: dict = {
    "perception": None,
    "cognition": None,
    "action": None,
    "interface": None,
    "orchestrator": None,
}

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


# ─── Redis subscriber (runs in background thread) ─────────────────────────────
def redis_listener(loop: asyncio.AbstractEventLoop):
    """
    Blocking Redis subscriber in its own thread.
    Whenever a message arrives, it schedules a broadcast on the FastAPI event loop.
    """
    pubsub = r.pubsub()
    pubsub.subscribe(*SUBSCRIBE_CHANNELS)

    for message in pubsub.listen():
        if message["type"] != "message":
            continue

        channel: str = message["channel"]
        raw: str = message["data"]

        # Try to parse JSON payload; fall back to raw string
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            payload = {"raw": raw}

        # Determine agent name from channel
        agent = channel.split(".")[0]  # e.g. "perception" from "perception.out"

        # Build the envelope we send to the frontend
        envelope = {
            "channel": channel,
            "agent": agent,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data": payload,
        }

        # Update latest state snapshot
        latest_state[agent] = envelope

        # Broadcast to all WS clients (thread-safe via asyncio)
        asyncio.run_coroutine_threadsafe(broadcast(envelope), loop)


async def broadcast(message: dict):
    dead = []
    for ws in connected_clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connected_clients.remove(ws)


# ─── Startup: launch Redis listener thread ────────────────────────────────────
@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    t = threading.Thread(target=redis_listener, args=(loop,), daemon=True)
    t.start()
    print(f"[API] Redis listener started on {REDIS_HOST}:{REDIS_PORT}")
    print(f"[API] Subscribed to: {SUBSCRIBE_CHANNELS}")


# ─── WebSocket endpoint ───────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    print(f"[WS] Client connected. Total: {len(connected_clients)}")

    # Send current state snapshot immediately on connect
    await websocket.send_json({
        "channel": "system.snapshot",
        "agent": "system",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data": latest_state,
    })

    try:
        while True:
            # Receive override commands from frontend
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
                command = msg.get("command", "")
                level = msg.get("level", 1)
                rationale = msg.get("rationale", "")

                override_payload = json.dumps({
                    "source": "human",
                    "command": command,
                    "level": level,          # Armstrong Protocol level 1-4
                    "rationale": rationale,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                })
                r.publish(HUMAN_IN_CHANNEL, override_payload)
                print(f"[API] Override published → human.in: {override_payload}")

            except Exception as e:
                print(f"[API] Bad WS message: {e}")

    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        print(f"[WS] Client disconnected. Total: {len(connected_clients)}")


# ─── REST: get latest snapshot ────────────────────────────────────────────────
@app.get("/status")
def get_status():
    return JSONResponse(content=latest_state)


# ─── REST: health check ───────────────────────────────────────────────────────
@app.get("/health")
def health():
    try:
        r.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    return {
        "api": "ok",
        "redis": "ok" if redis_ok else "ERROR - is Redis running?",
        "clients_connected": len(connected_clients),
    }


# ─── REST: trigger a human override via HTTP (optional convenience) ───────────
@app.post("/override")
async def post_override(body: dict):
    """
    Body: { "command": "hold_position", "level": 2, "rationale": "sun glare" }
    """
    payload = json.dumps({
        "source": "human",
        "command": body.get("command", "hold_position"),
        "level": body.get("level", 1),
        "rationale": body.get("rationale", ""),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })
    r.publish(HUMAN_IN_CHANNEL, payload)
    return {"published": True, "payload": payload}


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
