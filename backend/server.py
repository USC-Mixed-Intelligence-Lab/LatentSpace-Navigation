"""
Latent Space Navigator — WebSocket Server

FastAPI application that serves the frontend and manages real-time
WebSocket communication between the browser and the Flux engine.

Architecture:
  Two concurrent tasks per WebSocket connection:
  - Receiver: reads messages into an asyncio.Queue (never blocks processing)
  - Processor: pulls from queue, drains redundant move messages, processes
    jump/control messages immediately without waiting behind stale moves
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Lazy imports so the module can be imported without torch
engine_instance = None
axes_module = None


def get_engine():
    global engine_instance
    if engine_instance is None:
        from engine import LatentEngine
        engine_instance = LatentEngine(device=_device)
    return engine_instance


def get_axes_module():
    global axes_module
    if axes_module is None:
        import axes as _axes
        axes_module = _axes
    return axes_module


# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Latent Space Navigator")

# Allow cross-origin requests (needed for ngrok tunnels, Cloud Run, etc.)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")


# ------------------------------------------------------------------
# WebSocket endpoint
# ------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("[server] Client connected.")

    # Per-session state
    smoothed = [0.0] * 6
    scale = 1.5
    base_prompt = ""
    axis_data = []
    anchor_data = []
    jumping = False
    jump_steps = 30  # number of transition frames (configurable)

    # Message queue: receiver puts messages here, processor consumes
    msg_queue: asyncio.Queue = asyncio.Queue()
    stop_event = asyncio.Event()

    async def send_status(msg: str):
        await ws.send_text(json.dumps({"type": "status", "msg": msg}))

    async def send_axes(axes: list[dict]):
        await ws.send_text(json.dumps({"type": "axes", "axes": axes}))

    async def send_anchors(anchors: list[dict]):
        await ws.send_text(json.dumps({"type": "anchors", "anchors": anchors}))

    async def send_image(jpeg_bytes: bytes):
        await ws.send_bytes(jpeg_bytes)

    # ---- Receiver task: reads WebSocket → queue ----
    async def receiver():
        try:
            while True:
                raw = await ws.receive_text()
                await msg_queue.put(json.loads(raw))
        except WebSocketDisconnect:
            print("[server] Client disconnected.")
            stop_event.set()
            await msg_queue.put(None)  # sentinel
        except Exception as e:
            print(f"[server] Receiver error: {e}")
            stop_event.set()
            await msg_queue.put(None)

    # ---- Processor task: consumes queue ----
    async def processor():
        nonlocal smoothed, scale, base_prompt, axis_data, anchor_data
        nonlocal jumping, jump_steps

        while not stop_event.is_set():
            msg = await msg_queue.get()
            if msg is None:
                break

            msg_type = msg.get("type")

            # ---- Initialize with base prompt ----
            if msg_type == "init":
                base_prompt = msg["prompt"]
                loop = asyncio.get_event_loop()

                # --- Generate local axes via Gemini ---
                await send_status("Generating semantic axes...")
                axis_data = await loop.run_in_executor(
                    None, get_axes_module().generate_axes, base_prompt
                )
                await send_axes(axis_data)

                # --- Generate global anchors via Gemini ---
                await send_status("Generating concept anchors...")
                anchor_data = await loop.run_in_executor(
                    None, get_axes_module().generate_anchors, base_prompt
                )
                await send_anchors(anchor_data)

                # --- Encode everything ---
                await send_status("Encoding prompts & building directions...")
                await loop.run_in_executor(
                    None, get_engine().encode_all, base_prompt, axis_data
                )
                await loop.run_in_executor(
                    None, get_engine().encode_anchors, anchor_data
                )

                # --- Generate initial image at origin ---
                await send_status("Generating base image...")
                embeds = get_engine().compute_embedding([0.0] * 6, scale)
                jpeg = await loop.run_in_executor(
                    None, get_engine().generate, embeds, "preview"
                )
                await send_image(jpeg)
                await send_status("Ready — move to explore!")
                smoothed = [0.0] * 6

            # ---- Controller movement ----
            elif msg_type == "move":
                if jumping:
                    continue  # skip moves during a jump

                # Drain queue: discard stale moves, keep latest coefficients.
                # If a non-move message appears (e.g. jump), put it back and stop.
                coeffs = msg["coefficients"]
                deferred = []
                while not msg_queue.empty():
                    try:
                        peeked = msg_queue.get_nowait()
                        if peeked is None:
                            stop_event.set()
                            break
                        if peeked.get("type") == "move":
                            coeffs = peeked["coefficients"]  # keep latest
                        else:
                            deferred.append(peeked)
                            break  # stop draining at first control message
                    except asyncio.QueueEmpty:
                        break

                # Re-queue any non-move messages we found
                for d in deferred:
                    await msg_queue.put(d)

                # Update state with latest coefficients (already smoothed
                # by the frontend — no server-side LERP needed)
                for i in range(len(smoothed)):
                    if i < len(coeffs):
                        smoothed[i] = coeffs[i]

                # Generate frame
                embeds = get_engine().compute_embedding(smoothed, scale)
                loop = asyncio.get_event_loop()
                t0 = time.time()
                jpeg = await loop.run_in_executor(
                    None, get_engine().generate, embeds, "preview"
                )
                dt = time.time() - t0
                await send_image(jpeg)
                await ws.send_text(json.dumps({
                    "type": "perf",
                    "gen_ms": round(dt * 1000),
                    "mode": "preview",
                }))

            # ---- High-quality render ----
            elif msg_type == "render_hq":
                await send_status("Rendering high quality...")
                embeds = get_engine().compute_embedding(smoothed, scale)
                loop = asyncio.get_event_loop()
                t0 = time.time()
                jpeg = await loop.run_in_executor(
                    None, get_engine().generate, embeds, "final"
                )
                dt = time.time() - t0
                await send_image(jpeg)
                await ws.send_text(json.dumps({
                    "type": "perf",
                    "gen_ms": round(dt * 1000),
                    "mode": "final",
                }))
                await send_status("High-quality render complete.")

            # ---- Global concept jump ----
            elif msg_type == "jump":
                if jumping:
                    continue
                jumping = True
                try:
                    anchor_idx = msg["index"]
                    anchor_label = anchor_data[anchor_idx].get("label", f"anchor-{anchor_idx}")
                    await send_status(f"Jumping to: {anchor_label}...")
                    loop = asyncio.get_event_loop()

                    # Start transition
                    get_engine().start_jump(anchor_idx)
                    step_size = 1.0 / max(jump_steps, 1)

                    # Run N-frame LERP transition (preserve current axis offsets)
                    # ~300ms per frame → 30 frames ≈ 10s transition
                    min_frame_ms = 300
                    for step in range(jump_steps):
                        frame_t0 = time.time()
                        done = get_engine().step_transition(step_size=step_size)
                        embeds = get_engine().compute_embedding(smoothed, scale)
                        jpeg = await loop.run_in_executor(
                            None, get_engine().generate, embeds, "preview"
                        )
                        await send_image(jpeg)
                        # Enforce minimum frame duration for artistic pacing
                        elapsed_ms = (time.time() - frame_t0) * 1000
                        if elapsed_ms < min_frame_ms:
                            await asyncio.sleep((min_frame_ms - elapsed_ms) / 1000)
                        if done:
                            break

                    # Finalize jump
                    get_engine().complete_jump()
                    new_center_prompt = get_engine().center_prompt

                    # Regenerate local axes + global anchors around new center
                    await send_status("Recentering — generating new axes...")
                    axis_data = await loop.run_in_executor(
                        None, get_axes_module().generate_axes, new_center_prompt
                    )
                    await send_status("Recentering — generating new anchors...")
                    anchor_data = await loop.run_in_executor(
                        None, get_axes_module().generate_anchors, new_center_prompt
                    )

                    # Re-encode directions and anchors
                    await send_status("Recentering — encoding...")
                    await loop.run_in_executor(
                        None, get_engine().recenter, axis_data, anchor_data
                    )

                    # Send updated UI data
                    await send_axes(axis_data)
                    await send_anchors(anchor_data)

                    # Generate a frame at the new center with current axis offsets
                    embeds = get_engine().compute_embedding(smoothed, scale)
                    jpeg = await loop.run_in_executor(
                        None, get_engine().generate, embeds, "preview"
                    )
                    await send_image(jpeg)

                    await ws.send_text(json.dumps({
                        "type": "jump_complete",
                        "prompt": new_center_prompt,
                    }))
                    await send_status(f"Arrived at: {anchor_label}")
                finally:
                    jumping = False

            # ---- Update a single axis ----
            elif msg_type == "update_axis":
                idx = msg["index"]
                label = msg["label"]
                await send_status(f"Regenerating axis: {label}...")

                loop = asyncio.get_event_loop()
                current_prompt = get_engine().center_prompt or base_prompt
                new_pair = await loop.run_in_executor(
                    None,
                    get_axes_module().regenerate_axis,
                    current_prompt,
                    label,
                )
                axis_data[idx] = new_pair

                await loop.run_in_executor(
                    None, get_engine().replace_axis, idx, new_pair
                )
                await send_axes(axis_data)
                await send_status(f"Axis '{label}' updated.")

                # Re-generate preview at current position
                embeds = get_engine().compute_embedding(smoothed, scale)
                jpeg = await loop.run_in_executor(
                    None, get_engine().generate, embeds, "preview"
                )
                await send_image(jpeg)

            # ---- Settings ----
            elif msg_type == "set_seed":
                get_engine().set_seed(msg["seed"])
                await send_status(f"Seed changed to {msg['seed']}.")

            elif msg_type == "set_scale":
                scale = float(msg["scale"])
                await send_status(f"Scale set to {scale:.2f}.")

            elif msg_type == "set_jump_steps":
                jump_steps = max(5, min(60, int(msg["steps"])))
                await send_status(f"Jump speed: {jump_steps} frames.")

    # ---- Run both tasks concurrently ----
    receiver_task = asyncio.create_task(receiver())
    processor_task = asyncio.create_task(processor())

    try:
        # Wait for either task to finish (disconnect or error)
        done, pending = await asyncio.wait(
            [receiver_task, processor_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        # Re-raise any exceptions from completed tasks
        for task in done:
            if task.exception():
                print(f"[server] Task error: {task.exception()}")
    except Exception as e:
        print(f"[server] Error: {e}")
        import traceback
        traceback.print_exc()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

_device = "cuda"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent Space Navigator Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8765, help="Port")
    parser.add_argument("--device", default="cuda", help="cuda / mps / cpu")
    args = parser.parse_args()

    _device = args.device

    print(f"[server] Starting on {args.host}:{args.port} (device={args.device})")
    print(f"[server] Frontend dir: {FRONTEND_DIR}")

    # Pre-load engine at startup
    get_engine()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        proxy_headers=True,
        forwarded_allow_ips="*",
        ws_ping_interval=30,
        ws_ping_timeout=30,
    )
