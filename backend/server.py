"""
Latent Space Navigator — WebSocket Server

FastAPI application that serves the frontend and manages real-time
WebSocket communication between the browser and the Flux engine.
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
    generating = False

    async def send_status(msg: str):
        await ws.send_text(json.dumps({"type": "status", "msg": msg}))

    async def send_axes(axes: list[dict]):
        await ws.send_text(json.dumps({"type": "axes", "axes": axes}))

    async def send_image(jpeg_bytes: bytes):
        await ws.send_bytes(jpeg_bytes)

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            # ---- Initialize with base prompt ----
            if msg_type == "init":
                base_prompt = msg["prompt"]
                await send_status("Generating semantic axes...")

                # Generate axis pairs via Gemini
                loop = asyncio.get_event_loop()
                axis_data = await loop.run_in_executor(
                    None, get_axes_module().generate_axes, base_prompt
                )
                await send_axes(axis_data)
                await send_status("Encoding prompts & building directions...")

                # Encode all prompts and orthogonalize
                await loop.run_in_executor(
                    None, get_engine().encode_all, base_prompt, axis_data
                )

                # Generate initial image at origin
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
                if generating:
                    continue  # skip if still generating previous frame

                coeffs = msg["coefficients"]

                # Server-side LERP smoothing
                lerp_factor = 0.15
                for i in range(len(smoothed)):
                    if i < len(coeffs):
                        smoothed[i] += lerp_factor * (coeffs[i] - smoothed[i])

                generating = True
                try:
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
                finally:
                    generating = False

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

            # ---- Update a single axis ----
            elif msg_type == "update_axis":
                idx = msg["index"]
                label = msg["label"]
                await send_status(f"Regenerating axis: {label}...")

                loop = asyncio.get_event_loop()
                new_pair = await loop.run_in_executor(
                    None,
                    get_axes_module().regenerate_axis,
                    base_prompt,
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

    except WebSocketDisconnect:
        print("[server] Client disconnected.")
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

    uvicorn.run(app, host=args.host, port=args.port)
