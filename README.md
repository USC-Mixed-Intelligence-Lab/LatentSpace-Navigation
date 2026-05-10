# 🌌 Latent Space Navigator

Explore a diffusion model's latent space in real-time using hand gestures or a DualShock controller.

Move through 6 semantic dimensions — style, texture, mood, lighting, composition, abstraction — with paired prompt embeddings and orthogonalized direction vectors.

## How It Works

1. **Enter a base prompt** (e.g., "a small robot sitting in a rainy alley, cinematic lighting")
2. **Gemini Flash** generates 6 contrastive prompt pairs that isolate semantic dimensions
3. **Flux.2-klein-4B** encodes all prompts; directions are orthogonalized via Gram-Schmidt
4. **Move** with your hand / controller / sliders — coefficients control position in latent space
5. **Images update in real-time** at 256² preview, with auto high-quality 1024² renders on pause

```
E_current = E_base + scale × Σ αᵢ · dᵢ
```

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA GPU with ≥13 GB VRAM (for Flux.2-klein-4B)
- A [Google AI API key](https://aistudio.google.com/apikey)

### 1. Setup

```bash
# Clone
git clone https://github.com/your-user/latent-space-navigator.git
cd latent-space-navigator

# Install dependencies
pip install -r backend/requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 2. Run Locally

```bash
python backend/server.py --device cuda
```

Open `http://localhost:8765` in your browser.

### 3. Run on Google Colab

```python
# In a Colab notebook with GPU runtime:

# Install dependencies
!pip install -r backend/requirements.txt
!pip install pyngrok

# Set API keys
import os
os.environ["GOOGLE_API_KEY"] = "your-key-here"

# Start server in background
import subprocess
proc = subprocess.Popen(["python", "backend/server.py", "--device", "cuda"])

# Expose via ngrok (requires free account: https://dashboard.ngrok.com/signup)
from pyngrok import ngrok
ngrok.set_auth_token(os.environ.get("NGROK_AUTHTOKEN", "your-ngrok-authtoken"))
tunnel = ngrok.connect(8765)
print(f"Open: {tunnel.public_url}")
```

> **Note:** Get your free ngrok authtoken at https://dashboard.ngrok.com/get-started/your-authtoken.
> You can set it as a [Colab secret](https://colab.research.google.com/) or pass it directly:
> ```python
> from google.colab import userdata
> os.environ["NGROK_AUTHTOKEN"] = userdata.get("NGROK_AUTHTOKEN")
> ```

### 4. Deploy to Cloud Run (GPU)

```bash
# Build and deploy
gcloud run deploy latent-navigator \
  --source . \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --memory 16Gi \
  --set-env-vars GOOGLE_API_KEY=your-key \
  --allow-unauthenticated
```

## Controls

### Hand Tracking 🖐️
| Gesture | Axis |
|---------|------|
| Palm X position | Style |
| Palm Y position | Texture |
| Palm depth (Z) | Mood |
| Finger pitch | Lighting |
| Hand spread | Composition |
| Thumb-index pinch | Abstraction |

### DualShock Controller 🎮
| Input | Axis |
|-------|------|
| Left stick X | Style |
| Left stick Y | Texture |
| Right stick X | Mood |
| Right stick Y | Lighting |
| L2 trigger | Composition |
| R2 trigger | Abstraction |

### Settings
- **Scale**: Global coefficient multiplier (how far you can travel)
- **Smoothing**: Input smoothing factor (lower = smoother, higher = responsive)
- **Seed**: Fixed noise seed for visual continuity
- **Axis labels**: Click any label to rename it — triggers re-generation of that axis's prompts

## Architecture

```
backend/
  server.py     — FastAPI + WebSocket (serves frontend + handles real-time comms)
  engine.py     — Flux pipeline, embeddings, Gram-Schmidt, dual-resolution generation
  axes.py       — Gemini 3.1 Flash Lite for semantic axis pair generation

frontend/
  index.html    — UI layout
  style.css     — Dark mode, glassmorphism
  app.js        — WebSocket, MediaPipe hand tracking, Gamepad API, visualizer
```

## License

MIT
