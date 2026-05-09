"""
Latent Space Navigator — Axis Generator

Uses Gemini 3.1 Flash Lite to generate semantic axis pairs from a base prompt.
Each axis is a contrastive pair (e.g., "watercolor" ↔ "photorealistic") that
isolates a single semantic dimension for latent space navigation.
"""

import os
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

load_dotenv()

# ------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------

class AxisPair(BaseModel):
    """A single semantic axis defined by a contrastive prompt pair."""
    label: str
    positive: str
    negative: str


class SemanticAxes(BaseModel):
    """The full set of 6 semantic axes for latent space navigation."""
    axes: list[AxisPair]


# ------------------------------------------------------------------
# Client
# ------------------------------------------------------------------

_client = None

def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY not set. Copy .env.example to .env and add your key."
            )
        _client = genai.Client(api_key=api_key)
    return _client


# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a semantic axis generator for latent space exploration of image generation models.

Given a base image prompt, generate exactly 6 pairs of contrasting prompts. Each pair \
isolates a single semantic dimension by modifying ONLY that dimension while keeping \
everything else about the base prompt identical.

Rules:
- The base prompt's subject, scene, and core elements must be IDENTICAL in both the \
positive and negative prompt of each pair.
- Only modify the specific semantic dimension being targeted.
- The positive and negative should be clear semantic opposites.
- Do NOT add unrelated details, characters, or objects.
- Append the style/modification naturally to the base prompt.

The 6 dimensions to generate (use these as labels):
1. style — artistic rendering (e.g., "watercolor painting" ↔ "photorealistic photograph")
2. texture — surface/material quality (e.g., "smooth polished glass" ↔ "rough weathered stone")
3. mood — emotional atmosphere (e.g., "warm hopeful serene" ↔ "cold lonely melancholic")
4. lighting — light quality (e.g., "golden hour warm sunlight" ↔ "harsh cold clinical lighting")
5. composition — framing/perspective (e.g., "extreme macro close-up" ↔ "wide cinematic establishing shot")
6. abstraction — detail level (e.g., "minimalist abstract simplified" ↔ "hyperdetailed intricate realistic")
"""

_SINGLE_AXIS_PROMPT = """\
You are a semantic axis generator for latent space exploration of image generation models.

Given a base image prompt and a custom axis label, generate exactly 1 contrastive prompt \
pair that isolates that semantic dimension.

Rules:
- The base prompt's subject, scene, and core elements must be IDENTICAL in both prompts.
- Only modify the specified semantic dimension.
- The positive and negative should be clear semantic opposites.
- Do NOT add unrelated details.
- Append the modification naturally to the base prompt.

Axis to generate: "{axis_label}"
Interpret this label creatively as a semantic spectrum and produce a meaningful contrastive pair.
"""


# ------------------------------------------------------------------
# API functions
# ------------------------------------------------------------------

def generate_axes(base_prompt: str) -> list[dict]:
    """
    Generate all 6 semantic axis pairs for a base prompt.

    Returns: list of {"label": str, "positive": str, "negative": str}
    """
    client = _get_client()

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=f"Base prompt: \"{base_prompt}\"",
        config={
            "system_instruction": _SYSTEM_PROMPT,
            "response_mime_type": "application/json",
            "response_schema": SemanticAxes,
        },
    )

    axes = response.parsed
    result = [a.model_dump() for a in axes.axes]
    print(f"[axes] Generated {len(result)} axis pairs for: '{base_prompt}'")
    for a in result:
        print(f"  {a['label']}: {a['positive'][:60]}... ↔ {a['negative'][:60]}...")
    return result


def regenerate_axis(base_prompt: str, axis_label: str) -> dict:
    """
    Regenerate a single axis pair for a custom label.

    Returns: {"label": str, "positive": str, "negative": str}
    """
    client = _get_client()

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=f"Base prompt: \"{base_prompt}\"\nAxis label: \"{axis_label}\"",
        config={
            "system_instruction": _SINGLE_AXIS_PROMPT.format(axis_label=axis_label),
            "response_mime_type": "application/json",
            "response_schema": AxisPair,
        },
    )

    pair = response.parsed.model_dump()
    pair["label"] = axis_label  # ensure the label matches what the user typed
    print(f"[axes] Regenerated axis '{axis_label}': {pair['positive'][:60]}... ↔ {pair['negative'][:60]}...")
    return pair
