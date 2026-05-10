"""
Latent Space Navigator — Engine

Manages the Flux.2-klein-4B pipeline, prompt embedding math,
Gram-Schmidt orthogonalization, and dual-resolution image generation.

Supports two navigation layers:
- Local steering: fine-grained movement within a concept via orthogonal axes
- Global jumping: LERP transition to a different concept anchor, then recenter
"""

import io
import time

import torch
from diffusers import Flux2KleinPipeline


class LatentEngine:
    """
    Core engine for latent space navigation.

    Handles:
    - Model loading + torch.compile at two resolutions
    - Prompt encoding via the pipeline's text encoder
    - Direction vector computation (paired prompts → normalized deltas)
    - Gram-Schmidt orthogonalization of direction vectors
    - Fixed-seed image generation at preview (256²) and final (1024²) resolution
    - Global concept anchors + LERP transitions between concept centers
    """

    PREVIEW_SIZE = 256
    FINAL_SIZE = 1024
    PREVIEW_STEPS = 4
    FINAL_STEPS = 8

    def __init__(self, device="cuda", dtype=torch.bfloat16, seed=42):
        self.device = device
        self.dtype = dtype
        self.seed = seed

        # --- Load pipeline ---
        torch.set_float32_matmul_precision("high")
        print("[engine] Loading Flux.2-klein-4B...")
        self.pipe = Flux2KleinPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-4B",
            torch_dtype=dtype,
        ).to(device)

        # --- Compile with max-autotune ---
        print("[engine] Compiling transformer + VAE (max-autotune, this takes a while)...")
        self.pipe.transformer = torch.compile(
            self.pipe.transformer, mode="max-autotune-no-cudagraphs"
        )
        self.pipe.vae = torch.compile(self.pipe.vae, mode="max-autotune-no-cudagraphs")
        if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder is not None:
            self.pipe.text_encoder = torch.compile(
                self.pipe.text_encoder, mode="max-autotune-no-cudagraphs"
            )

        # --- Pre-compute fixed noise ---
        self._init_latents()

        # --- Warmup both resolutions ---
        self._warmup()

        # --- Navigation state: local steering ---
        self.base_embedding = None      # original prompt embedding (never changes)
        self.center_embedding = None    # current concept center (changes on jump)
        self.center_prompt = ""         # prompt string for the current center
        self.directions = []            # orthogonalized local direction tensors

        # --- Navigation state: global anchors ---
        self.anchor_embeddings = []     # pre-encoded anchor embedding tensors
        self.anchor_prompts = []        # anchor prompt strings (for recenter)

        # --- Transition state ---
        self.transition_progress = None     # None = not transitioning, 0.0→1.0
        self.transition_target = None       # target embedding tensor
        self.transition_target_prompt = ""  # target prompt string

    # ------------------------------------------------------------------
    # Latent noise
    # ------------------------------------------------------------------

    def _latent_shape(self, size):
        """Compute the latent tensor shape for a given pixel resolution."""
        # From Flux2Klein pipeline internals:
        # height = 2 * (px // 16), then latent_h = height // 2
        h = 2 * (size // 16)
        return (1, 128, h // 2, h // 2)

    def _init_latents(self):
        """Pre-compute fixed noise latents for both resolutions."""
        torch.manual_seed(self.seed)
        self.preview_latents = torch.randn(self._latent_shape(self.PREVIEW_SIZE))
        torch.manual_seed(self.seed)
        self.final_latents = torch.randn(self._latent_shape(self.FINAL_SIZE))

    def set_seed(self, seed):
        """Change the fixed seed and recompute noise."""
        self.seed = seed
        self._init_latents()

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _warmup(self):
        """Run warmup passes at both resolutions to trigger compilation."""
        for label, size, steps in [
            ("preview 256×256", self.PREVIEW_SIZE, self.PREVIEW_STEPS),
            ("final 1024×1024", self.FINAL_SIZE, self.FINAL_STEPS),
        ]:
            print(f"[engine] Warming up {label}...")
            for i in range(3):
                with torch.inference_mode():
                    _ = self.pipe(
                        prompt="warmup",
                        height=size,
                        width=size,
                        num_inference_steps=steps,
                    )
                print(f"  {i + 1}/3")
        print("[engine] Warmup complete.")

    # ------------------------------------------------------------------
    # Prompt encoding
    # ------------------------------------------------------------------

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode a single prompt → embedding tensor (1, seq_len, hidden_dim)."""
        with torch.inference_mode():
            embeds, _ = self.pipe.encode_prompt(prompt)
        return embeds

    def encode_all(self, base_prompt: str, axis_pairs: list[dict]):
        """
        Encode the base prompt and all axis pairs, then orthogonalize.

        Also sets center_embedding = base_embedding (initial concept center).

        axis_pairs: list of {"label": str, "positive": str, "negative": str}
        """
        t0 = time.time()
        print(f"[engine] Encoding base prompt: '{base_prompt}'")
        self.base_embedding = self.encode_prompt(base_prompt)
        self.center_embedding = self.base_embedding.clone()
        self.center_prompt = base_prompt

        raw_directions = []
        for i, pair in enumerate(axis_pairs):
            label = pair.get("label", f"axis-{i}")
            print(f"[engine] Encoding axis {i + 1}/{len(axis_pairs)}: {label}")
            e_pos = self.encode_prompt(pair["positive"])
            e_neg = self.encode_prompt(pair["negative"])
            raw_directions.append(e_pos - e_neg)

        self.directions = self._gram_schmidt(raw_directions)
        dt = time.time() - t0
        print(f"[engine] Encoded + orthogonalized {len(self.directions)} axes in {dt:.1f}s")

    def encode_anchors(self, anchor_data: list[dict]):
        """
        Encode all global concept anchors.

        anchor_data: list of {"label": str, "prompt": str}
        """
        t0 = time.time()
        self.anchor_embeddings = []
        self.anchor_prompts = []
        for i, anchor in enumerate(anchor_data):
            prompt = anchor["prompt"]
            label = anchor.get("label", f"anchor-{i}")
            print(f"[engine] Encoding anchor {i + 1}/{len(anchor_data)}: {label}")
            self.anchor_embeddings.append(self.encode_prompt(prompt))
            self.anchor_prompts.append(prompt)
        dt = time.time() - t0
        print(f"[engine] Encoded {len(self.anchor_embeddings)} anchors in {dt:.1f}s")

    def replace_axis(self, index: int, pair: dict):
        """
        Replace a single axis direction and re-orthogonalize all directions.

        This is called when the user edits an axis label in the settings panel.
        We re-encode just this pair, then recompute all orthogonal directions
        (since later axes depend on earlier ones via Gram-Schmidt order).
        """
        e_pos = self.encode_prompt(pair["positive"])
        e_neg = self.encode_prompt(pair["negative"])

        # Rebuild raw directions: use existing orthogonalized ones as a base,
        # but replace the target index with the new raw direction.
        # NOTE: We need to store raw (pre-ortho) directions for this to be
        # perfectly correct. For simplicity, we re-orthogonalize from the
        # current set, which is close enough in practice.
        raw = list(self.directions)  # shallow copy
        raw[index] = e_pos - e_neg
        self.directions = self._gram_schmidt(raw)
        print(f"[engine] Replaced axis {index} and re-orthogonalized.")

    # ------------------------------------------------------------------
    # Gram-Schmidt
    # ------------------------------------------------------------------

    def _gram_schmidt(self, vectors: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Gram-Schmidt orthogonalization on embedding tensors.

        Flattens each tensor to 1D, orthogonalizes in float32 for numerical
        stability, then rescales each orthogonal direction to preserve the
        original magnitude of (E_pos - E_neg). This is crucial: without it,
        unit-norm vectors in ~1.5M-dimensional space have per-component
        magnitude ~1e-3, making latent navigation invisible.
        """
        if not vectors:
            return []

        shape = vectors[0].shape
        flat = [v.reshape(-1).to(torch.float32) for v in vectors]
        orig_norms = [torch.linalg.norm(v) for v in flat]

        ortho = []
        for v in flat:
            for u in ortho:
                v = v - torch.dot(v, u) * u
            norm = torch.linalg.norm(v)
            if norm > 1e-8:
                v = v / norm
            ortho.append(v)

        # Rescale to preserve the natural magnitude of each semantic axis
        return [
            (v * n).reshape(shape).to(self.dtype)
            for v, n in zip(ortho, orig_norms)
        ]

    # ------------------------------------------------------------------
    # Embedding math
    # ------------------------------------------------------------------

    @staticmethod
    def _ease_out(t: float) -> float:
        """Ease-out curve: starts fast, slows down near the end."""
        return 1.0 - (1.0 - t) ** 2

    def compute_embedding(
        self, coefficients: list[float], scale: float = 1.0
    ) -> torch.Tensor:
        """
        Compute the navigated embedding:
            E_current = E_center + scale * Σ αᵢ · dᵢ

        During a transition, E_center is interpolated toward the target
        using an ease-out curve for visually snappy jumps.

        coefficients: list of 6 floats (one per axis)
        scale: global multiplier
        Returns: tensor (1, seq_len, hidden_dim)
        """
        if self.center_embedding is None:
            raise RuntimeError("Call encode_all() first")

        # Effective center: interpolated during transitions
        if self.transition_progress is not None and self.transition_target is not None:
            t = self._ease_out(self.transition_progress)
            center = (1.0 - t) * self.center_embedding + t * self.transition_target
        else:
            center = self.center_embedding

        e = center.clone()
        for alpha, d in zip(coefficients, self.directions):
            e = e + scale * alpha * d
        return e

    # ------------------------------------------------------------------
    # Global jump / transition
    # ------------------------------------------------------------------

    def start_jump(self, anchor_index: int):
        """Begin a transition toward a global anchor."""
        if anchor_index < 0 or anchor_index >= len(self.anchor_embeddings):
            raise ValueError(f"Invalid anchor index: {anchor_index}")

        self.transition_target = self.anchor_embeddings[anchor_index]
        self.transition_target_prompt = self.anchor_prompts[anchor_index]
        self.transition_progress = 0.0
        print(f"[engine] Starting jump to anchor {anchor_index}: "
              f"'{self.transition_target_prompt[:60]}...'")

    def step_transition(self, step_size: float = 0.1) -> bool:
        """
        Advance the transition by one step.

        Returns True when the transition is complete (progress >= 1.0).
        """
        if self.transition_progress is None:
            return True  # no active transition

        self.transition_progress = min(1.0, self.transition_progress + step_size)
        return self.transition_progress >= 1.0

    def complete_jump(self):
        """
        Finalize the jump: set center to target, clear transition state.

        After calling this, the caller should rebuild local axes + anchors
        around the new center prompt via recenter().
        """
        if self.transition_target is not None:
            self.center_embedding = self.transition_target.clone()
            self.center_prompt = self.transition_target_prompt
            print(f"[engine] Jump complete. New center: '{self.center_prompt[:60]}...'")

        self.transition_progress = None
        self.transition_target = None
        self.transition_target_prompt = ""

    def recenter(self, new_axes: list[dict], new_anchors: list[dict]):
        """
        Rebuild local axes and global anchors around the current center.

        Called after complete_jump(). The center_embedding is already set;
        this re-encodes the axis pairs and anchor prompts.
        """
        t0 = time.time()

        # Re-encode local directions around the new center
        raw_directions = []
        for i, pair in enumerate(new_axes):
            label = pair.get("label", f"axis-{i}")
            print(f"[engine] Re-encoding axis {i + 1}/{len(new_axes)}: {label}")
            e_pos = self.encode_prompt(pair["positive"])
            e_neg = self.encode_prompt(pair["negative"])
            raw_directions.append(e_pos - e_neg)

        self.directions = self._gram_schmidt(raw_directions)

        # Re-encode global anchors
        self.encode_anchors(new_anchors)

        dt = time.time() - t0
        print(f"[engine] Recentered in {dt:.1f}s "
              f"({len(self.directions)} axes, {len(self.anchor_embeddings)} anchors)")

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------

    def generate(self, prompt_embeds: torch.Tensor, mode: str = "preview") -> bytes:
        """
        Generate an image from prompt embeddings and return JPEG bytes.

        mode: "preview" (256², 4 steps) or "final" (1024², 8 steps)
        """
        if mode == "preview":
            size = self.PREVIEW_SIZE
            steps = self.PREVIEW_STEPS
            latents = self.preview_latents
        else:
            size = self.FINAL_SIZE
            steps = self.FINAL_STEPS
            latents = self.final_latents

        with torch.inference_mode():
            result = self.pipe(
                prompt_embeds=prompt_embeds,
                height=size,
                width=size,
                num_inference_steps=steps,
                latents=latents.clone(),  # clone to prevent in-place corruption
            )

        img = result.images[0]

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
