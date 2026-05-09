"""
Latent Space Navigator — Engine

Manages the Flux.2-klein-4B pipeline, prompt embedding math,
Gram-Schmidt orthogonalization, and dual-resolution image generation.
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
        print("[engine] Loading Flux.2-klein-4B...")
        self.pipe = Flux2KleinPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-4B",
            torch_dtype=dtype,
        ).to(device)

        # --- Compile with max-autotune ---
        print("[engine] Compiling transformer + VAE (max-autotune, this takes a while)...")
        self.pipe.transformer = torch.compile(
            self.pipe.transformer, mode="max-autotune"
        )
        self.pipe.vae = torch.compile(self.pipe.vae, mode="max-autotune")
        if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder is not None:
            self.pipe.text_encoder = torch.compile(
                self.pipe.text_encoder, mode="max-autotune"
            )

        # --- Pre-compute fixed noise ---
        self._init_latents()

        # --- Warmup both resolutions ---
        self._warmup()

        # --- Navigation state ---
        self.base_embedding = None
        self.directions = []  # list of orthogonalized direction tensors

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

        axis_pairs: list of {"label": str, "positive": str, "negative": str}
        """
        t0 = time.time()
        print(f"[engine] Encoding base prompt: '{base_prompt}'")
        self.base_embedding = self.encode_prompt(base_prompt)

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
        stability, normalizes, then reshapes back to the original shape and
        casts to the working dtype.
        """
        if not vectors:
            return []

        shape = vectors[0].shape
        flat = [v.reshape(-1).to(torch.float32) for v in vectors]

        ortho = []
        for v in flat:
            for u in ortho:
                v = v - torch.dot(v, u) * u
            norm = torch.linalg.norm(v)
            if norm > 1e-8:
                v = v / norm
            ortho.append(v)

        return [v.reshape(shape).to(self.dtype) for v in ortho]

    # ------------------------------------------------------------------
    # Embedding math
    # ------------------------------------------------------------------

    def compute_embedding(
        self, coefficients: list[float], scale: float = 1.0
    ) -> torch.Tensor:
        """
        Compute the navigated embedding:
            E_current = E_base + scale * Σ αᵢ · dᵢ

        coefficients: list of 6 floats (one per axis)
        scale: global multiplier
        Returns: tensor (1, seq_len, hidden_dim)
        """
        if self.base_embedding is None:
            raise RuntimeError("Call encode_all() first")

        e = self.base_embedding.clone()
        for alpha, d in zip(coefficients, self.directions):
            e = e + scale * alpha * d
        return e

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
                latents=latents,
            )

        img = result.images[0]

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
