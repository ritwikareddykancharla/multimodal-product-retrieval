"""
Text Encoder Module (CLIP-style)

Responsibilities:
- Load a text encoder from a CLIP-style model
- Convert text descriptions into normalized embedding vectors
- Support batched inference

This module ONLY handles text encoding.
"""

from typing import List
import numpy as np
import torch

try:
    import clip
except ImportError:
    raise ImportError(
        "CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git"
    )


class TextEncoder:
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda",
        normalize: bool = True,
    ):
        """
        Args:
            model_name: CLIP model name (must match image encoder)
            device: 'cuda' or 'cpu'
            normalize: whether to L2-normalize embeddings
        """
        self.device = device
        self.normalize = normalize

        self.model, _ = clip.load(model_name, device=device)
        self.model.eval()

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Encode a list of text strings into embedding vectors.

        Args:
            texts: list of product descriptions / queries
            batch_size: batch size for encoding

        Returns:
            embeddings: np.ndarray of shape (N, D)
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            tokens = clip.tokenize(batch_texts).to(self.device)

            text_features = self.model.encode_text(tokens)

            if self.normalize:
                text_features = text_features / text_features.norm(
                    dim=-1, keepdim=True
                )

            all_embeddings.append(text_features.cpu().numpy())

        return np.vstack(all_embeddings)
