"""
Image Encoder Module (CLIP-style)

Responsibilities:
- Load a vision encoder from a CLIP-style model
- Convert images into normalized embedding vectors
- Support batched GPU inference

This module ONLY handles image encoding.
"""

from typing import List
import numpy as np
import torch
from PIL import Image

try:
    import clip
except ImportError:
    raise ImportError(
        "CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git"
    )


class ImageEncoder:
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda",
        normalize: bool = True,
    ):
        """
        Args:
            model_name: CLIP vision backbone
            device: 'cuda' or 'cpu'
            normalize: whether to L2-normalize embeddings
        """
        self.device = device
        self.normalize = normalize

        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

    @torch.no_grad()
    def encode(
        self,
        images: List[Image.Image],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode a list of PIL images into embedding vectors.

        Args:
            images: list of PIL Image objects
            batch_size: batch size for encoding

        Returns:
            embeddings: np.ndarray of shape (N, D)
        """
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            batch_tensor = torch.stack(
                [self.preprocess(img) for img in batch_images]
            ).to(self.device)

            image_features = self.model.encode_image(batch_tensor)

            if self.normalize:
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

            all_embeddings.append(image_features.cpu().numpy())

        return np.vstack(all_embeddings)
