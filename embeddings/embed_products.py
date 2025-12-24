"""
Offline Embedding Generation for Multimodal Product Retrieval

Responsibilities:
- Load product dataset
- Generate image and text embeddings using CLIP encoders
- Save embeddings and product IDs to disk for indexing

This script is designed for offline, batched processing.
"""

import os
import argparse
import numpy as np
from typing import Dict

from encoders.image_encoder import ImageEncoder
from encoders.text_encoder import TextEncoder
from data.dataset import ProductDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Generate multimodal product embeddings")

    parser.add_argument("--csv_path", type=str, required=True, help="Path to product CSV")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory for images")
    parser.add_argument("--output_dir", type=str, default="artifacts", help="Output directory")
    parser.add_argument("--model_name", type=str, default="ViT-B/32", help="CLIP model name")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--image_batch_size", type=int, default=32)
    parser.add_argument("--text_batch_size", type=int, default=64)

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    dataset = ProductDataset(
        csv_path=args.csv_path,
        image_root=args.image_root,
    )

    product_ids = dataset.get_product_ids()
    images = dataset.get_images()
    texts = dataset.get_texts()

    # Initialize encoders (MUST use same CLIP checkpoint)
    image_encoder = ImageEncoder(
        model_name=args.model_name,
        device=args.device,
    )
    text_encoder = TextEncoder(
        model_name=args.model_name,
        device=args.device,
    )

    # Encode
    print("🔹 Encoding images...")
    image_embeddings = image_encoder.encode(
        images, batch_size=args.image_batch_size
    )

    print("🔹 Encoding text...")
    text_embeddings = text_encoder.encode(
        texts, batch_size=args.text_batch_size
    )

    assert image_embeddings.shape == text_embeddings.shape, (
        "Image and text embeddings must have same shape"
    )

    # Save artifacts
    artifacts: Dict[str, np.ndarray] = {
        "product_ids": np.array(product_ids),
        "image_embeddings": image_embeddings.astype(np.float32),
        "text_embeddings": text_embeddings.astype(np.float32),
    }

    for name, array in artifacts.items():
        path = os.path.join(args.output_dir, f"{name}.npy")
        np.save(path, array)
        print(f"✅ Saved {name} → {path}")

    print("🎉 Embedding generation complete.")


if __name__ == "__main__":
    main()
