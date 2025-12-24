"""
CLI Runner for Multimodal Product Retrieval

Supports:
- Text → Image retrieval
- Image → Text retrieval

Assumes embeddings are precomputed and stored on disk.
"""

import argparse
import numpy as np
from PIL import Image

from encoders.image_encoder import ImageEncoder
from encoders.text_encoder import TextEncoder
from indexing.faiss_index import FaissIndex


def parse_args():
    parser = argparse.ArgumentParser(description="Run multimodal product retrieval")

    parser.add_argument("--artifacts_dir", type=str, required=True, help="Directory with saved embeddings")
    parser.add_argument("--query_text", type=str, default=None, help="Text query")
    parser.add_argument("--query_image", type=str, default=None, help="Path to image query")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results")
    parser.add_argument("--model_name", type=str, default="ViT-B/32")
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load artifacts
    product_ids = np.load(f"{args.artifacts_dir}/product_ids.npy")
    image_embeddings = np.load(f"{args.artifacts_dir}/image_embeddings.npy").astype(np.float32)
    text_embeddings = np.load(f"{args.artifacts_dir}/text_embeddings.npy").astype(np.float32)

    embedding_dim = image_embeddings.shape[1]

    # Build FAISS index on image embeddings
    index = FaissIndex(embedding_dim=embedding_dim)
    index.add(image_embeddings)

    # Initialize encoders
    image_encoder = ImageEncoder(model_name=args.model_name, device=args.device)
    text_encoder = TextEncoder(model_name=args.model_name, device=args.device)

    if args.query_text is not None:
        print(f"🔎 Text → Image search: '{args.query_text}'")

        query_embedding = text_encoder.encode([args.query_text])
        scores, indices = index.search(query_embedding, top_k=args.top_k)

        for rank, idx in enumerate(indices[0]):
            pid = product_ids[idx]
            score = scores[0][rank]
            print(f"{rank+1}. Product ID: {pid} | Similarity: {score:.4f}")

    elif args.query_image is not None:
        print(f"🔎 Image → Text search: {args.query_image}")

        img = Image.open(args.query_image).convert("RGB")
        query_embedding = image_encoder.encode([img])

        scores, indices = index.search(query_embedding, top_k=args.top_k)

        for rank, idx in enumerate(indices[0]):
            pid = product_ids[idx]
            score = scores[0][rank]
            print(f"{rank+1}. Product ID: {pid} | Similarity: {score:.4f}")

    else:
        raise ValueError("You must provide either --query_text or --query_image")

    print("✅ Retrieval complete.")


if __name__ == "__main__":
    main()
