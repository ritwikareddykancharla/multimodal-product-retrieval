"""
Retrieval Evaluation Metrics for Multimodal Product Retrieval

Implements Recall@K for:
- Text → Image retrieval
- Image → Text retrieval

Assumes one-to-one correspondence between image and text embeddings
(i.e., same index = same product).
"""

from typing import List
import numpy as np


def recall_at_k(
    similarity_matrix: np.ndarray,
    k: int,
) -> float:
    """
    Compute Recall@K.

    Args:
        similarity_matrix: np.ndarray of shape (N, N)
            similarity_matrix[i, j] = similarity between query i and item j
        k: top-K cutoff

    Returns:
        Recall@K value
    """
    assert similarity_matrix.ndim == 2
    num_queries = similarity_matrix.shape[0]

    correct = 0
    for i in range(num_queries):
        top_k_indices = np.argsort(similarity_matrix[i])[::-1][:k]
        if i in top_k_indices:
            correct += 1

    return correct / num_queries


def evaluate_cross_modal_retrieval(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    ks: List[int] = [1, 5, 10],
) -> dict:
    """
    Evaluate image-text and text-image retrieval.

    Args:
        image_embeddings: (N, D) image embeddings (normalized)
        text_embeddings: (N, D) text embeddings (normalized)
        ks: list of K values for Recall@K

    Returns:
        metrics dictionary
    """
    assert image_embeddings.shape == text_embeddings.shape

    # Cosine similarity via dot product (normalized embeddings)
    sim_image_to_text = image_embeddings @ text_embeddings.T
    sim_text_to_image = text_embeddings @ image_embeddings.T

    results = {}

    for k in ks:
        results[f"image_to_text_recall@{k}"] = recall_at_k(
            sim_image_to_text, k
        )
        results[f"text_to_image_recall@{k}"] = recall_at_k(
            sim_text_to_image, k
        )

    return results


if __name__ == "__main__":
    # Example sanity check
    N, D = 100, 512
    img_emb = np.random.randn(N, D)
    txt_emb = np.random.randn(N, D)

    # Normalize
    img_emb /= np.linalg.norm(img_emb, axis=1, keepdims=True)
    txt_emb /= np.linalg.norm(txt_emb, axis=1, keepdims=True)

    metrics = evaluate_cross_modal_retrieval(img_emb, txt_emb)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
