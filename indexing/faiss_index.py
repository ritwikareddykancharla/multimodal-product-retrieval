"""
FAISS Index Utilities for Multimodal Retrieval

Responsibilities:
- Build FAISS indices for fast nearest neighbor search
- Support cosine similarity via inner product
- Save and load indices for reuse

This module is modality-agnostic: it works for image or text embeddings.
"""

import os
from typing import Tuple
import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError(
        "FAISS not installed. Run: pip install faiss-cpu or faiss-gpu"
    )


class FaissIndex:
    def __init__(
        self,
        embedding_dim: int,
        use_gpu: bool = False,
    ):
        """
        Args:
            embedding_dim: dimensionality of embedding vectors
            use_gpu: whether to use FAISS GPU (if available)
        """
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu

        # Inner product index (for cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(embedding_dim)

        if use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def add(self, embeddings: np.ndarray):
        """
        Add embeddings to the index.

        Args:
            embeddings: np.ndarray of shape (N, D), must be float32 and normalized
        """
        assert embeddings.dtype == np.float32
        assert embeddings.shape[1] == self.embedding_dim
        self.index.add(embeddings)

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search nearest neighbors.

        Args:
            query_embeddings: np.ndarray (Q, D)
            top_k: number of neighbors to retrieve

        Returns:
            distances: cosine similarity scores (Q, top_k)
            indices: indices of nearest neighbors (Q, top_k)
        """
        assert query_embeddings.dtype == np.float32
        assert query_embeddings.shape[1] == self.embedding_dim

        distances, indices = self.index.search(query_embeddings, top_k)
        return distances, indices

    def save(self, path: str):
        """
        Save FAISS index to disk.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if self.use_gpu:
            index_cpu = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(index_cpu, path)
        else:
            faiss.write_index(self.index, path)

    @classmethod
    def load(
        cls,
        path: str,
        use_gpu: bool = False,
    ) -> "FaissIndex":
        """
        Load FAISS index from disk.
        """
        index = faiss.read_index(path)
        embedding_dim = index.d

        faiss_index = cls(
            embedding_dim=embedding_dim,
            use_gpu=False,
        )
        faiss_index.index = index

        if use_gpu:
            res = faiss.StandardGpuResources()
            faiss_index.index = faiss.index_cpu_to_gpu(res, 0, faiss_index.index)
            faiss_index.use_gpu = True

        return faiss_index
