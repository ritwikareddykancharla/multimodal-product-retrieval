"""
Dataset Utilities for Multimodal Product Retrieval

Responsibilities:
- Load product images and text descriptions
- Maintain consistent product IDs across modalities
- Provide iterable access for embedding pipelines

Expected data format:
A CSV file with columns:
- product_id
- image_path
- text

Images are loaded lazily from disk.
"""

from typing import List, Dict, Tuple
import csv
import os
from PIL import Image


class ProductDataset:
    def __init__(
        self,
        csv_path: str,
        image_root: str,
    ):
        """
        Args:
            csv_path: path to CSV file containing product metadata
            image_root: root directory containing product images
        """
        self.csv_path = csv_path
        self.image_root = image_root
        self.items = self._load_metadata()

    def _load_metadata(self) -> List[Dict]:
        items = []
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                items.append(
                    {
                        "product_id": row["product_id"],
                        "image_path": row["image_path"],
                        "text": row["text"],
                    }
                )
        return items

    def __len__(self) -> int:
        return len(self.items)

    def get_images(self) -> List[Image.Image]:
        """
        Load all product images as PIL Images.
        """
        images = []
        for item in self.items:
            path = os.path.join(self.image_root, item["image_path"])
            img = Image.open(path).convert("RGB")
            images.append(img)
        return images

    def get_texts(self) -> List[str]:
        """
        Return all product text descriptions.
        """
        return [item["text"] for item in self.items]

    def get_product_ids(self) -> List[str]:
        """
        Return product IDs in dataset order.
        """
        return [item["product_id"] for item in self.items]

    def __getitem__(self, idx: int) -> Tuple[str, Image.Image, str]:
        """
        Return a single product sample.

        Returns:
            (product_id, image, text)
        """
        item = self.items[idx]
        image_path = os.path.join(self.image_root, item["image_path"])
        image = Image.open(image_path).convert("RGB")
        return item["product_id"], image, item["text"]
