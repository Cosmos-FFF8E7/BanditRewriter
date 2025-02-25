import os
import json
import random
import logging
import numpy as np
from typing import List, Dict, Tuple, Any

from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Dataset:
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.sbert_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        self.expanded_data = []
        self.clusters = []

    def __len__(self):
        return len(self.data)

    def sample(self, n=1):
        return random.sample(self.data, min(n, len(self.data)))

    def expand_with_raw_prompts(self, llm):
        logger.info("Expanding dataset with raw prompts")
        self.expanded_data = []

        for item in self.data:
            detailed_prompt = item["prompt"]
            raw_prompt = llm.distill_prompt(detailed_prompt)

            self.expanded_data.append(
                {
                    "raw_prompt": raw_prompt,
                    "detailed_prompt": detailed_prompt,
                    "image_path": item.get("image_path", None),
                }
            )

        logger.info(f"Expanded dataset with {len(self.expanded_data)} items")
        return self.expanded_data

    def cluster_by_semantics(self, num_clusters):
        logger.info(f"Clustering dataset into {num_clusters} semantic clusters")

        raw_prompts = [item["raw_prompt"] for item in self.expanded_data]

        embeddings = self.sbert_model.encode(
            raw_prompts, convert_to_tensor=True
        ).numpy()

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        self.clusters = [[] for _ in range(num_clusters)]
        for idx, cluster_id in enumerate(clusters):
            self.clusters[cluster_id].append(idx)

        self.clusters = [c for c in self.clusters if len(c) > 0]

        logger.info(f"Created {len(self.clusters)} non-empty clusters")
        return self.clusters

    def select_diverse_raw_prompts(self, n=20):
        logger.info(f"Selecting {n} diverse raw prompts")

        raw_prompts = [item["raw_prompt"] for item in self.expanded_data]
        detailed_prompts = [item["detailed_prompt"] for item in self.expanded_data]

        embeddings = self.sbert_model.encode(
            raw_prompts, convert_to_tensor=True
        ).numpy()

        selected_indices = [random.randint(0, len(raw_prompts) - 1)]

        while len(selected_indices) < n:
            min_similarity = float("inf")
            next_idx = -1

            for i in range(len(raw_prompts)):
                if i in selected_indices:
                    continue

                avg_sim = 0
                for j in selected_indices:
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    avg_sim += sim

                avg_sim /= len(selected_indices)

                if avg_sim < min_similarity:
                    min_similarity = avg_sim
                    next_idx = i

            if next_idx != -1:
                selected_indices.append(next_idx)

        return [(raw_prompts[i], detailed_prompts[i]) for i in selected_indices]

    def generate_demonstration_set(self, clip_model, num_samples_per_cluster=5):
        logger.info(f"Generating demonstration set")

        if not self.clusters:
            raise ValueError(
                "Dataset must be clustered before generating demonstration set"
            )

        demonstration_set = []

        for cluster in self.clusters:
            cluster_samples = [self.expanded_data[idx] for idx in cluster]

            selected_samples = random.sample(
                cluster_samples, min(num_samples_per_cluster, len(cluster_samples))
            )

            for sample in selected_samples:
                demonstration_set.append(
                    (sample["raw_prompt"], sample["detailed_prompt"])
                )

        logger.info(
            f"Generated demonstration set with {len(demonstration_set)} examples"
        )
        return demonstration_set


def load_dataset(path, config):
    _, ext = os.path.splitext(path)

    data = []

    if ext.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif ext.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    return Dataset(data, config)
