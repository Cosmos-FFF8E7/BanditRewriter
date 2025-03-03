import glob
from io import BytesIO
import os
import json
import pickle
import random
import logging
import shutil
import tarfile
import numpy as np
from typing import List, Dict, Tuple, Any
import requests
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import PIL.Image as Image

# 引入huggingface的datasets库
from datasets import Dataset
from datasets import load_dataset as hf_load_dataset

logger = logging.getLogger(__name__)


class BRDataset:
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


def download_dalle3_data(
    shard_indices: List[int], save_dir: str = "downloaded_data"
) -> None:
    """Download DALL-E 3 synthetic dataset shards

    Args:
        shard_indices (List[int]): List of shard indices to download
        save_dir (str): Directory to save downloaded files
    """
    BASE_URL = "https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-high-quality-captions/resolve/main/data/data-{i:06d}.tar"

    # Create save directory if not exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in tqdm(shard_indices, desc="Downloading shards"):
        url = BASE_URL.format(i=i)
        filename = f"data-{i:06d}.tar"
        save_path = os.path.join(save_dir, filename)

        if os.path.exists(save_path):
            print(f"Shard {filename} already exists, skipping...")
            continue

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            print(f"Successfully downloaded shard {filename}")

        except Exception as e:
            print(f"Error downloading shard {filename}: {str(e)}")


def load_dataset(path, config):
    dataset = config.get("dataset")

    if not dataset:
        raise ValueError("Dataset name must be specified in the config")

    # 针对不同的数据集进行不同的处理
    if dataset == "dalle3":
        processed_data = process_dalle3_data()
    elif dataset == "diffusiondb":
        processed_data = process_diffusiondb_data()
    elif dataset == "lexica":
        processed_data = process_lexica_data()
    elif dataset == "bittensor":
        processed_data = process_bittensor_data()
    else:
        raise ValueError("Unsupported dataset type")

    return processed_data


def process_dalle3_data(
    shard_indices: List[int], save_dir: str = "downloaded_data"
) -> Dataset:
    os.makedirs(save_dir, exist_ok=True)
    dataset_path = os.path.join(save_dir, "dalle3_dataset")

    # 如果数据集已存在，直接加载
    if os.path.exists(dataset_path):
        dataset = Dataset.load_from_disk(dataset_path)
        print(f"Loaded dataset with {len(dataset)} samples")
        return dataset

    # 检查每个分片是否已下载，如果没有则下载
    for shard_index in shard_indices:
        shard_path = os.path.join(save_dir, f"data-{shard_index:06d}.tar")
        if not os.path.exists(shard_path):
            print(f"Shard {shard_index} not found, downloading...")
            download_dalle3_data(shard_indices, save_dir)

    # 首次处理
    all_data = {"prompt": [], "image": []}

    for shard_index in tqdm(shard_indices, desc="Processing DALL-E 3 dataset"):
        shard_path = os.path.join(save_dir, f"data-{shard_index:06d}.tar")
        extract_dir = os.path.join(save_dir, f"data-{shard_index:06d}")

        try:
            os.makedirs(extract_dir, exist_ok=True)
            print(f"Processing shard {shard_index} from {shard_path}")

            # 解压整个tar文件
            with tarfile.open(shard_path, "r") as tar:
                tar.extractall(extract_dir)

            # 处理所有jpg文件
            jpg_files = glob.glob(os.path.join(extract_dir, "*.jpg"))
            print(f"Found {len(jpg_files)} jpg files in shard {shard_index}")

            # 随机选择3000个文件
            if len(jpg_files) > 3000:
                import random
                jpg_files = random.sample(jpg_files, 3000)

            for jpg_file in tqdm(jpg_files):
                try:
                    json_file = jpg_file.replace(".jpg", ".json")

                    # 读取JSON数据
                    with open(json_file, "r", encoding="utf-8") as f:
                        json_data = json.load(f)

                    # 验证文件是否存在
                    if not os.path.exists(jpg_file):
                        print(f"Warning: {jpg_file} does not exist")
                        continue

                    # 直接加载图片为PIL Image对象
                    with Image.open(jpg_file) as img:
                        image = img.convert("RGB")

                    all_data["prompt"].append(json_data["long_caption"])
                    all_data["image"].append(image)

                    # 如果已经收集了3000个样本，就停止
                    if len(all_data["prompt"]) >= 3000:
                        break

                except Exception as e:
                    logger.error(f"Error processing {jpg_file}: {str(e)}")
                    continue

            # 如果已经收集了3000个样本，就停止处理其他分片
            if len(all_data["prompt"]) >= 3000:
                break

        except Exception as e:
            logger.error(f"Error processing shard {shard_index}: {str(e)}")
            continue

    print(f"Total samples collected: {len(all_data['prompt'])}")

    # 创建并保存数据集
    print("Creating dataset...")
    dataset = Dataset.from_dict(all_data)
    print("Saving dataset...")
    dataset.save_to_disk(dataset_path)
    print(f"Saved dataset with {len(dataset)} samples")

    return dataset


def process_diffusiondb_data(subset="large_random_1k"):
    dataset = load_dataset("poloclub/diffusiondb", subset)
    """
    {'image': <PIL.WebPImagePlugin.WebPImageFile image mode=RGB size=512x512 at 0x2053495C490>, 'prompt': 'nasi goreng, realistic, sharp focus, 8 k high definition, insanely detailed, intricate, elegant, food photography ', 'seed': 2834991788, 'step': 50, 'cfg': 7.0, 'sampler': 'k_lms', 'width': 512, 'height': 512, 'user_name': '0f0c127a510a5c6b9432bf90908973778ad7889628984a092d8de4ee2d5aae37', 'timestamp': datetime.datetime(2022, 8, 15, 8, 13, tzinfo=<UTC>), 'image_nsfw': 0.023086752742528915, 'prompt_nsfw': 0.0018341769464313984}
    """
    # 处理成{"prompt": [], "image": []}格式
    all_data = {"prompt": [], "image": []}
    for item in dataset:
        all_data["prompt"].append(item["prompt"])
        all_data["image"].append(item["image"])

    # 创建并保存数据集
    dataset = Dataset.from_dict(all_data)
    dataset.save_to_disk("diffusiondb_dataset")

    return dataset


def process_lexica_data():
    ds = load_dataset("Gustavosta/Stable-Diffusion-Prompts", "train")
    # 这个数据集只有prompt，处理成{"prompt": [], "image": []}格式，image留空
    all_data = {"prompt": [], "image": []}
    for item in ds:
        all_data["prompt"].append(item["prompt"])
        all_data["image"].append(None)

    # 创建并保存数据集
    dataset = Dataset.from_dict(all_data)
    dataset.save_to_disk("lexica_dataset")

    print(f"Saved dataset with {len(dataset)} samples")
    return dataset


def process_bittensor_data():
    ds = load_dataset("CortexLM/midjourney-v6")

    # 取前2000个
    ds = ds.select(range(2000))

    # 这个数据集是prompt、image_url格式
    all_data = {"prompt": [], "image": []}
    for item in ds:
        all_data["prompt"].append(item["prompt"])
        # 发送请求下载图片
        try:
            response = requests.get(item["image_url"])
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            all_data["image"].append(img)
        except Exception as e:
            print(f"Error downloading image: {e}")
            all_data["image"].append(None)

    # 由于是midjourny图片，所以需要裁剪画面左上的四分之一保留
    for i in range(len(all_data["image"])):
        if all_data["image"][i] is not None:
            all_data["image"][i] = all_data["image"][i].crop(
                (
                    0,
                    0,
                    all_data["image"][i].width // 2,
                    all_data["image"][i].height // 2,
                )
            )
            all_data["image"][i] = all_data["image"][i].resize((512, 512))

    # 创建并保存数据集
    dataset = Dataset.from_dict(all_data)
    dataset.save_to_disk("bittensor_dataset")

    print(f"Saved dataset with {len(dataset)} samples")
    return dataset


# 当需要实际加载图片时，可以添加这个函数
def load_image(image_path):
    with Image.open(image_path) as img:
        return img.convert("RGB")
