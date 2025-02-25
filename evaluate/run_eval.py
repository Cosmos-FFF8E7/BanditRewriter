
import os
import json
import argparse
import logging
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np

from config.config import Config
from data.dataset import load_dataset, Dataset
from models.llm_wrapper import LLMWrapper
from models.t2i_wrapper import T2IWrapper
from metrics import evaluate_prompt, CLIPScorer, AestheticScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BanditRewriter")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["sd21", "dalle3"],
        help="Text-to-Image model to evaluate",
    )

    parser.add_argument(
        "--test_data", type=str, required=True, help="Path to test data file"
    )

    parser.add_argument(
        "--rewrite_prompt",
        type=str,
        required=True,
        help="Path to optimal rewrite prompt file",
    )

    parser.add_argument(
        "--metrics",
        type=str,
        default="clip,aes",
        help="Comma-separated list of metrics to compute",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples to evaluate"
    )

    parser.add_argument("--openai_key", type=str, help="OpenAI API key")
    parser.add_argument("--dalle_key", type=str, help="DALL·E API key")

    parser.add_argument(
        "--baselines",
        type=str,
        help="Comma-separated list of baseline methods to compare",
    )
    parser.add_argument(
        "--baseline_files",
        type=str,
        help="Comma-separated list of baseline rewrite prompt files",
    )

    return parser.parse_args()


def load_rewrite_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def evaluate_test_set(
    test_data: Dataset,
    rewrite_prompt: str,
    llm: LLMWrapper,
    t2i_model: T2IWrapper,
    metrics: List[str],
    output_dir: str,
    num_samples: int = 100,
) -> Dict[str, float]:
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    clip_scorer = CLIPScorer() if "clip" in metrics else None
    aesthetic_scorer = AestheticScorer() if "aes" in metrics else None

    results = []
    avg_metrics: Dict[str, float] = {metric: 0.0 for metric in metrics}
    count = 0

    data_sample = test_data.sample(min(num_samples, len(test_data)))

    for idx, item in enumerate(data_sample):
        original_prompt = item["prompt"]

        raw_prompt = llm.distill_prompt(original_prompt)

        optimized_prompt = llm.rewrite_prompt(raw_prompt, rewrite_prompt)

        image_path = t2i_model.generate(optimized_prompt, output_dir=images_dir)

        metrics_dict: Dict[str, float] = {}

        if clip_scorer:
            clip_score = clip_scorer.score(optimized_prompt, image_path)
            metrics_dict["clip_score"] = clip_score

        if aesthetic_scorer:
            aes_score = aesthetic_scorer.score(image_path)
            metrics_dict["aesthetic_score"] = aes_score

        result = {
            "id": idx,
            "original_prompt": original_prompt,
            "raw_prompt": raw_prompt,
            "optimized_prompt": optimized_prompt,
            "image_path": image_path,
            **metrics_dict,
        }
        results.append(result)

        for metric in metrics:
            metric_key = f"{metric}_score" if not metric.endswith("_score") else metric
            if metric_key in metrics_dict:
                avg_metrics[metric] += metrics_dict[metric_key]

        count += 1

        if idx % 10 == 0:
            logger.info(f"Evaluated {idx+1}/{len(data_sample)} samples")

    for metric in avg_metrics:
        avg_metrics[metric] /= count if count > 0 else 1

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "results.csv"), index=False)

    with open(os.path.join(output_dir, "avg_metrics.json"), "w") as f:
        json.dump(avg_metrics, f, indent=4)

    logger.info(f"Average metrics: {avg_metrics}")

    return avg_metrics


def main() -> None:
    args = parse_args()

    config = Config()
    config.config = {
        "openai_key": args.openai_key,
        "dalle_key": args.dalle_key,
        "t2i_model": args.model,
        "metrics": args.metrics,
    }

    if args.openai_key:
        os.environ["OPENAI_API_KEY"] = args.openai_key
    if args.dalle_key:
        os.environ["DALLE_API_KEY"] = args.dalle_key

    logger.info(f"Loading test data from: {args.test_data}")
    test_data = load_dataset(args.test_data, config)

    llm = LLMWrapper(model="gpt4", config=config)

    t2i_model = T2IWrapper(model=args.model, config=config)

    metrics = args.metrics.split(",")

    logger.info(f"Loading rewrite prompt from: {args.rewrite_prompt}")
    rewrite_prompt = load_rewrite_prompt(args.rewrite_prompt)

    output_dir = os.path.join(args.output_dir, "banditrewriter")

    logger.info("Evaluating BanditRewriter on test set")
    br_metrics = evaluate_test_set(
        test_data=test_data,
        rewrite_prompt=rewrite_prompt,
        llm=llm,
        t2i_model=t2i_model,
        metrics=metrics,
        output_dir=output_dir,
        num_samples=args.num_samples,
    )

    if args.baselines and args.baseline_files:
        baselines = args.baselines.split(",")
        baseline_files = args.baseline_files.split(",")

        if len(baselines) != len(baseline_files):
            logger.error("Number of baselines and baseline files must match")
        else:
            baseline_metrics: Dict[str, Dict[str, float]] = {}

            for baseline, baseline_file in zip(baselines, baseline_files):
                logger.info(f"Evaluating baseline: {baseline}")

                baseline_prompt = load_rewrite_prompt(baseline_file)

                baseline_output_dir = os.path.join(args.output_dir, baseline)

                baseline_metrics[baseline] = evaluate_test_set(
                    test_data=test_data,
                    rewrite_prompt=baseline_prompt,
                    llm=llm,
                    t2i_model=t2i_model,
                    metrics=metrics,
                    output_dir=baseline_output_dir,
                    num_samples=args.num_samples,
                )

            comparison: Dict[str, Dict[str, float]] = {
                "BanditRewriter": br_metrics,
                **baseline_metrics,
            }

            with open(os.path.join(args.output_dir, "comparison.json"), "w") as f:
                json.dump(comparison, f, indent=4)

            comparison_table: Dict[str, List[float]] = {}
            for metric in metrics:
                comparison_table[metric] = []
                for method, metrics_dict in comparison.items():
                    comparison_table[metric].append(metrics_dict.get(metric, 0.0))

            methods = ["BanditRewriter"] + baselines
            comparison_df = pd.DataFrame(comparison_table, index=methods)

            comparison_df.to_csv(os.path.join(args.output_dir, "comparison.csv"))

            logger.info("Evaluation Results Comparison:")
            logger.info("\n" + str(comparison_df))

            improvements: Dict[str, Dict[str, float]] = {}
            for metric in metrics:
                improvements[metric] = {}
                br_value = comparison["BanditRewriter"].get(metric, 0.0)
                for baseline in baselines:
                    baseline_value = comparison[baseline].get(metric, 0.0)
                    if baseline_value > 0:
                        rel_improvement = (
                            (br_value - baseline_value) / baseline_value * 100
                        )
                        improvements[metric][baseline] = rel_improvement

            with open(os.path.join(args.output_dir, "improvements.json"), "w") as f:
                json.dump(improvements, f, indent=4)

            logger.info("Relative improvements (%):")
            for metric, baseline_improvements in improvements.items():
                logger.info(f"{metric}:")
                for baseline, imp in baseline_improvements.items():
                    logger.info(f"  vs {baseline}: {imp:.2f}%")


if __name__ == "__main__":
    main()
