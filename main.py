
import os
import argparse
import logging

from config.config import load_config
from data.dataset import load_dataset
from models.llm_wrapper import LLMWrapper
from models.t2i_wrapper import T2IWrapper
from optimize.bandit_optimize import BanditOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="BanditRewriter")

    parser.add_argument("--mode", type=str, required=True, choices=["train", "infer"])

    parser.add_argument("--dataset", type=str, help="Path to dataset (for training)")
    parser.add_argument("--prompt", type=str, help="Input prompt for inference")

    parser.add_argument(
        "--llm",
        type=str,
        default="gpt4",
        choices=["gpt4", "gpt3", "gpt4o", "gpt-4o-mini"],
    )
    parser.add_argument(
        "--t2i_model", type=str, required=True, choices=["sd21", "dalle3"]
    )

    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--num_iterations", type=int, default=5)
    parser.add_argument("--ucb_iterations", type=int, default=10)
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--num_clusters", type=int, default=50)

    parser.add_argument("--openai_key", type=str)
    parser.add_argument("--dalle_key", type=str)
    parser.add_argument(
        "--openai_url", type=str, default="https://api.openai.com"
    )

    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument(
        "--rewrite_prompt", type=str, help="Path to rewrite prompt (for inference)"
    )

    return parser.parse_args()


def train_mode(args, config, llm, t2i_model):
    logger.info("Loading dataset from: %s", args.dataset)
    dataset = load_dataset(args.dataset, config)

    logger.info("Initializing BanditOptimizer")
    optimizer = BanditOptimizer(
        llm=llm,
        t2i_model=t2i_model,
        alpha=args.alpha,
        num_iterations=args.num_iterations,
        ucb_iterations=args.ucb_iterations,
        beam_width=args.beam_width,
        num_clusters=args.num_clusters,
        config=config,
    )

    logger.info("Starting optimization process")
    optimal_rewrite_prompt = optimizer.optimize(dataset)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(optimal_rewrite_prompt)
        logger.info("Saved optimal rewrite prompt to: %s", args.output)

    return optimal_rewrite_prompt


def infer_mode(args, config, llm, t2i_model):
    if not args.prompt or not args.rewrite_prompt:
        raise ValueError("Prompt and rewrite_prompt are required for inference mode")

    with open(args.rewrite_prompt, "r", encoding="utf-8") as f:
        rewrite_prompt_template = f.read()

    raw_prompt = llm.distill_prompt(args.prompt)
    logger.info("Distilled raw prompt: %s", raw_prompt)

    optimized_prompt = llm.rewrite_prompt(raw_prompt, rewrite_prompt_template)
    logger.info("Optimized prompt: %s", optimized_prompt)

    image_path = t2i_model.generate(optimized_prompt)
    logger.info("Generated image saved to: %s", image_path)

    return optimized_prompt, image_path


def main():
    args = parse_args()
    config = load_config(args)

    if args.openai_key:
        os.environ["OPENAI_API_KEY"] = args.openai_key
    if args.dalle_key:
        os.environ["DALLE_API_KEY"] = args.dalle_key

    llm = LLMWrapper(model=args.llm, config=config)
    t2i_model = T2IWrapper(model=args.t2i_model, config=config)

    if args.mode == "train":
        train_mode(args, config, llm, t2i_model)
    elif args.mode == "infer":
        infer_mode(args, config, llm, t2i_model)


if __name__ == "__main__":
    main()
