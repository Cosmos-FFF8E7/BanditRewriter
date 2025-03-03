
import logging
import math
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

from .beam_search import beam_search
from evaluate.metrics import CLIPScorer, AestheticScorer
from data.br_dataset import BRDataset
from models.llm_wrapper import LLMWrapper
from models.t2i_wrapper import T2IWrapper

logger = logging.getLogger(__name__)


class BanditOptimizer:

    def __init__(
        self,
        llm: LLMWrapper,
        t2i_model: T2IWrapper,
        alpha: float = 0.65,
        num_iterations: int = 5,
        ucb_iterations: int = 10,
        beam_width: int = 5,
        num_clusters: int = 50,
        config: Optional[Any] = None,
    ) -> None:
        self.llm = llm
        self.t2i_model = t2i_model
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.ucb_iterations = ucb_iterations
        self.beam_width = beam_width
        self.num_clusters = num_clusters
        self.config = config

        self.clip_scorer = CLIPScorer()
        self.aesthetic_scorer = AestheticScorer()

        self.lambda_weight = config.get("lambda", 0.6) if config else 0.6

    def score_prompt(self, prompt: str, raw_prompt: str) -> float:

        optimized_prompt = self.llm.rewrite_prompt(raw_prompt, prompt)
        image_path = self.t2i_model.generate(optimized_prompt)
        clip_score = self.clip_scorer.score(optimized_prompt, image_path)
        aesthetic_score = self.aesthetic_scorer.score(image_path)
        score = self.lambda_weight * aesthetic_score + (1 - self.lambda_weight) * clip_score

        # score = np.random.random()
        return score

    def ucb_score(self, mean: float, n: int, t: int) -> float:
        if n == 0:
            return float("inf")

        return mean + self.alpha * math.sqrt(2 * math.log(t) / n)
    def expand_and_replace(
        self,
        selected_prompts: List[str],
        demonstration_set: List[Tuple[str, str]],
        rewrite_instructions: List[str],
        m: int,
    ) -> List[str]:

        n = len(selected_prompts)
        new_candidates = []

        # 1. Keep all the prompts selected by beam search
        new_candidates.extend(selected_prompts)

        # 2. Generate a variant for each selected prompt
        for prompt in selected_prompts:
            variants = self.llm.expand_prompt_variants(prompt, 1)
            if variants:
                new_candidates.extend(variants)

        # 3. Create new random combinations for the remaining candidate slots
        remaining_slots = m - len(new_candidates)

        if remaining_slots > 0:
            for _ in range(remaining_slots):
                # Randomly select a prompt from the current candidates as the base
                base_prompt = np.random.choice(new_candidates)

                # 50% chance to replace demonstration, 50% chance to replace instruction
                if np.random.rand() < 0.5:
                    # Replace demonstration
                    parts = base_prompt.split("Examples:")
                    if len(parts) > 1:
                        instruction = parts[0].strip()

                        # Randomly select a new demonstration from the set
                        raw_prompt, detailed_prompt = demonstration_set[
                            np.random.randint(len(demonstration_set))
                        ]
                        new_demo = f"Raw: {raw_prompt}\nDetailed: {detailed_prompt}"

                        # Create new prompt
                        new_prompt = f"{instruction}\n\nExamples:\n{new_demo}"
                        new_candidates.append(new_prompt)
                else:
                    # Replace instruction
                    parts = base_prompt.split("Examples:")
                    if len(parts) > 1:
                        demos = parts[1].strip()

                        # Randomly select a new instruction
                        new_instruction = rewrite_instructions[
                            np.random.randint(len(rewrite_instructions))
                        ]

                        # Create new prompt
                        new_prompt = f"{new_instruction}\n\nExamples:\n{demos}"
                        new_candidates.append(new_prompt)

        # Ensure we have exactly m candidates
        if len(new_candidates) > m:
            new_candidates = new_candidates[:m]

        return new_candidates

    def optimize(self, dataset: BRDataset) -> str:
        logger.info("Starting optimization process")

        # Step 1: Expand the dataset with raw prompts
        expanded_data = dataset.expand_with_raw_prompts(self.llm)

        # Step 2: Select diverse raw prompts
        diverse_pairs = dataset.select_diverse_raw_prompts(n=20)

        # Step 3: Synthesize optimization instructions
        rewrite_instructions: List[str] = []
        for raw_prompt, detailed_prompt in diverse_pairs:
            instruction = self.llm.synthesize_optimization_instructions(
                raw_prompt, detailed_prompt
            )
            rewrite_instructions.append(instruction)

        # Step 4: Cluster the dataset by semantics
        clusters = dataset.cluster_by_semantics(num_clusters=self.num_clusters)

        # Step 5: Generate demonstration set
        demonstration_set = dataset.generate_demonstration_set(
            clip_model=None, num_samples_per_cluster=5  # Simplified version without using CLIP scorer
        )

        # Step 6: Create initial candidate set
        m = len(rewrite_instructions)  # Number of candidates
        candidate_prompts: List[str] = []
        for instruction in rewrite_instructions:
            # Randomly select demonstrations
            selected_demos = []
            for _ in range(3):  # Use 3 examples for each instruction
                demo = demonstration_set[np.random.randint(len(demonstration_set))]
                raw, detailed = demo
                selected_demos.append(f"Raw: {raw}\nDetailed: {detailed}")

            # Create rewrite prompt
            rewrite_prompt = f"{instruction}\n\nExamples:\n" + "\n\n".join(
                selected_demos
            )
            candidate_prompts.append(rewrite_prompt)

        # Initialize UCB tracking
        means: Dict[str, float] = {prompt: 0.0 for prompt in candidate_prompts}
        counts: Dict[str, int] = {prompt: 0 for prompt in candidate_prompts}

        # Step 7: Outer iteration optimization - N iterations of beam search
        for iteration in range(self.num_iterations):
            logger.info(f"Starting iteration {iteration+1}/{self.num_iterations}")

            # Update UCB scores in each iteration
            for t in range(1, self.ucb_iterations + 1):
                # Create evaluation subset
                eval_sample = dataset.sample(min(128, len(dataset)))

                # Calculate UCB scores for each candidate
                ucb_scores: Dict[str, float] = {}
                for prompt in candidate_prompts:
                    if counts[prompt] == 0:
                        ucb_scores[prompt] = float("inf")
                    else:
                        ucb_score = self.ucb_score(means[prompt], counts[prompt], t)
                        ucb_scores[prompt] = ucb_score

                # Select prompts to evaluate based on UCB scores
                def ucb_scoring_function(prompt: str) -> float:
                    return ucb_scores[prompt]

                # Use beam_search to select top prompts
                selected_prompts = beam_search(
                    candidate_prompts, ucb_scoring_function, self.beam_width
                )

                # Evaluate selected prompts
                for prompt in selected_prompts:
                    # Score random raw prompt
                    sample = eval_sample[np.random.randint(len(eval_sample))]
                    raw_prompt = sample.get(
                        "raw_prompt", self.llm.distill_prompt(sample["prompt"])
                    )

                    score = self.score_prompt(prompt, raw_prompt)

                    # Update statistics
                    counts[prompt] += 1
                    means[prompt] += (score - means[prompt]) / counts[prompt]

                    logger.debug(
                        f"Prompt score: {score:.4f}, Mean: {means[prompt]:.4f}, Count: {counts[prompt]}"
                    )

            # If not the last iteration, use beam search to select top prompts and generate new candidates
            if iteration < self.num_iterations - 1:
                # Use mean score for beam search
                def mean_scoring_function(prompt: str) -> float:
                    return means[prompt]

                selected_prompts = beam_search(
                    candidate_prompts, mean_scoring_function, self.beam_width
                )

                # Generate new candidate set for the next iteration
                candidate_prompts = self.expand_and_replace(
                    selected_prompts, demonstration_set, rewrite_instructions, m
                )

        # Step 8: Return the best rewrite prompt
        def final_scoring_function(prompt: str) -> float:
            return means[prompt]

        best_prompts = beam_search(candidate_prompts, final_scoring_function, 1)
        best_prompt = best_prompts[0]  # Get the one with the highest score

        logger.info(f"Optimization complete. Best mean score: {means[best_prompt]:.4f}")
        return best_prompt
