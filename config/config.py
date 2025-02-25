
import os
import argparse
from typing import Dict, Any


class Config:

    def __init__(self, args=None):
        self.config = {}

        if args:
            self.config["openai_key"] = getattr(
                args, "openai_key", os.environ.get("OPENAI_API_KEY", "")
            )
            self.config["dalle_key"] = getattr(
                args, "dalle_key", os.environ.get("DALLE_API_KEY", "")
            )

            self.config["llm"] = getattr(args, "llm", "gpt4")
            self.config["t2i_model"] = getattr(args, "t2i_model", "sd21")

            self.config["alpha"] = getattr(args, "alpha", 0.65)
            self.config["num_iterations"] = getattr(args, "num_iterations", 5)
            self.config["ucb_iterations"] = getattr(args, "ucb_iterations", 10)
            self.config["beam_width"] = getattr(args, "beam_width", 5)
            self.config["num_clusters"] = getattr(args, "num_clusters", 50)
            self.config["num_candidates"] = getattr(args, "num_candidates", 20)
            self.config["batch_size"] = getattr(args, "batch_size", 128)

            self.config["lambda"] = getattr(args, "lambda", 0.6)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getitem__(self, key):
        return self.config.get(key)


def load_config(args):
    return Config(args)
