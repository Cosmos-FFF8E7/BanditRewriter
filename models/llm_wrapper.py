#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Large Language Model wrapper for BanditRewriter
"""

import os
import logging
import time
from typing import List, Dict

import openai

logger = logging.getLogger(__name__)


class LLMWrapper:

    def __init__(self, model="gpt4", config=None):
        self.config = config
        self.model = model
        self.url = "https://api.openai.com"

        if config:
            openai.api_key = config.get(
                "openai_key", os.environ.get("OPENAI_API_KEY", "")
            )
            if config.get("openai_url") != None:
                self.url = config.get("openai_url")

        self.model_map = {
            "gpt4": "gpt-4",
            "gpt4o": "gpt-4o",
            "gpt3": "gpt-3.5-turbo",
            "gpt-4o-mini": "gpt-4o-mini",
        }

        self.model_id = self.model_map.get(model, "gpt-4")
        self.url = self.url = config.get(
            "openai_url", "https://api.openai.com"
        )

    def call_api(self, messages, temperature=0.7, max_tokens=1024):
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_base=self.url + "/v1/chat/completions",
                )
                return response.choices[0].message["content"]
            except Exception as e:
                logger.warning(
                    f"API call failed (attempt {attempt+1}/{max_retries}): {str(e)}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

    def distill_prompt(self, detailed_prompt):
        messages = [
            {
                "role": "system",
                "content": "You are an assistant that helps extract the core theme from detailed prompts.",
            },
            {
                "role": "user",
                "content": f"Extract the basic theme from this detailed prompt, only return the core theme:\n\n{detailed_prompt}",
            },
        ]

        return self.call_api(messages, temperature=0.3, max_tokens=256).strip()

    def synthesize_optimization_instructions(self, raw_prompt, detailed_prompt):
        messages = [
            {
                "role": "system",
                "content": "You are a T2I prompt engineering expert who can identify effective optimization strategies.",
            },
            {
                "role": "user",
                "content": f"""Analyze how the detailed prompt enhances the original theme and provide optimization strategies:
            
            Original theme: {raw_prompt}
            Detailed prompt: {detailed_prompt}
            
            Optimization instructions:""",
            },
        ]

        return self.call_api(messages, temperature=0.7, max_tokens=512).strip()

    def rewrite_prompt(self, raw_prompt, rewrite_template):
        messages = [
            {"role": "system", "content": "You are an expert in optimizing text-to-image generation prompts."},
            {
                "role": "user",
                "content": f"{rewrite_template}\n\nNow optimize this original prompt: {raw_prompt}\n\nOptimized prompt:",
            },
        ]

        return self.call_api(messages, temperature=0.7, max_tokens=1024).strip()

    def expand_prompt_variants(self, prompt, n=5):
        messages = [
            {"role": "system", "content": "You are a creative assistant capable of generating prompt variants."},
            {
                "role": "user",
                "content": f"Generate {n} variants of the following prompt by replacing words with synonyms:\n\n{prompt}",
            },
        ]

        response = self.call_api(messages, temperature=0.8, max_tokens=1024).strip()

        variants = [line.strip() for line in response.split("\n") if line.strip()]

        return variants[:n]
