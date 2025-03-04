#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Large Language Model wrapper for BanditRewriter
"""

import os
import logging
import textwrap
import time
from typing import List, Dict

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMWrapper:
    def __init__(self, model="gpt4", config=None):
        self.config = config
        self.model = model
        self.url = "https://api.openai.com"

        api_key = None
        if config:
            api_key = config.get("openai_key", os.environ.get("OPENAI_API_KEY", ""))
            if config.get("openai_url"):
                self.url = config.get("openai_url")

        self.client = OpenAI(api_key=api_key, base_url=self.url)

        self.model_map = {
            "gpt4": "gpt-4",
            "gpt4o": "gpt-4o",
            "gpt3": "gpt-3.5-turbo",
            "gpt-4o-mini": "gpt-4o-mini",
        }

        self.model_id = self.model_map.get(model, "gpt-4")

    def call_api(self, messages, temperature=0.7, max_tokens=1024):
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content

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
                "content": textwrap.dedent(
                    """
                    Instruction: Summarize this image description in 10 words or less and ignore words like archdaily, wallpaper, highly detailed, 8k. Ignore modifiers likes 'portrait of', 'by somebody', 'with xxx' or 'in xxx'. Ignore adjective. Check English.

                    Input: a beautiful very detailed illustration of abandoned urbex unfinished building city nature industrial architecture architecture building spaceport by caspar david friedrich, scumm bar meadow nature synthwave, archdaily, wallpaper, highly detailed, trending on artstation.
                    Output: <RES>abandoned urban building</RES>

                    Input: realistic detailed face portrait of Angelina Jolie as Salome by Alphonse Mucha, Ayami Kojima, Amano, Charlie Bowater, Karol Bak, Greg Hildebrandt, Jean Delville, and Mark Brooks, Art Nouveau, Neo-Gothic, Surreality, gothic, rich deep moody colors
                    Output: <RES>Angelina Jolie</RES>

                    Input: A stunning photograph of a majestic snow-capped mountain peak at sunset, with dramatic clouds and golden light, captured with a high-end DSLR camera, ultra HD quality, professional photography
                    Output: <RES>mountain peak at sunset</RES>

                    Input: An intricate steampunk-inspired mechanical pocket watch with brass gears, ornate Victorian decorations, and glowing crystal elements, created in the style of vintage technical illustrations
                    Output: <RES>steampunk mechanical pocket watch</RES>
                    """
                ),
            },
            {
                "role": "user",
                "content": f"Input: {detailed_prompt}",
            },
        ]

        response = self.call_api(messages, temperature=0.3, max_tokens=256).strip()

        # Extract content between RES tags
        import re

        match = re.search(r"<RES>(.*?)</RES>", response)
        if match:
            return match.group(1).strip()
        return response.strip()

    def synthesize_optimization_instructions(self, raw_prompt, detailed_prompt) -> str:
        messages = [
            {
                "role": "system",
                "content": textwrap.dedent(
                    """
                    You are a T2I prompt engineering expert who can identify effective optimization strategies.
                    Please analyze the prompts and provide optimization suggestions.
                    Wrap your suggestions with <RES></RES> tags.
                    
                    Example:
                    Raw: A cat
                    Detailed: A fluffy orange tabby cat sitting on windowsill
                    Optimization suggestions: <RES>Add descriptive details about appearance, setting and lighting. Use specific breed names. Include artistic style references.</RES>
                    """
                ),
            },
            {
                "role": "user",
                "content": f"""Analyze how the detailed prompt enhances the original theme and provide optimization strategies:
                
                Raw: {raw_prompt}
                Detailed: {detailed_prompt}
                
                Optimization suggestions:""",
            },
        ]

        response = self.call_api(messages, temperature=0.7, max_tokens=512).strip()

        # Extract content between RES tags
        import re

        match = re.search(r"<RES>(.*?)</RES>", response)
        if match:
            return match.group(1).strip()
        return response.strip()

    def rewrite_prompt(self, raw_prompt, rewrite_template) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are an expert in optimizing text-to-image generation prompts. Please wrap your optimized prompt with <RES></RES> tags.",
            },
            {
                "role": "user",
                "content": f"{rewrite_template}\n\nNow optimize this original prompt: {raw_prompt}\n\nOptimized prompt:",
            },
        ]

        response = self.call_api(messages, temperature=0.7, max_tokens=1024).strip()

        # Extract content between RES tags
        import re

        match = re.search(r"<RES>(.*?)</RES>", response)
        if match:
            return match.group(1).strip()
        return response.strip()

    def expand_prompt_variants(self, prompt, n=5) -> List[str]:
        messages = [
            {
                "role": "system",
                "content": textwrap.dedent(
                    """
                    You are a creative assistant specialized in linguistic variations.
                    Your task is to create prompt variants by:
                    - Using synonyms and alternative phrasings 
                    - Maintaining the exact same meaning
                    - Focusing on vocabulary and grammar changes only
                    - Keeping the same level of detail as original
                    - Wrap each variant with <RES></RES> tags
                """
                ).strip(),
            },
            {
                "role": "user",
                "content": textwrap.dedent(
                    f"""
                    Generate {n} variants of this prompt by only modifying vocabulary and grammar:

                    Original: {prompt}

                    Rules:
                    - Use synonyms and alternative word choices
                    - Keep the exact same meaning
                    - Don't add new details or style elements
                    - Focus only on linguistic variations
                    - Wrap each variant with <RES></RES> tags
                    
                    Example format:
                    <RES>First variant here</RES>
                    <RES>Second variant here</RES>
                    
                    Variants:
                """
                ).strip(),
            },
        ]

        response = self.call_api(messages, temperature=0.8, max_tokens=1024).strip()

        import re

        variants = re.findall(r"<RES>(.*?)</RES>", response)

        return variants[:n]
