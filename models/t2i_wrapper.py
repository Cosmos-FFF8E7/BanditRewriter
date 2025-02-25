#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text-to-Image model wrapper for BanditRewriter
"""

import os
import logging
import uuid
from io import BytesIO

import requests
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

logger = logging.getLogger(__name__)


class T2IWrapper:

    def __init__(self, model="sd21", config=None):
        self.config = config
        self.model_name = model
        self.model = None
        self.url = "https://api.openai.com"

        if config:
            if model == "dalle3" and config.get("dalle_key"):
                os.environ["OPENAI_API_KEY"] = config.get("dalle_key")

                self.url = config.get("openai_url", "https://api.openai.com")

        if model == "sd21" and torch.cuda.is_available():
            self._init_stable_diffusion()

    def _init_stable_diffusion(self):
        logger.info("Initializing Stable Diffusion 2.1 model")
        try:
            model_id = "stabilityai/stable-diffusion-2-1"
            self.model = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16
            )
            self.model.scheduler = DPMSolverMultistepScheduler.from_config(
                self.model.scheduler.config
            )
            self.model = self.model.to("cuda")
        except Exception as e:
            logger.error(f"Failed to initialize Stable Diffusion: {str(e)}")
            raise

    def generate(self, prompt, output_dir="outputs"):
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{uuid.uuid4()}.png"
        output_path = os.path.join(output_dir, filename)

        if self.model_name == "sd21" and self.model:
            image = self.model(prompt, guidance_scale=7.5).images[0]

            image.save(output_path)

        elif self.model_name == "dalle3":
            try:
                import openai

                response = openai.Image.create(
                    model="dall-e-3",
                    prompt=prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1,
                    api_base=self.url + "/v1/images/generations",
                )

                image_url = response.data[0].url
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content))
                image.save(output_path)

            except Exception as e:
                logger.error(f"DALL·E 3 API call failed: {str(e)}")
                raise

        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return output_path
