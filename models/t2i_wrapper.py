# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# """
# Text-to-Image model wrapper for BanditRewriter
# """

# import os
# import logging
# import uuid
# from io import BytesIO

# import requests
# import torch
# from PIL import Image
# from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# logger = logging.getLogger(__name__)


# class T2IWrapper:

#     def __init__(self, model="sd21", config=None):
#         self.config = config
#         self.model_name = model
#         self.model = None
#         self.url = "https://api.openai.com"

#         if config:
#             if model == "dalle3" and config.get("dalle_key"):
#                 os.environ["OPENAI_API_KEY"] = config.get("dalle_key")

#                 self.url = config.get("openai_url", "https://api.openai.com")

#         if model == "sd21" and torch.cuda.is_available():
#             self._init_stable_diffusion()

#     def _init_stable_diffusion(self):
#         logger.info("Initializing Stable Diffusion 2.1 model")
#         try:
#             model_id = "stabilityai/stable-diffusion-2-1"
#             self.model = StableDiffusionPipeline.from_pretrained(
#                 model_id, torch_dtype=torch.float16
#             )
#             self.model.scheduler = DPMSolverMultistepScheduler.from_config(
#                 self.model.scheduler.config
#             )
#             self.model = self.model.to("cuda")
#         except Exception as e:
#             logger.error(f"Failed to initialize Stable Diffusion: {str(e)}")
#             raise

#     def generate(self, prompt, output_dir="outputs"):
#         os.makedirs(output_dir, exist_ok=True)

#         filename = f"{uuid.uuid4()}.png"
#         output_path = os.path.join(output_dir, filename)

#         if self.model_name == "sd21" and self.model:
#             image = self.model(prompt, guidance_scale=7.5).images[0]

#             image.save(output_path)

#         elif self.model_name == "dalle3":
#             try:
#                 import openai

#                 response = openai.Image.create(
#                     model="dall-e-3",
#                     prompt=prompt,
#                     size="1024x1024",
#                     quality="standard",
#                     n=1,
#                     api_base=self.url + "/v1/images/generations",
#                 )

#                 image_url = response.data[0].url
#                 response = requests.get(image_url)
#                 image = Image.open(BytesIO(response.content))
#                 image.save(output_path)

#             except Exception as e:
#                 logger.error(f"DALL·E 3 API call failed: {str(e)}")
#                 raise

#         else:
#             raise ValueError(f"Unsupported model: {self.model_name}")

#         return output_path


import json
import os
import uuid
import torch
import logging
import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

logger = logging.getLogger(__name__)


class T2IWrapper:
    def __init__(self, model="sd21", config=None):
        self.config = config or {}  
        self.model_name = model
        self.model = None

        self.dalle_key = self.config.get("dalle_key")
        self.dalle_url = self.config.get("dalle_url", "https://api.openai.com")
        self.mj_key = self.config.get("mj_key")
        self.mj_url = self.config.get("mj_url", "https://api.midjourney.com")

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

    def _generate_sd21(self, prompt, output_path):
        image = self.model(prompt, guidance_scale=7.5).images[0]
        image.save(output_path)
        return output_path

    def _generate_dalle3(self, prompt, output_path):
        if not self.dalle_key:
            raise ValueError("DALL-E 3 API key not provided")

        try:
            payload = json.dumps(
                {"prompt": prompt, "n": 1, "model": "dall-e-3", "size": "1024x1024"}
            )
            headers = {
                "Authorization": f"Bearer {self.dalle_key}",
                "Content-Type": "application/json",
            }

            response = requests.post(
                f"{self.dalle_url}/v1/images/generations", headers=headers, data=payload
            )

            if response.status_code == 200:
                result = response.json()
                image_url = result["data"][0]["url"]
                image_response = requests.get(image_url)
                image = Image.open(BytesIO(image_response.content))
                image.save(output_path)
                return output_path
            else:
                raise Exception(
                    f"DALL-E API returned status code: {response.status_code}, Response: {response.text}"
                )
        except Exception as e:
            logger.error(f"DALL·E 3 API call failed: {str(e)}")
            raise

    def _generate_midjourney(self, prompt, output_path):
        if not self.mj_key:
            raise ValueError("Midjourney API key not provided")

        try:
            headers = {"TT-API-KEY": self.mj_key}

            data = {"prompt": prompt, "model": "fast", "hookUrl": "", "timeout": 300}

            response = requests.post(self.mj_url, headers=headers, json=data)

            if response.status_code == 200:
                result = response.json()
                if "imageUrl" in result:
                    image_url = result["imageUrl"]
                    image_response = requests.get(image_url)
                    image = Image.open(BytesIO(image_response.content))
                    image.save(output_path)
                    return output_path
                else:
                    raise Exception(
                        f"Midjourney API response missing imageUrl: {result}"
                    )
            else:
                raise Exception(
                    f"Midjourney API returned status code: {response.status_code}, Response: {response.text}"
                )
        except Exception as e:
            logger.error(f"Midjourney API call failed: {str(e)}")
            raise

    def generate(self, prompt, output_dir="outputs"):
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{uuid.uuid4()}.png"
        output_path = os.path.join(output_dir, filename)

        generators = {
            "sd21": self._generate_sd21,
            "dalle3": self._generate_dalle3,
            "midjourney": self._generate_midjourney,
        }

        if self.model_name not in generators:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return generators[self.model_name](prompt, output_path)
