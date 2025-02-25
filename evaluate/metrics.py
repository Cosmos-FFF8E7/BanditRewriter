
import logging
import torch
import clip
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

logger = logging.getLogger(__name__)


class CLIPScorer:

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Failed to initialize CLIP model: {str(e)}")
            raise

    def score(self, text, image_path):
        try:
            image = Image.open(image_path).convert("RGB")

            inputs = self.processor(
                text=text, images=image, return_tensors="pt", padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

                image_embeds = outputs.image_embeds / outputs.image_embeds.norm(
                    dim=-1, keepdim=True
                )
                text_embeds = outputs.text_embeds / outputs.text_embeds.norm(
                    dim=-1, keepdim=True
                )

                similarity = torch.matmul(text_embeds, image_embeds.T).item()

            return similarity
        except Exception as e:
            logger.error(f"CLIP scoring failed: {str(e)}")
            return 0.0


class AestheticScorer:
    def __init__(self):
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        except Exception as e:
            logger.error(f"Failed to initialize Aesthetic scorer: {str(e)}")
            raise

    def score(self, image_path: str) -> float:
        from pathlib import Path

        import torch
        from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
        from PIL import Image

        SAMPLE_IMAGE_PATH = Path(image_path)

        model, preprocessor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model = model.to(torch.bfloat16).cuda()

        image = Image.open(SAMPLE_IMAGE_PATH).convert("RGB")

        pixel_values = (
            preprocessor(images=image, return_tensors="pt")
            .pixel_values.to(torch.bfloat16)
            .cuda()
        )

        with torch.inference_mode():
            score = model(pixel_values).logits.squeeze().float().cpu().numpy()

        # print(f"Aesthetics score: {score:.4f}")

        return {"image_path": image_path, "score": score}


def evaluate_prompt(original_prompt, optimized_prompt, image_path, config=None):
    results = {}

    clip_scorer = CLIPScorer()
    aes_scorer = AestheticScorer()

    results["clip_score"] = clip_scorer.score(optimized_prompt, image_path)
    results["aesthetic_score"] = aes_scorer.score(image_path)

    return results
