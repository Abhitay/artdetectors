from pathlib import Path
from typing import Union, Optional

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BlipCaptioner:
    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: str = DEVICE,
    ):
        self.device = device
        self.model_name = model_name

        self.processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model.eval()

    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        else:
            pil_img = image.convert("RGB")
        return pil_img

    def caption(
        self,
        image: Union[str, Path, Image.Image],
        prompt: Optional[str] = None,
        max_new_tokens: int = 30,
        num_beams: int = 3,
        do_sample: bool = False,
        **generate_kwargs,
    ) -> str:
        pil_img = self._load_image(image)

        if prompt is not None:
            inputs = self.processor(
                images=pil_img,
                text=prompt,
                return_tensors="pt",
            ).to(self.device)
        else:
            inputs = self.processor(
                images=pil_img,
                return_tensors="pt",
            ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                **generate_kwargs,
            )

        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption.strip()
