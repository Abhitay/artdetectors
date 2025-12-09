# src/artdetectors/restyle.py

from pathlib import Path
from typing import Union, Optional

import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline


def _get_default_device() -> str:
    """
    Prefer MPS on Apple Silicon, then CUDA, then CPU.
    """
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
            return "cuda"
    return "cpu"


class ImageRestyler:
    """
    Image-to-image restyling using Stable Diffusion 1.5 (fast).

    This class takes an input image and a text description of a target style
    (e.g. "ukiyo-e", "cubism") and generates a new image with the same
    underlying content but in the new style.
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[str] = None,
    ):
        self.device = device or _get_default_device()

        # On MPS: use full float32 (fp16 can be flaky / NaN-y)
        if self.device == "mps":
            torch_dtype = torch.float32
        elif self.device == "cuda":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        )

        self.pipe.to(self.device)

        # Performance tweaks (safe on M1)
        self.pipe.enable_attention_slicing()

        # Disable safety checker (can return black images)
        if hasattr(self.pipe, "safety_checker") and self.pipe.safety_checker is not None:
            def dummy_safety_checker(images, **kwargs):
                batch_size = len(images)
                return images, [False] * batch_size

            self.pipe.safety_checker = dummy_safety_checker

    @staticmethod
    def _to_pil(image: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        return image.convert("RGB")

    def restyle(
        self,
        image: Union[str, Path, Image.Image],
        base_prompt: str,
        target_style: str,
        negative_prompt: Optional[str] = None,
        strength: float = 0.3,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 20,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Recreate the same image in a new style using img2img.

        Parameters
        ----------
        image:
            Original image or path to it.
        base_prompt:
            Semantic description of the scene (e.g. BLIP caption, possibly edited).
        target_style:
            Style description, e.g. "traditional Japanese ukiyo-e woodblock print".
        negative_prompt:
            Optional negative prompt to suppress artifacts.
        strength:
            How far to move away from the original image (0–1).
            Lower = closer to original composition.
        guidance_scale:
            Classifier-free guidance scale. Typical: 3–7 for img2img.
        num_inference_steps:
            Diffusion steps. Lower = faster, higher = more detailed.
        seed:
            Optional random seed for reproducibility.

        Returns
        -------
        PIL.Image.Image
            Restyled image.
        """
        init_image = self._to_pil(image)

        # SD 1.5 works best at 512x512; also keeps shapes simple
        init_image = init_image.resize((512, 512), Image.LANCZOS)

        target_style = target_style.strip()
        if target_style:
            prompt = f"{base_prompt}, in the style of {target_style}"
        else:
            prompt = base_prompt

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        out = self.pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        # out.images is already a list of PIL images
        return out.images[0]
