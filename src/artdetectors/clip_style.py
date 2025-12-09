import re
from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict

import torch
import clip
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _extract_style_name(prompt: str) -> str:
    m = re.search(r"style of (.+?) by the artist", prompt, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(" .")
    return prompt.strip()


class ClipStylePredictor:
    def __init__(
        self,
        style_txt_path: Union[str, Path],
        device: str = DEVICE,
        clip_model: Optional[torch.nn.Module] = None,
        clip_preprocess=None,
    ):
        self.device = device
        self.style_txt_path = Path(style_txt_path)

        self.prompts: List[str] = self._load_style_prompts(self.style_txt_path)
        self.style_names: List[str] = [_extract_style_name(p) for p in self.prompts]

        if clip_model is None or clip_preprocess is None:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        else:
            self.clip_model = clip_model
            self.clip_preprocess = clip_preprocess

        self.clip_model.eval()
        self.text_features = self._encode_text_prompts(self.prompts)

    @staticmethod
    def _load_style_prompts(style_txt_path: Path) -> List[str]:
        with style_txt_path.open("r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        if not prompts:
            raise ValueError(f"No prompts found in {style_txt_path}")
        return prompts

    def _encode_text_prompts(self, prompts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            text_tokens = clip.tokenize(prompts).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def predict(
        self,
        image: Union[str, Path, Image.Image],
        topk_styles: int = 5,
    ) -> List[Tuple[str, float]]:
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        image_input = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity_probs = (100.0 * image_features @ self.text_features.T.to(image_features.dtype)).softmax(dim=-1)[0]

        style_scores: Dict[str, float] = {}
        for idx, style in enumerate(self.style_names):
            style_scores.setdefault(style, 0.0)
            style_scores[style] += float(similarity_probs[idx].item())

        sorted_styles = sorted(style_scores.items(), key=lambda x: x[1], reverse=True)
        if topk_styles is not None:
            sorted_styles = sorted_styles[:topk_styles]

        return sorted_styles

    def save_text_features(self, path: Union[str, Path]):
        path = Path(path)
        torch.save(
            {
                "prompts": self.prompts,
                "style_names": self.style_names,
                "text_features": self.text_features.cpu(),
            },
            path,
        )

    @classmethod
    def from_cached_features(
        cls,
        cache_path: Union[str, Path],
        device: str = DEVICE,
        clip_model: Optional[torch.nn.Module] = None,
        clip_preprocess=None,
    ):
        cache_path = Path(cache_path)
        data = torch.load(cache_path, map_location="cpu")

        obj = cls.__new__(cls)
        obj.device = device
        obj.style_txt_path = None

        obj.prompts = data["prompts"]
        obj.style_names = data["style_names"]
        obj.text_features = data["text_features"].to(device)

        if clip_model is None or clip_preprocess is None:
            obj.clip_model, obj.clip_preprocess = clip.load("ViT-B/32", device=device)
        else:
            obj.clip_model = clip_model
            obj.clip_preprocess = clip_preprocess

        obj.clip_model.eval()
        return obj
