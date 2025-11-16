from pathlib import Path
from typing import Union, Optional, Dict, List

import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from torchvision import transforms

from .clip_style import ClipStylePredictor
from .blip_caption import BlipCaptioner

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ImageAnalysisPipeline:
    """
    Unified pipeline:
      - CLIP style prediction
      - BLIP captioning
      - SuSy authenticity/source classification
    """

    def __init__(
        self,
        style_txt_path: Union[str, Path, None] = None,
        style_features_cache: Optional[Union[str, Path]] = None,
        susy_repo_id: str = "HPAI-BSC/SuSy",
        susy_filename: str = "SuSy.pt",
        device: str = DEVICE,
    ):
        self.device = device

         # ----- CLIP Style Predictor -----
        if style_features_cache is not None:
            cache_path = Path(style_features_cache)
        else:
            # load bundled cache from package
            cache_path = Path(__file__).parent / "data" / "style_clip_features.pt"

        if cache_path.exists():
            self.style_predictor = ClipStylePredictor.from_cached_features(
                cache_path,
                device=device,
            )
        else:
            # fallback: build from style.txt if cache missing
            if style_txt_path is None:
                style_txt_path = Path(__file__).parent / "data" / "style.txt"
            self.style_predictor = ClipStylePredictor(
                style_txt_path,
                device=device,
            )

        # ---------- BLIP Captioner ----------
        self.captioner = BlipCaptioner(device=device)

        # ---------- SuSy ----------
        model_path = hf_hub_download(repo_id=susy_repo_id, filename=susy_filename)
        self.susy_model = torch.jit.load(model_path, map_location=device)
        self.susy_model.eval()

        self.susy_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.susy_class_names: List[str] = [
            "authentic",
            "dalle-3-images",
            "diffusiondb",
            "midjourney-images",
            "midjourney_tti",
            "realisticSDXL",
        ]

    # ---------- internal helpers ----------

    @staticmethod
    def _to_pil(image: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        return image.convert("RGB")

    def _predict_susy(self, pil_img: Image.Image) -> Dict[str, object]:
        patch = self.susy_preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.susy_model(patch)
            probs = torch.softmax(logits, dim=1)[0]

        pred_idx = int(probs.argmax().item())
        pred_class = self.susy_class_names[pred_idx]

        prob_dict = {
            name: float(probs[i]) for i, name in enumerate(self.susy_class_names)
        }

        return {
            "pred_class": pred_class,
            "probs": prob_dict,
        }

    # ---------- public API ----------

    def analyze(
        self,
        image: Union[str, Path, Image.Image],
        topk_styles: int = 5,
        caption_prompt: Optional[str] = None,
    ) -> Dict[str, object]:
        pil_img = self._to_pil(image)

        styles = self.style_predictor.predict(pil_img, topk_styles=topk_styles)
        caption = self.captioner.caption(pil_img, prompt=caption_prompt)
        susy_result = self._predict_susy(pil_img)

        return {
            "styles": styles,
            "caption": caption,
            "susy": susy_result,
        }
