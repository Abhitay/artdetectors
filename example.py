# example.py
from artdetectors import ImageAnalysisPipeline

# load pipeline (auto-loads CLIP, BLIP, SuSy, SDXL if enabled)
pipe = ImageAnalysisPipeline()

# 1) Fast detectors only
det = pipe.analyze("input.jpg")
print("\n=== DETECTORS ===")
print("styles:", det["styles"])
print("caption:", det["caption"])
print("susy:", det["susy"])

# 2) Slow restyling
out = pipe.restyle_image(
    "input.jpg",
    target_style="impressionism",
    strength=0.5,
    guidance_scale=5.0,
    num_inference_steps=18,
)
out["restyled_image"].save("test_ukiyoe.png")
print("\nSaved new image: test_ukiyoe.png")
