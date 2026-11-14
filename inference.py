"""Inference + diagnostic engine.

CLI:   python inference.py path/to/plant.jpg
Code:  diagnoser = PlantDiagnoser(); result = diagnoser.predict("plant.jpg")
"""
import io
import sys
from pathlib import Path
from typing import Union

import torch
from PIL import Image

from config import CHECKPOINT, LOW_CONFIDENCE
from dataset import get_transforms
from knowledge_base import get_recommendations
from model import build_model

ImageInput = Union[str, Path, bytes, Image.Image]


class PlantDiagnoser:
    def __init__(self, checkpoint: Union[str, Path] = CHECKPOINT, device: str = None):
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint}. Run `python train.py` first.")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint, map_location=self.device)
        self.class_names = ckpt["class_names"]
        self.val_acc = ckpt.get("val_acc")
        self.model = build_model(len(self.class_names), pretrained=False)  # weights come from ckpt
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.transform = get_transforms(train=False)

    @staticmethod
    def _load_image(image: ImageInput) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (bytes, bytearray)):
            return Image.open(io.BytesIO(image)).convert("RGB")
        return Image.open(image).convert("RGB")

    @torch.no_grad()
    def predict(self, image: ImageInput, top_k: int = 3) -> dict:
        img = self._load_image(image)
        x = self.transform(img).unsqueeze(0).to(self.device)
        # Light test-time augmentation: average original + horizontal flip
        logits = (self.model(x) + self.model(torch.flip(x, dims=[3]))) / 2
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

        k = min(top_k, len(self.class_names))
        top_p, top_i = probs.topk(k)
        top = [{"label": self.class_names[i], "confidence": float(p)}
               for p, i in zip(top_p, top_i)]
        all_probs = {name: float(p) for name, p in zip(self.class_names, probs)}

        primary = top[0]
        margin = top[0]["confidence"] - (top[1]["confidence"] if k > 1 else 0.0)
        warnings = []
        if primary["confidence"] < LOW_CONFIDENCE:
            warnings.append("Low confidence: retake the photo (good light, plain background, "
                            "affected leaf in focus) and confirm with a soil or tissue test.")
        elif margin < 0.15:
            warnings.append(f"Close call between '{top[0]['label']}' and '{top[1]['label']}'. "
                            "Treat the diagnosis as provisional.")

        return {
            "primary_label": primary["label"],
            "primary_confidence": primary["confidence"],
            "top_k": top,
            "all_probabilities": all_probs,
            "warnings": warnings,
            "recommendations": get_recommendations(primary["label"]),
        }


def print_report(result: dict) -> None:
    rec = result["recommendations"]
    print(f"\nDiagnosis: {rec['title']}  ({result['primary_confidence']:.1%} confidence)")
    print(f"Severity : {rec['severity']}\n{rec['summary']}\n")
    for w in result["warnings"]:
        print(f"! {w}")
    print("\nTop predictions:")
    for t in result["top_k"]:
        print(f"  {t['label']:<28} {t['confidence']:.1%}")
    for section in ("root_causes", "soil_amendments", "precautions", "prevention"):
        if rec[section]:
            print(f"\n{section.replace('_', ' ').title()}:")
            for item in rec[section]:
                print(f"  - {item}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python inference.py <image_path>")
    print_report(PlantDiagnoser().predict(sys.argv[1]))
