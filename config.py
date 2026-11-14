"""Shared configuration for the Plant Health Diagnostic project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"                       # data/<class_name>/*.jpg
MODEL_DIR = ROOT / "models"
CHECKPOINT = MODEL_DIR / "plant_health_effnet_b0.pt"

IMG_SIZE = 224
# ImageNet statistics (required because the backbone was pre-trained on ImageNet)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Below this top-1 probability the app flags the result as "low confidence".
LOW_CONFIDENCE = 0.50
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
