# config.py (nuevo archivo)
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).parent.parent.resolve()
    DATA_DIR = BASE_DIR / "data"
    DATASET_DIR = BASE_DIR / "datasets" / "OpenLogo-Dataset"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    YOLOV5_DIR = BASE_DIR / "yolov5"
    MODELS_DIR = BASE_DIR / "models"