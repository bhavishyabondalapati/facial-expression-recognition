"""Shared paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FER2013_CSV = RAW_DIR / "fer2013.csv"
FERPLUS_CSV = RAW_DIR / "fer2013new.csv"
PROCESSED_NPZ = PROCESSED_DIR / "ferplus.npz"
FIGURES_DIR = ROOT / "results" / "figures"
RUNS_DIR = ROOT / "runs"

# The 8 FER+ emotion classes, in the column order of fer2013new.csv.
CLASSES = ["neutral", "happiness", "surprise", "sadness",
           "anger", "disgust", "fear", "contempt"]
NUM_CLASSES = len(CLASSES)

# Original FER2013 label ids -> FER+ class names (FER2013 has no contempt).
FER2013_TO_NAME = {0: "anger", 1: "disgust", 2: "fear", 3: "happiness",
                   4: "sadness", 5: "surprise", 6: "neutral"}

SPLITS = {"train": "Training", "val": "PublicTest", "test": "PrivateTest"}
IMG_SIZE = 48
TIE_BREAK_SEED = 0
