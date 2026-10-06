"""Load FER2013 + FER+ votes and turn them into hard and soft labels.

Both label types are built from the SAME set of images, so a model trained
on hard labels and one trained on soft labels differ only in their targets.
"""
import numpy as np
import pandas as pd

from src.config import CLASSES, FER2013_TO_NAME, IMG_SIZE, PROCESSED_NPZ, SPLITS


def verify_alignment(fer: pd.DataFrame, ferplus: pd.DataFrame) -> dict:
    """Check that fer2013.csv and fer2013new.csv line up row by row.

    FER+ has no copy of the pixels; its votes are matched to images purely
    by row position, so a reordered mirror would silently mislabel faces.
    Raises ValueError if the files don't match.
    """
    if len(fer) != len(ferplus):
        raise ValueError(f"Row counts differ: {len(fer)} vs {len(ferplus)}")
    mismatched = int((fer["Usage"].values != ferplus["Usage"].values).sum())
    if mismatched:
        raise ValueError(f"Usage column differs on {mismatched} rows")

    # Extra sanity check: the original FER2013 label should usually agree
    # with the FER+ crowd's top vote. If rows were shifted, this falls to chance.
    top_vote = ferplus[CLASSES].idxmax(axis=1)
    original = fer["emotion"].map(FER2013_TO_NAME)
    has_votes = ferplus[CLASSES].sum(axis=1) > 0
    agreement = float((original == top_vote)[has_votes].mean())
    shifted = float((original.shift(1) == top_vote)[has_votes].mean())
    return {"rows": len(fer), "label_agreement": agreement,
            "label_agreement_if_shifted": shifted}


def keep_mask(ferplus: pd.DataFrame) -> np.ndarray:
    """Rows we keep: have an image name, NF isn't the top vote, and at
    least one vote went to one of the 8 emotions."""
    has_image = ferplus["Image name"].notna().values
    others = ferplus[CLASSES + ["unknown"]].max(axis=1).values
    nf_top = ferplus["NF"].values >= others   # ties with NF also count as "not a face"
    has_emotion_votes = ferplus[CLASSES].sum(axis=1).values > 0
    return has_image & ~nf_top & has_emotion_votes


def soft_labels(votes: np.ndarray) -> np.ndarray:
    """Turn vote counts (N, 8) into probability distributions that sum to 1."""
    votes = votes.astype(np.float32)
    return votes / votes.sum(axis=1, keepdims=True)


def hard_labels(votes: np.ndarray, seed: int) -> np.ndarray:
    """Majority-vote label per image. Ties are broken by a seeded random
    choice among the tied classes, so results are reproducible."""
    rng = np.random.default_rng(seed)
    is_max = votes == votes.max(axis=1, keepdims=True)
    # Random score per class; only tied-for-max classes are eligible.
    scores = np.where(is_max, rng.random(votes.shape), -1.0)
    return scores.argmax(axis=1)


def entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy (in bits) of each row. 0 = all annotators agree,
    3 = votes spread evenly over all 8 classes."""
    p = np.clip(probs, 1e-12, 1.0)
    return -(probs * np.log2(p)).sum(axis=1)


def parse_pixels(pixels: pd.Series) -> np.ndarray:
    """Space-separated pixel strings -> uint8 array of shape (N, 48, 48)."""
    flat = np.array([np.array(s.split(), dtype=np.uint8) for s in pixels])
    return flat.reshape(-1, IMG_SIZE, IMG_SIZE)


def build_dataset(fer: pd.DataFrame, ferplus: pd.DataFrame, seed: int) -> dict:
    """Combine both CSVs into arrays ready for training."""
    keep = keep_mask(ferplus)
    votes = ferplus.loc[keep, CLASSES].values.astype(np.int64)
    split_name = {v: k for k, v in SPLITS.items()}
    soft = soft_labels(votes)
    return {
        "images": parse_pixels(fer.loc[keep, "pixels"]),
        "votes": votes,
        "soft": soft,
        "hard": hard_labels(votes, seed),
        "is_tie": (votes == votes.max(axis=1, keepdims=True)).sum(axis=1) > 1,
        "entropy": entropy(soft),
        "split": ferplus.loc[keep, "Usage"].map(split_name).values.astype("U5"),
        "row": np.flatnonzero(keep),   # row in the original CSVs
    }


def load_processed(path=PROCESSED_NPZ) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def get_split(data: dict, split: str) -> dict:
    m = data["split"] == split
    return {k: v[m] for k, v in data.items()}
