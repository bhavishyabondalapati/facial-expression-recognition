"""Tests for label building. They use tiny hand-made tables, so they run
without the real dataset."""
import numpy as np
import pandas as pd
import pytest

from src.config import CLASSES
from src.data import (build_dataset, entropy, hard_labels, keep_mask,
                      soft_labels, verify_alignment)


def make_ferplus(rows):
    """rows: list of (usage, image_name, votes_dict)."""
    records = []
    for usage, name, votes in rows:
        r = {"Usage": usage, "Image name": name}
        r.update({c: 0 for c in CLASSES + ["unknown", "NF"]})
        r.update(votes)
        records.append(r)
    return pd.DataFrame(records)


def make_fer(usages, emotions=None):
    n = len(usages)
    return pd.DataFrame({
        "emotion": emotions if emotions is not None else [6] * n,
        "pixels": [" ".join(["128"] * 48 * 48)] * n,
        "Usage": usages,
    })


def test_soft_labels_sum_to_one():
    votes = np.array([[4, 0, 0, 1, 3, 2, 0, 0], [10, 0, 0, 0, 0, 0, 0, 0]])
    soft = soft_labels(votes)
    assert np.allclose(soft.sum(axis=1), 1.0)
    assert np.allclose(soft[0], [0.4, 0, 0, 0.1, 0.3, 0.2, 0, 0])


def test_hard_label_is_majority_when_no_tie():
    votes = np.array([[1, 7, 2, 0, 0, 0, 0, 0]])
    assert hard_labels(votes, seed=0)[0] == 1


def test_tie_break_picks_a_tied_class_and_is_reproducible():
    votes = np.tile([5, 0, 5, 0, 0, 0, 0, 0], (200, 1))
    a = hard_labels(votes, seed=0)
    b = hard_labels(votes, seed=0)
    assert set(a) == {0, 2}               # only tied classes, and both get picked
    assert np.array_equal(a, b)           # same seed -> same labels


def test_keep_mask_drops_missing_image_and_not_a_face():
    df = make_ferplus([
        ("Training", "fer0.png", {"happiness": 10}),
        ("Training", None, {"NF": 10}),                     # no image
        ("Training", "fer1.png", {"NF": 6, "neutral": 4}),  # mostly not-a-face
        ("Training", "fer2.png", {"NF": 5, "neutral": 5}),  # tie with NF -> dropped
        ("Training", "fer3.png", {"unknown": 6, "fear": 4}),  # kept: has emotion votes
    ])
    assert keep_mask(df).tolist() == [True, False, False, False, True]


def test_entropy_bounds():
    agree = np.eye(8)[:1]
    uniform = np.full((1, 8), 1 / 8)
    assert entropy(agree)[0] == pytest.approx(0.0)
    assert entropy(uniform)[0] == pytest.approx(3.0)


def test_verify_alignment_raises_on_usage_mismatch():
    ferplus = make_ferplus([("Training", "a", {"neutral": 10}),
                            ("PublicTest", "b", {"neutral": 10})])
    fer = make_fer(["Training", "PrivateTest"])
    with pytest.raises(ValueError):
        verify_alignment(fer, ferplus)


def test_verify_alignment_raises_on_row_count():
    ferplus = make_ferplus([("Training", "a", {"neutral": 10})])
    with pytest.raises(ValueError):
        verify_alignment(make_fer(["Training", "Training"]), ferplus)


def test_build_dataset_hard_and_soft_cover_same_images():
    ferplus = make_ferplus([
        ("Training", "a", {"happiness": 6, "surprise": 4}),
        ("Training", None, {"NF": 10}),
        ("PublicTest", "b", {"anger": 5, "disgust": 5}),
        ("PrivateTest", "c", {"contempt": 7, "neutral": 3}),
    ])
    fer = make_fer(ferplus["Usage"].tolist())
    d = build_dataset(fer, ferplus, seed=0)
    n = len(d["row"])
    assert n == 3
    assert all(len(d[k]) == n for k in ["images", "soft", "hard", "split", "entropy"])
    assert d["images"].shape == (3, 48, 48)
    assert d["row"].tolist() == [0, 2, 3]
    assert d["split"].tolist() == ["train", "val", "test"]
    assert d["hard"][2] == CLASSES.index("contempt")
    assert d["is_tie"].tolist() == [False, True, False]
