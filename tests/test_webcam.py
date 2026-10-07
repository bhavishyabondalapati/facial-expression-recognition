"""Tests for the webcam demo's helper functions (no camera needed)."""
import numpy as np
import pytest

from src.config import CLASSES
from src.webcam import Smoother, describe, preprocess_face


def test_preprocess_face_shape_and_range():
    gray = np.random.default_rng(0).integers(0, 256, (480, 640), dtype=np.uint8)
    x = preprocess_face(gray, (100, 50, 200, 200))
    assert tuple(x.shape) == (1, 1, 48, 48)
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0


def test_preprocess_face_margin_stays_inside_frame():
    gray = np.zeros((100, 100), dtype=np.uint8)
    x = preprocess_face(gray, (0, 0, 100, 100), margin=0.5)   # box would go off-frame
    assert tuple(x.shape) == (1, 1, 48, 48)


def test_smoother_moves_gradually_toward_new_value():
    s = Smoother(alpha=0.5)
    a, b = np.eye(8)[0], np.eye(8)[1]
    assert np.allclose(s.update(a), a)                    # first frame passes through
    mixed = s.update(b)
    assert mixed[0] == pytest.approx(0.5) and mixed[1] == pytest.approx(0.5)
    assert mixed.sum() == pytest.approx(1.0)


def test_describe_flags_uncertain_predictions():
    confident = np.array([0.05, 0.85, 0.05, 0.05, 0, 0, 0, 0])
    unsure = np.array([0.40, 0.05, 0.05, 0.35, 0.05, 0.05, 0.05, 0])
    assert describe(confident) == f"{CLASSES[1]} 85%"
    assert describe(unsure).startswith("uncertain: neutral 40% / sadness 35%")
