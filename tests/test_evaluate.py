"""Tests for the evaluation metrics (small hand-made arrays)."""
import numpy as np
import pytest

from src.evaluate import bucket_of, correct_any_top, expected_calibration_error, kl_per_image


def test_kl_is_zero_when_model_matches_votes():
    human = np.array([[0.6, 0.4, 0, 0, 0, 0, 0, 0]])
    assert kl_per_image(human, human)[0] == pytest.approx(0.0, abs=1e-9)


def test_kl_punishes_overconfidence_on_split_votes():
    human = np.array([[0.5, 0.5, 0, 0, 0, 0, 0, 0]])
    sure = np.array([[0.99, 0.01, 0, 0, 0, 0, 0, 0]])
    hedged = np.array([[0.55, 0.45, 0, 0, 0, 0, 0, 0]])
    assert kl_per_image(human, sure)[0] > kl_per_image(human, hedged)[0]


def test_correct_any_top_accepts_either_tied_class():
    votes = np.array([[5, 5, 0, 0, 0, 0, 0, 0], [5, 5, 0, 0, 0, 0, 0, 0], [2, 8, 0, 0, 0, 0, 0, 0]])
    preds = np.array([0, 1, 0])
    assert correct_any_top(votes, preds).tolist() == [True, True, False]


def test_ece_zero_when_confidence_equals_accuracy():
    probs = np.tile([0.75, 0.25, 0, 0, 0, 0, 0, 0], (4, 1))
    correct = np.array([True, True, True, False])   # 75% right at 75% confidence
    assert expected_calibration_error(probs, correct) == pytest.approx(0.0)


def test_ambiguity_buckets():
    assert bucket_of(np.array([1.0, 0.8, 0.7, 0.6, 0.5, 0.3])).tolist() == [
        "clear (8-10/10)", "clear (8-10/10)", "medium (6-7/10)", "medium (6-7/10)",
        "ambiguous (≤5/10)", "ambiguous (≤5/10)"]
