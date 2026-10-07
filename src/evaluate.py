"""Evaluate the 4 trained models on the test set (PrivateTest).

Every metric is reported twice: on the full test set, and with the test
images that have a pixel-identical copy in the training set removed.

Run:  python -m src.evaluate
Writes results/metrics.json, results/results.md and figures in results/figures/.
"""
import json

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from src.config import CLASSES, NUM_CLASSES, ROOT, RUNS_DIR
from src.data import get_split, load_processed
from src.models import build_model
from src.train import get_device

RUNS = ["cnn_hard", "cnn_soft", "resnet18_hard", "resnet18_soft"]
# Ambiguity buckets by how many of the 10 annotators chose the top emotion.
BUCKETS = {"clear (8-10/10)": (0.8, 1.01), "medium (6-7/10)": (0.6, 0.8), "ambiguous (≤5/10)": (0.0, 0.6)}
N_BOOT = 2000


# ---------- metrics (plain numpy, so they are easy to test) ----------

def kl_per_image(human, probs):
    """KL(human votes || model). 0 means the model reproduces the vote split."""
    h = np.clip(human, 1e-12, 1)
    return (human * (np.log(h) - np.log(np.clip(probs, 1e-12, 1)))).sum(axis=1)


def correct_any_top(votes, preds):
    """A prediction counts as correct if it's one of the most-voted emotions.
    Fair for ties, where the 'majority label' was a coin flip."""
    return votes[np.arange(len(preds)), preds] == votes.max(axis=1)


def expected_calibration_error(probs, correct, n_bins=15):
    """Gap between confidence and accuracy, averaged over confidence bins.
    0 = when the model says 70% it's right 70% of the time."""
    conf = probs.max(axis=1)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def bucket_of(top_share):
    out = np.empty(len(top_share), dtype=object)
    for name, (lo, hi) in BUCKETS.items():
        out[(top_share >= lo) & (top_share < hi)] = name
    return out


def summarize(probs, test):
    """All headline metrics for one model on one subset of the test set."""
    preds = probs.argmax(axis=1)
    strict = preds == test["hard"]
    any_top = correct_any_top(test["votes"], preds)
    kl = kl_per_image(test["soft"], probs)
    buckets = bucket_of(test["soft"].max(axis=1))
    out = {
        "n": int(len(preds)),
        "acc_majority": float(strict.mean()),
        "acc_any_top": float(any_top.mean()),
        "macro_f1": float(f1_score(test["hard"], preds, average="macro", labels=range(NUM_CLASSES))),
        "kl": float(kl.mean()),
        "ece": expected_calibration_error(probs, strict),
        "mean_confidence": float(probs.max(axis=1).mean()),
        "per_class_f1": dict(zip(CLASSES, f1_score(test["hard"], preds, average=None,
                                                    labels=range(NUM_CLASSES), zero_division=0).round(4).tolist())),
        "by_ambiguity": {},
    }
    for name in BUCKETS:
        m = buckets == name
        out["by_ambiguity"][name] = {
            "n": int(m.sum()),
            "acc_any_top": float(any_top[m].mean()),
            "kl": float(kl[m].mean()),
            "model_confidence": float(probs[m].max(axis=1).mean()),
            "human_top_share": float(test["soft"][m].max(axis=1).mean()),
        }
    return out


def paired_bootstrap(hard_probs, soft_probs, test, seed=0):
    """95% intervals for (soft - hard) by resampling test images.
    If an interval excludes 0, the difference is unlikely to be chance
    from which test images we happened to have (it says nothing about
    training randomness: we trained one seed per model)."""
    rng = np.random.default_rng(seed)
    n = len(test["hard"])
    ph, ps = hard_probs.argmax(1), soft_probs.argmax(1)
    acc_h = correct_any_top(test["votes"], ph).astype(float)
    acc_s = correct_any_top(test["votes"], ps).astype(float)
    kl_h, kl_s = kl_per_image(test["soft"], hard_probs), kl_per_image(test["soft"], soft_probs)
    diffs = {"acc_any_top": [], "kl": [], "macro_f1": []}
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        diffs["acc_any_top"].append(acc_s[i].mean() - acc_h[i].mean())
        diffs["kl"].append(kl_s[i].mean() - kl_h[i].mean())
        y = test["hard"][i]
        diffs["macro_f1"].append(
            f1_score(y, ps[i], average="macro", labels=range(NUM_CLASSES), zero_division=0)
            - f1_score(y, ph[i], average="macro", labels=range(NUM_CLASSES), zero_division=0))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in diffs.items()}


# ---------- running the models ----------

@torch.no_grad()
def predict(run, images, device):
    ckpt = torch.load(RUNS_DIR / run / "final.pt", map_location="cpu", weights_only=False)
    model = build_model(ckpt["model"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    x = torch.from_numpy(images).unsqueeze(1)
    probs = [torch.softmax(model(x[i:i + 512].to(device).float() / 255), dim=1).cpu()
             for i in range(0, len(x), 512)]
    return torch.cat(probs).numpy().astype(np.float64)


def duplicate_mask(data):
    """True for test images that have a pixel-identical copy in train."""
    train_hashes = {img.tobytes() for img in data["images"][data["split"] == "train"]}
    test_imgs = data["images"][data["split"] == "test"]
    return np.array([img.tobytes() in train_hashes for img in test_imgs])


def subset(d, mask):
    return {k: v[mask] for k, v in d.items()}


def main():
    from src import report   # tables + figures

    data = load_processed()
    test = get_split(data, "test")
    dup = duplicate_mask(data)
    subsets = {"full": np.ones(len(dup), bool), "dedup": ~dup}
    print(f"Test images: {len(dup)} | with a duplicate in train: {dup.sum()} | dedup set: {(~dup).sum()}")

    device = get_device()
    probs = {run: predict(run, test["images"], device) for run in RUNS}

    results = {"n_duplicates_removed": int(dup.sum()), "metrics": {}, "bootstrap_soft_minus_hard": {}}
    for sname, m in subsets.items():
        t = subset(test, m)
        results["metrics"][sname] = {run: summarize(probs[run][m], t) for run in RUNS}
        results["bootstrap_soft_minus_hard"][sname] = {
            arch: paired_bootstrap(probs[f"{arch}_hard"][m], probs[f"{arch}_soft"][m], t)
            for arch in ["cnn", "resnet18"]}

    results["confusion_full"] = {
        run: confusion_matrix(test["hard"], probs[run].argmax(1), labels=range(NUM_CLASSES)).tolist()
        for run in RUNS}
    (ROOT / "results" / "metrics.json").write_text(json.dumps(results, indent=2))
    report.write_markdown(results)
    report.make_figures(results, probs, test, dup)
    print((ROOT / "results" / "results.md").read_text())


if __name__ == "__main__":
    main()
