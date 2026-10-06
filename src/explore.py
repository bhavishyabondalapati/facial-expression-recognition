"""Explore the processed FER+ data: class balance, how much annotators
disagree, and the contempt class. Saves figures to results/figures/ and a
summary table to results/data_summary.md.

Run:  python -m src.explore
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import CLASSES, FIGURES_DIR, ROOT
from src.data import get_split, load_processed

# Chart colors: a validated colorblind-safe palette (blue, orange) + neutral inks.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def style_grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_class_counts(train):
    """Training images per class (majority label). Shows the imbalance."""
    counts = np.bincount(train["hard"], minlength=len(CLASSES))
    order = np.argsort(counts)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(np.array(CLASSES)[order], counts[order], color=BLUE, height=0.6)
    for i, c in enumerate(counts[order]):
        ax.text(c + 120, i, f"{c:,}", va="center", color=INK, fontsize=9)
    ax.set_xlabel("Training images (majority-vote label)")
    ax.set_title("Training set is imbalanced; contempt is tiny", loc="left")
    style_grid(ax)
    fig.savefig(FIGURES_DIR / "class_counts.png")
    plt.close(fig)


def plot_hard_vs_soft_mass(train):
    """How much total label 'weight' each class gets under each scheme.
    Soft labels spread weight onto minority emotions."""
    hard = np.bincount(train["hard"], minlength=len(CLASSES)) / len(train["hard"])
    soft = train["soft"].mean(axis=0)
    y = np.arange(len(CLASSES))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(y + 0.19, hard * 100, height=0.36, color=BLUE, label="Hard (majority vote)")
    ax.barh(y - 0.19, soft * 100, height=0.36, color=ORANGE, label="Soft (vote share)")
    ax.set_yticks(y, CLASSES)
    ax.invert_yaxis()
    ax.set_xlabel("Share of training label weight (%)")
    ax.set_title("Soft labels give more weight to minority emotions", loc="left")
    ax.legend(frameon=False, loc="lower right")
    style_grid(ax)
    fig.savefig(FIGURES_DIR / "hard_vs_soft_label_mass.png")
    plt.close(fig)


def plot_ambiguity(data):
    """How many of the 10 annotators chose each image's top emotion."""
    top_votes = np.rint(data["soft"].max(axis=1) * 10).astype(int)
    k = np.arange(2, 11)
    counts = np.array([(top_votes == v).sum() for v in k])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(k, counts, color=BLUE, width=0.7)
    ax.set_xticks(k, [f"{v}/10" for v in k])
    ax.set_xlabel("Annotators who chose the top emotion")
    ax.set_ylabel("Images")
    ax.set_title("Only about 1 in 4 faces gets a unanimous vote", loc="left")
    style_grid(ax, axis="y")
    fig.savefig(FIGURES_DIR / "ambiguity_hist.png")
    plt.close(fig)


def plot_agreement_by_class(train):
    """Average top-vote share for images of each majority class."""
    top_share = train["soft"].max(axis=1)
    means = np.array([top_share[train["hard"] == k].mean() for k in range(len(CLASSES))])
    order = np.argsort(means)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(np.array(CLASSES)[order], means[order] * 100, color=BLUE, height=0.6)
    for i, m in enumerate(means[order]):
        ax.text(m * 100 + 1, i, f"{m:.0%}", va="center", color=INK, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Average annotator agreement with the majority label (%)")
    ax.set_title("Disgust, contempt and fear are the least-agreed emotions", loc="left")
    style_grid(ax)
    fig.savefig(FIGURES_DIR / "agreement_by_class.png")
    plt.close(fig)


def plot_examples(train, per_class=6, seed=0):
    """Clear (top) vs ambiguous (bottom) examples for each class, with votes."""
    rng = np.random.default_rng(seed)
    top_share = train["soft"].max(axis=1)
    fig, axes = plt.subplots(len(CLASSES), per_class, figsize=(per_class * 1.5, len(CLASSES) * 1.75))
    for k, name in enumerate(CLASSES):
        is_k = train["hard"] == k
        clear = np.flatnonzero(is_k & (top_share >= 0.8))
        ambiguous = np.flatnonzero(is_k & (top_share <= 0.5))
        half = per_class // 2
        picks = list(rng.choice(clear, min(half, len(clear)), replace=False)) + \
            list(rng.choice(ambiguous, min(per_class - half, len(ambiguous)), replace=False))
        for j in range(per_class):
            ax = axes[k, j]
            ax.axis("off")
            if j >= len(picks):
                continue
            i = picks[j]
            ax.imshow(train["images"][i], cmap="gray", vmin=0, vmax=255)
            top2 = np.argsort(train["votes"][i])[::-1][:2]
            caption = "\n".join(f"{CLASSES[c][:5]} {train['votes'][i][c]}"
                                for c in top2 if train["votes"][i][c] > 0)
            ax.set_title(caption, fontsize=7, color=INK_2)
        axes[k, 0].text(-8, 24, name, ha="right", va="center", fontsize=10, color=INK,
                        fontweight="bold")
    fig.suptitle("Left 3: clear (≥8/10 agree)    Right 3: ambiguous (≤5/10 agree)",
                 color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "examples_clear_vs_ambiguous.png")
    plt.close(fig)


def write_summary(data):
    splits = {s: get_split(data, s) for s in ["train", "val", "test"]}
    lines = ["# Data summary", "",
             "Generated by `python -m src.explore`.", "",
             "## Images per class (majority-vote label)", "",
             "| class | train | val | test |", "|---|---:|---:|---:|"]
    for k, name in enumerate(CLASSES):
        row = [str((splits[s]["hard"] == k).sum()) for s in splits]
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    lines.append("| **total** | " + " | ".join(str(len(splits[s]["hard"])) for s in splits) + " |")

    top_share = data["soft"].max(axis=1)
    pix = data["images"].reshape(len(data["images"]), -1)
    train_hashes = {p.tobytes() for p in pix[data["split"] == "train"]}
    leaks = {s: sum(p.tobytes() in train_hashes for p in pix[data["split"] == s])
             for s in ["val", "test"]}
    contempt = data["hard"] == CLASSES.index("contempt")

    lines += ["", "## Ambiguity", "",
              f"- Hard-label ties broken by seeded random choice: {data['is_tie'].sum():,} "
              f"({data['is_tie'].mean():.1%} of images)",
              f"- Images where 10/10 annotators agree: {(top_share == 1).sum():,} "
              f"({(top_share == 1).mean():.1%})",
              f"- Images where the top emotion got 5 or fewer of 10 votes: {(top_share <= 0.5).sum():,} "
              f"({(top_share <= 0.5).mean():.1%})",
              f"- Mean vote entropy: {data['entropy'].mean():.2f} bits (0 = unanimous, 3 = uniform)",
              "", "## Contempt", "",
              f"- Majority-contempt images: {contempt.sum()} of {len(contempt):,} "
              f"({contempt.mean():.2%}); train {(splits['train']['hard'] == 7).sum()}, "
              f"test {(splits['test']['hard'] == 7).sum()}",
              f"- Images with at least one contempt vote: {(data['votes'][:, 7] > 0).sum():,}",
              f"- Average agreement on majority-contempt images: {top_share[contempt].mean():.0%}",
              "", "## Duplicates across splits", "",
              f"- Val images with pixel-identical copies in train: {leaks['val']}",
              f"- Test images with pixel-identical copies in train: {leaks['test']}",
              "", "FER2013 contains some exact duplicate images. We keep the official",
              "splits (so results compare to published numbers) but note the overlap."]
    (ROOT / "results" / "data_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = load_processed()
    train = get_split(data, "train")
    plot_class_counts(train)
    plot_hard_vs_soft_mass(train)
    plot_ambiguity(data)
    plot_agreement_by_class(train)
    plot_examples(train)
    write_summary(data)
    print(f"\nFigures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
